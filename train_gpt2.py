import math
import inspect
from dataclasses import dataclass
from rms_norm import RMSNorm
import torch
import torch.nn as nn
from torch.nn import functional as F

from rotary import apply_rotary_emb, precompute_freqs_cis
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.window_size = config.window_size
        
        # Group Query Attention (GQA) setup
        # n_kv_head determines number of key/value heads (can be less than query heads)
        self.n_kv_head = config.n_kv_head if hasattr(config, 'n_kv_head') else config.n_head
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        
        # Combined projection for Q, K, V:
        # - Query dims: (n_head * head_dim)           # Full number of query heads
        # - Key dims: (n_kv_head * head_dim)          # Reduced number of key heads
        # - Value dims: (n_kv_head * head_dim)        # Reduced number of value heads
        self.c_attn = nn.Linear(config.n_embd, (2 * self.n_kv_head + self.n_head) * self.head_dim)
        # Output projection
        self.c_proj = nn.Linear(
            config.n_embd,                            # Input: (B, T, n_embd) 
            config.n_embd                             # Output: (B, T, n_embd)
        )
        self.c_proj.NANOGPT_SCALE_INIT = 1           # Custom initialization scale
        
        # register attention mask buffer
        self.register_buffer("mask", None, persistent=False)

    def build_sliding_causal_mask(self, T, device):
        """Build combined sliding window and causal mask"""

        mask = torch.full((T, T), float('-inf'))
        mask = torch.triu(mask, diagonal=1) # Create causal mask (upper triangle)
        # Apply sliding window: for each position i, mask tokens before (i - window_size)
        for i in range(T):
            window_start = max(0, i - self.window_size + 1)
            mask[i, :window_start] = float('-inf')
            
        return mask

    def forward(self, x, freqs_cis, position_ids=None):
        # x shape: (batch_size, seq_length, embedding_dim)
        B, T, C = x.size()
        
        # Create or update attention mask if needed
        if self.mask is None or self.mask.size(0) != T:
            self.mask = self.build_sliding_causal_mask(T, device=x.device)
        
        # Project input to get query, key, value
        qkv = self.c_attn(x)                         # Shape: (B, T, (2*n_kv_head + n_head)*head_dim)
        
        # Split into query, key, value projections
        q, k, v = qkv.split([
            self.n_head * self.head_dim,             # Query: (B, T, n_head*head_dim)
            self.n_kv_head * self.head_dim,          # Key: (B, T, n_kv_head*head_dim)
            self.n_kv_head * self.head_dim           # Value: (B, T, n_kv_head*head_dim)
        ], dim=2)
        
        # Reshape and transpose for attention computation
        # Move head dim to proper position for parallel attention heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        
        # Apply rotary position embeddings
        q = apply_rotary_emb(q.transpose(1, 2), freqs_cis=freqs_cis).transpose(1, 2)
        k = apply_rotary_emb(k.transpose(1, 2), freqs_cis=freqs_cis).transpose(1, 2)

        # Handle Grouped Query Attention (GQA)
        # If using fewer K/V heads, repeat them to match number of query heads
        if self.n_head > self.n_kv_head:
            n_rep = self.n_head // self.n_kv_head    # Number of repetitions needed
            # Expand K/V: Add new dimension for repetition
            k = k[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_head, n_rep, T, self.head_dim)
            # Reshape to final dimensions
            k = k.reshape(B, self.n_head, T, self.head_dim)  # (B, n_head, T, head_dim)
            v = v.reshape(B, self.n_head, T, self.head_dim)  # (B, n_head, T, head_dim)

        def apply_attention(q, k, v, mask):
            """Apply scaled dot-product attention with sliding window mask
            
            Args:
                q: Query tensor (B, n_head, T, head_dim)
                k: Key tensor (B, n_head, T, head_dim)
                v: Value tensor (B, n_head, T, head_dim)
                mask: Attention mask (T, T)
                
            Returns:
                torch.Tensor: Attention output (B, n_head, T, head_dim)
            """
            # Compute attention scores
            wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_head, T, T)
            wei = wei + mask.unsqueeze(0).unsqueeze(0)  
            wei = F.softmax(wei, dim=-1)                
            return wei @ v                              

        y = apply_attention(q, k, v, self.mask)        # (B, n_head, T, head_dim)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        y = self.c_proj(y)                               # (B, T, n_embd)
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc1   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_fc2   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # silu(x)= x∗σ(x),where σ(x) is the logistic sigmoid.
        gate = F.silu(self.c_fc1(x)) 
        linear_out = self.c_fc2(x)
        # SwiGLU: element-wise multiplication of gate and linear output
        x = gate * linear_out
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    n_kv_head: int = 1  # number of key-value heads for multi-query (1) or group-query
    window_size = 512 # 1/4 of the block size

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # precomputed as contains constant value
        self.register_buffer(
            "freqs_cis", # dont treat as a learnable parameter
            precompute_freqs_cis(
                self.config.n_embd // self.config.n_head,
                # Need to compute until at least the max token limit for generation
                # (use 2x max sequence length to be safe)
                self.config.block_size * 2,
            ),
        )

        self.norm = RMSNorm(config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd) --> Replace absolute PE with ROPE
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = self.norm, # Replace LayerNorm with RMSNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):


        # freqs_cis moves with the model to to the correct device
        # precomputed when we intialise the weights
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = precompute_freqs_cis(
                self.config.n_embd // self.config.n_head,
                # Need to compute until at least the max token limit for generation
                # (use 2x max sequence length to be safe)
                self.config.block_size * 2,
            )

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings

        # ======== removed the absolute positional embedding
        # pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
       
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb 

        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[0:T]

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, freqs_cis)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------
# attempt to autodetect the device
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # micro batch size
T = 512 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)