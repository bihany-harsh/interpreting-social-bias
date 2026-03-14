# this code is borrowed and re-purposed from Andrej Karpathy's build-nanogpt tutorial
# https://github.com/karpathy/build-nanogpt

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x, return_neurons=False, patched_mlp_activation=None, target_positions=None):
        x = self.c_fc(x)
        inter_x = self.gelu(x)
        
        if patched_mlp_activation is not None and target_positions is not None:
            for pos in target_positions:
                inter_x[:, pos, :] = patched_mlp_activation
        
        out = self.c_proj(inter_x)
        if return_neurons:
            return out, inter_x
        else:
            return out

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_neurons=False, patched_mlp_activation=None, target_positions=None):
        x = x + self.attn(self.ln_1(x))
        if return_neurons or patched_mlp_activation is not None:
            mlp_out, mlp_inter_neurons = self.mlp(self.ln_2(x), return_neurons=True, patched_mlp_activation=patched_mlp_activation, target_positions=target_positions)
            x = x + mlp_out
            if return_neurons:
                return x, mlp_inter_neurons
            else:
                return x
        else:
            x = x + self.mlp(self.ln_2(x))
            return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, target_layer=None, return_neurons=False, patched_mlp_activation=None, target_positions=None, target_label=None):
        
        if patched_mlp_activation is not None:
            batch_size = patched_mlp_activation.shape[0]
            idx = idx.repeat(batch_size, 1)
            if targets is not None:
                targets = targets.repeat(batch_size, 1)
        
        
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        
        mlp_inter_neurons = None
        
        for block_idx, block in enumerate(self.transformer.h):
            if target_layer is not None and target_layer == block_idx:
                if patched_mlp_activation is not None:
                    # we need to do the activation patching
                    x = block(x, patched_mlp_activation=patched_mlp_activation, target_positions=target_positions)
                elif return_neurons:
                    # the normal forward pass to get the activatiojns
                    x, mlp_inter_neurons = block(x, return_neurons=True)
                else:
                    x = block(x)
                    
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # ig2 calculations
        if patched_mlp_activation is not None and target_label is not None:
            # we need to calculate the derivative of the prob of the said completion wrt the patched completion
            log_probs = F.log_softmax(logits, dim=-1)
            sum_log_probs = torch.zeros(B, device=logits.device)
            for i, pos in enumerate(target_positions):
                sum_log_probs += log_probs[:, pos, target_label[i]]
            
            gradient = torch.autograd.grad(torch.unbind(sum_log_probs), patched_mlp_activation)
            return sum_log_probs, gradient[0] # gradient is a tuple wrt all inputs (of which we have only-1)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if return_neurons:
            return logits, loss, mlp_inter_neurons
        else:
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


# Claude Sonnet-4.6 generated code (prompted) and subsequently edited
def gpt2_generate(model, x, gen_len, max_seq_length, target_layer=None, return_neurons=False, pooling='mean'):
    """
    args and other comments:
        model: autoregressive model
        x: input_ids (B, T)
        gen_len: the number of tokens to generate for each sequence in the batch
        max_seq_length
        target_layer: layer for which FFN activations need to be retrieved
        return_neurons: flag whether to return neurons
        pooling: ("mean" or "max") pooling is applied over the generated tokens corresponding neurons and the returned pooled_neurons is of size (B, 4*d_model)
    """
    
    # x: (B, T)
    log_probs_list = []
    generated_tokens = []
    neurons_list = []
    input_tokens = x                                                # (B, T)

    while len(generated_tokens) < gen_len and x.size(1) < max_seq_length:
        with torch.no_grad():
            if return_neurons and target_layer is not None:
                logits, _, mlp_inter_neurons = model(x, target_layer=target_layer, return_neurons=True)
                                                                    # mlp_inter_neurons: (B, T, 4*d_model)
                neurons_list.append(mlp_inter_neurons[:, -1, :])   # (B, 4*d_model) — last token only
            else:
                logits, _ = model(x)                                # (B, T, vocab_size)

            logits = logits[:, -1, :]                               # (B, vocab_size)

        log_probs = F.log_softmax(logits, dim=-1)                   # (B, vocab_size)
        probs = torch.exp(log_probs)                                # (B, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)        # (B, 1)
        token_log_prob = log_probs.gather(dim=-1, index=next_token) # (B, 1)

        log_probs_list.append(token_log_prob)
        generated_tokens.append(next_token)

        x = torch.cat([x, next_token], dim=1)                      # (B, T+i)

    generated_tokens = torch.cat(generated_tokens, dim=1)          # (B, gen_len)
    total_log_probs = torch.cat(log_probs_list, dim=1).sum(dim=-1) # (B,)
    full_sequence = torch.cat([input_tokens, generated_tokens], dim=1)  # (B, T + gen_len)

    if return_neurons and target_layer is not None:
        stacked_neurons = torch.stack(neurons_list, dim=1)          # (B, gen_len, 4*d_model)
        if pooling == 'mean':
            pooled_neurons = stacked_neurons.mean(dim=1)            # (B, 4*d_model)
        elif pooling == 'max':
            pooled_neurons = stacked_neurons.max(dim=1).values      # (B, 4*d_model)
        else:
            raise ValueError(f"Unsupported pooling type '{pooling}'. Choose 'mean' or 'max'.")
        return total_log_probs, generated_tokens, full_sequence, pooled_neurons

    return total_log_probs, generated_tokens, full_sequence
