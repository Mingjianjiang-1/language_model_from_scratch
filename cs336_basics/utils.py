import torch
import math
import numpy as np

def dict_adding(dict_, key, value):
    if key in dict_:
        dict_[key] += value
    else:
        dict_[key] = value
        
def dict_dict_add_key(dict_, pair_key, pre_token_key):
    if pair_key not in dict_:
        dict_[pair_key] = {pre_token_key: 1}
    else:
        # dict_[pair_key][pre_token_key] += 1
        dict_adding(dict_[pair_key], pre_token_key, 1)

def dict_dict_del_key(dict_, pair_key, pre_token_key):
    dict_[pair_key][pre_token_key] -= 1
    if dict_[pair_key][pre_token_key] == 0:
        del dict_[pair_key][pre_token_key]
        
def softmax(x, dim=-1):
    """
    Input: x any shape
    Output: softmax along the specified dimension
    """
    # x = x.to(torch.float64)
    max_x = torch.max(x, dim=dim, keepdim=True)[0] 
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    result = exp_x / sum_exp_x
    # result = result.to(torch.float32)
    
    return result

def scaled_dot_product_attention(key, query, value, mask=None):
    """
    Input: key, query: (batch_size, *, seq_len, d_k)
            value: (batch_size, *, seq_len, d_v)
           mask: (seq_len, seq_len)
    Output: result: (batch_size, *, seq_len, d_v)
    """
    batch_size, seq_len, d_k = key.shape[0], key.shape[-2], key.shape[-1]
    attention_logit = query @ key.transpose(-1, -2) / (d_k ** 0.5)  # (batch_size, *, seq_len, seq_len)
    if mask is not None:
        logit_mask = torch.zeros(mask.shape, device=mask.device)
        logit_mask[mask] = float('-inf')
        attention_logit += logit_mask
    softmaxed_attention = softmax(attention_logit, dim=-1) # (batch_size, *, seq_len, seq_len)
    result = softmaxed_attention @ value  # (batch_size, *, seq_len, d_v)
    return result
        
        
def cross_entropy_loss(logits, target):
    """
    Input: logits: (*, vocab_size)
           target: (*)
    Output: loss: scalar
    """
    logits = logits.view(-1, logits.shape[-1]) # (batch_size, vocab_size)
    target = target.view(-1) # (batch_size)
    loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
    # logits = logits - torch.max(logits, dim=-1, keepdim=True)[0] # (batch_size, vocab_size)
    # log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # (batch_size, vocab_size)
    # loss = -log_probs[torch.arange(target.shape[0]), target].mean()
    
    return loss

def perplexity(logits, target):
    """
    Input: logits: (*, seq_len, vocab_size)
           target: (*, seq_len)
    Output: perplexity: scalar
    """
    perplexity = torch.exp(cross_entropy_loss(logits, target))
    
    return perplexity.item()
    

def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return alpha_max * t / T_w
    elif T_w <= t <= T_c:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w)))
    else:
        return alpha_min
    
# def gradient_clip(params, max_norm, eps=1e-6):
#     """
#     Input: params: list of tensors
#            max_norm: float
#     Output: None
#     """
#     # Clip each grad in params
#     for p in params:
#         if p.grad is not None and p.grad.norm() > max_norm:
#             p.grad.data = p.grad.data / (p.grad.norm() + eps) * max_norm
            
def gradient_clip(params, max_norm, eps=1e-6):
    """
    Input: params: list of tensors
           max_norm: float
    Output: None
    """
    total_norm = 0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data / (total_norm + eps) * max_norm
            
def load_data(x, batch_size, context_length, device):
    """
    Input: x: (n,)
           batch_size: int
           context_length: int
           device: torch.device
    Output: source: (batch_size, context_length), target: (batch_size, context_length)
    """
    # x is numpy array
    rand_idx = np.random.randint(0, len(x) - context_length, batch_size) # (batch_size,)
    source = np.array([x[idx:idx+context_length] for idx in rand_idx]).astype(int) # (batch_size, context_length)
    target = np.array([x[idx+1:idx+context_length+1] for idx in rand_idx]).astype(int) # (batch_size, context_length)
    
    return torch.from_numpy(source).to(device), torch.from_numpy(target).to(device)
    
def save_checkpoint(model, optimizer, iteration, out):
    """
    should dump all the state from the first
    three parameters into the file-like object out. You can use the state_dict method of both the
    model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
    obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
    have obj be a dictionary, but you can use whatever format you want as long as you can load your
    checkpoint later.
    """
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save({"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}, out)
    
def load_checkpoint(src, model, optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model_state"])
    optimizer.load_state_dict(state_dict["optimizer_state"])
    iteration = state_dict["iteration"]
    return iteration