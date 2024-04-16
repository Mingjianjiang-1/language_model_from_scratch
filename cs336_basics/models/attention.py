from cs336_basics.utils import scaled_dot_product_attention
import torch

class CausalSelfMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None):
        super(CausalSelfMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.attn_pdrop = attn_pdrop
        self.query = torch.nn.Linear(d_model, d_model, bias=False)
        self.key = torch.nn.Linear(d_model, d_model, bias=False)
        self.value = torch.nn.Linear(d_model, d_model, bias=False)
        self.output = torch.nn.Linear(d_model, d_model, bias=False)
        self.dropout = torch.nn.Dropout(attn_pdrop) if attn_pdrop is not None else torch.nn.Dropout(0.0)
  
    def forward(self, x):
        # x.shape (*, seq_len, d_model)
        batches, seq_len = x.shape[:-2], x.shape[-2]
        # Split the input into num_heads
        query, key, value = self.query(x), self.key(x), self.value(x)  # (*, seq_len, d_model)
        query = query.view(*query.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)  # (*, num_heads, seq_len, d_k)
        key = key.view(*key.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)  # (*, num_heads, seq_len, d_k)
        value = value.view(*value.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)  # (*, num_heads, seq_len, d_k)
        
        # Causal mask for the attention
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device).to(torch.bool)  # (seq_len, seq_len)
        attention = scaled_dot_product_attention(key, query, value, mask)  # (*, num_heads, seq_len, d_k)
        # attention = attention.transpose(-3, -2).contiguous().view(*batches, seq_len, self.d_model)  # (*, seq_len, d_model)
        attention = attention.transpose(-3, -2).contiguous().view(*batches, seq_len, self.d_model) # (*, seq_len, d_model)
        
        attention = self.dropout(attention)
        output = self.output(attention)  # (*, seq_len, d_model)
        return output
    
 
    