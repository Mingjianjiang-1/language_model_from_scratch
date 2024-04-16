import torch
from .attention import CausalSelfMultiHeadAttention
from .item_layers import FeedForward, RMSNorm
from cs336_basics.utils import softmax

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None, parallel=False, layer_norm=True):
        super(TransformerBlock, self).__init__()
        self.attn = CausalSelfMultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.feedforward = FeedForward(d_model, d_ff)
        self.res_dropout = torch.nn.Dropout(residual_pdrop) if residual_pdrop is not None else torch.nn.Dropout(0.0)
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        self.parallel = parallel
    
    def forward(self, x):
        if self.parallel:
            if self.layer_norm:
                x1, x2 = self.norm1(x), self.norm2(x)
            else:
                x1, x2 = x, x
            x1 = self.res_dropout(self.attn(x1))
            x2 = self.res_dropout(self.feedforward(x2))
            x = x + x1 + x2
        else:
            if self.layer_norm:
                x = self.norm1(x)
            x = x + self.res_dropout(self.attn(x))
            if self.layer_norm:
                x = self.norm2(x)
            x = x + self.res_dropout(self.feedforward(x))
        return x

class TransformerBlockPostNorm(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None):
        super(TransformerBlockPostNorm, self).__init__()
        self.attn = CausalSelfMultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.feedforward = FeedForward(d_model, d_ff)
        self.res_dropout = torch.nn.Dropout(residual_pdrop) if residual_pdrop is not None else torch.nn.Dropout(0.0)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x):
        x = self.norm1(x + self.res_dropout(self.attn(x)))
        x = self.norm2(x + self.res_dropout(self.feedforward(x)))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None, parallel=False, layer_norm=True, pre_norm=True):
        super(Transformer, self).__init__() 
        self.word_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(context_length, d_model)
        if pre_norm:
            self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, parallel=parallel, layer_norm=layer_norm) for _ in range(num_layers)])
        else:
            self.layers = torch.nn.ModuleList([TransformerBlockPostNorm(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.dropout = torch.nn.Dropout(residual_pdrop) if residual_pdrop is not None else torch.nn.Dropout(0.0)
        self.output = torch.nn.Linear(d_model, vocab_size)

  
    def forward(self, x):
        # x.shape (batch_size, seq_len)
        seq_len = x.shape[-1]
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=x.device).unsqueeze(0)) # (1, seq_len, d_model)
        embeddings = self.word_embedding(x) + position_embeddings # (batch_size, seq_len, d_model)
        # Apply the transformer blocks
        embeddings = self.dropout(embeddings)
        for layer in self.layers:
            embeddings = layer(embeddings)
  
        embeddings = self.norm(embeddings)
        output = self.output(embeddings) # (batch_size, seq_len, vocab_size)
        
        return output
    
    def generate(self, x, max_len, temperature=0.7, top_p=0.9):
        # x.shape (1, seq_len)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.model.eval()
        generated = []
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.model(x)[:, -1, :] / temperature
                probs = softmax(logits)[0] # (vocab_size)
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff_index = (cumulative_probs > top_p).nonzero(as_tuple=True)[0].min()
                    sorted_probs[cutoff_index:] = 0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

                probs /= probs.sum()

                # Sample from the filtered distribution
                next_token = torch.multinomial(probs, 1) 
                next_token = next_token.unsqueeze(0)  # (1, 1)
                x = torch.cat([x, next_token], dim=1)  # (1, seq_len + 1)
                generated.append(next_token.item()[0])

        return generated
    
