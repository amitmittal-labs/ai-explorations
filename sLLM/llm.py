import torch
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, bias=False, dropout=0.1):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        # we will break d_out into num_heads heads, each of size head_dim = d_out // num_heads
        
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length    
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, query):
        batch_size, num_tokens, d_in = query.shape

        keys = self.W_k(query)  #shape: (batch_size, num_tokens, d_out)
        values = self.W_v(query)        
        queries = self.W_q(query)
        

        # Linear transformations
        # split d_out into num_heads and head_dim so that each head can focus on different prespectives
        #shape: (batch_size, num_heads, d_out) --> (batch_size, num_heads, num_tokens, head_dim)
        keys= keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)   
        values= values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries= queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
         
        #mask the future tokens for causal attention
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask, float('-inf')) 
        
        attention_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_vec = torch.matmul(attention_weights, values).transpose(1, 2)  #shape: (batch_size, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(batch_size, -1, self.d_out)
        
        output = self.out_proj(context_vec)
        
        return output, attention_weights