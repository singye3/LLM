import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a PyTorch module implementing multi-head attention mechanism.
    Attributes:
        hidden_dim (int): The dimension of the hidden layer.
        num_heads (int): The number of attention heads.
        value_weights (nn.Linear): Linear layer for value weights.
        key_weights (nn.Linear): Linear layer for key weights.
        query_weights (nn.Linear): Linear layer for query weights.
        output_weights (nn.Linear): Linear layer for output weights.
    Methods:
        check_sdpa_inputs(x):
            Checks the shape of the input tensor for scaled dot-product attention.
        scaled_dot_product_attention(query, key, value, attention_mask=None, key_padding_mask=None):
            Computes the scaled dot-product attention.
            Args:
                query (tensor): Query tensor of shape (batch_size, num_heads, query_sequence_length, hidden_dim//num_heads).
                key (tensor): Key tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads).
                value (tensor): Value tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads).
                attention_mask (tensor, optional): Attention mask tensor of shape (query_sequence_length, key_sequence_length).
                key_padding_mask (tensor, optional): Key padding mask tensor of shape (sequence_length, key_sequence_length).
            Returns:
                output (tensor): Output tensor after applying attention.
                attention (tensor): Attention weights.
        split_into_heads(x, num_heads):
            Splits the input tensor into multiple heads.
            Args:
                x (tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).
                num_heads (int): Number of attention heads.
            Returns:
                tensor: Tensor of shape (batch_size, num_heads, seq_length, hidden_dim // num_heads).
        combine_heads(x):
            Combines the multiple heads into a single tensor.
            Args:
                x (tensor): Input tensor of shape (batch_size, num_heads, seq_length, head_hidden_dim).
            Returns:
                tensor: Combined tensor of shape (batch_size, seq_length, num_heads * head_hidden_dim).
        forward(q, k, v, attention_mask=None, key_padding_mask=None):
            Forward pass for the multi-head attention mechanism.
            Args:
                q (tensor): Query tensor of shape (batch_size, query_sequence_length, hidden_dim).
                k (tensor): Key tensor of shape (batch_size, key_sequence_length, hidden_dim).
                v (tensor): Value tensor of shape (batch_size, key_sequence_length, hidden_dim).
                attention_mask (tensor, optional): Attention mask tensor of shape (query_sequence_length, key_sequence_length).
                key_padding_mask (tensor, optional): Key padding mask tensor of shape (sequence_length, key_sequence_length).
            Returns:
                tensor: Output tensor after applying multi-head attention.
    """
    def __init__(self, hidden_dim=256, num_heads=4):
       
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num heads"
        self.value_weights = nn.Linear(hidden_dim, hidden_dim, bias=False) # the Value part
        self.key_weights = nn.Linear(hidden_dim, hidden_dim, bias=False) # the Key part
        self.query_weights = nn.Linear(hidden_dim, hidden_dim, bias=False) # the Query part
        self.output_weights = nn.Linear(hidden_dim, hidden_dim, bias=False) # the output layer
        
        
    def check_sdpa_inputs(self, x):
        assert x.size(1) == self.num_heads, f"Expected size of x to be ({-1, self.num_heads, -1, self.hidden_dim // self.num_heads}), got {x.size()}"
        assert x.size(3) == self.hidden_dim // self.num_heads
        
        
    def scaled_dot_product_attention(
            self, 
            query, 
            key, 
            value, 
            attention_mask=None, 
            key_padding_mask=None):
        """
        query : tensor of shape (batch_size, num_heads, query_sequence_length, hidden_dim//num_heads)
        key : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        value : tensor of shape (batch_size, num_heads, key_sequence_length, hidden_dim//num_heads)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
        
    
        """
        self.check_sdpa_inputs(query)
        self.check_sdpa_inputs(key)
        self.check_sdpa_inputs(value)
        
        
        d_k = query.size(-1)
        tgt_len, src_len = query.size(-2), key.size(-2)

        
        # logits = (B, H, tgt_len, E) * (B, H, E, src_len) = (B, H, tgt_len, src_len)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
        
        # Attention mask here
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                assert attention_mask.size() == (tgt_len, src_len)
                attention_mask = attention_mask.unsqueeze(0)
                logits = logits + attention_mask
            else:
                raise ValueError(f"Attention mask size {attention_mask.size()}")
        
                
        # Key mask here
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # Broadcast over batch size, num heads
            logits = logits + key_padding_mask
        
        
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, value) # (batch_size, num_heads, sequence_length, hidden_dim)
        
        return output, attention

    
    def split_into_heads(self, x, num_heads):
        batch_size, seq_length, hidden_dim = x.size()
        x = x.view(batch_size, seq_length, num_heads, hidden_dim // num_heads)
        
        return x.transpose(1, 2) # Final dim will be (batch_size, num_heads, seq_length, , hidden_dim // num_heads)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, head_hidden_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * head_hidden_dim)
        
    
    def forward(
            self, 
            q, 
            k, 
            v, 
            attention_mask=None, 
            key_padding_mask=None):
        """
        q : tensor of shape (batch_size, query_sequence_length, hidden_dim)
        k : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        v : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
       
        """
        q = self.query_weights(q)
        k = self.key_weights(k)
        v = self.value_weights(v)

        q = self.split_into_heads(q, self.num_heads)
        k = self.split_into_heads(k, self.num_heads)
        v = self.split_into_heads(v, self.num_heads)
        
        # attn_values, attn_weights = self.multihead_attn(q, k, v, attn_mask=attention_mask)
        attn_values, attn_weights  = self.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        grouped = self.combine_heads(attn_values)
        output = self.output_weights(grouped)
        
        self.attention_weigths = attn_weights
        
        return output