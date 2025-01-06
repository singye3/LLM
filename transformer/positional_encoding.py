import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Implements the positional encoding as described in the paper "Attention Is All You Need" by Vaswani et al.
    Args:
        d_model (int): The dimension of the model.
        dropout (float, optional): The dropout rate. Default is 0.1.
        max_len (int, optional): The maximum length of the input sequences. Default is 5000.
    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (torch.Tensor): Positional encoding matrix.
    Methods:
        forward(x):
            Adds positional encoding to the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim].
            Returns:
                torch.Tensor: Tensor with positional encoding added, of shape [batch_size, seq_len, embedding_dim].
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

