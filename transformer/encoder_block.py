import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from position_wise_feedforward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    """
    EncoderBlock is a single block of the Transformer encoder.
    Args:
        n_dim (int): The dimension of the input and output features.
        dropout (float): The dropout rate to be applied after the multi-head attention and feed-forward layers.
        n_heads (int): The number of attention heads in the multi-head attention mechanism.
    Attributes:
        mha (MultiHeadAttention): Multi-head attention mechanism.
        norm1 (nn.LayerNorm): Layer normalization applied after the multi-head attention.
        ff (PositionWiseFeedForward): Position-wise feed-forward network.
        norm2 (nn.LayerNorm): Layer normalization applied after the feed-forward network.
        dropout (nn.Dropout): Dropout layer applied after the multi-head attention and feed-forward layers.
    Methods:
        forward(x, src_padding_mask=None):
            Passes the input through the encoder block.
            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_length, n_dim).
                src_padding_mask (torch.Tensor, optional): The padding mask for the input tensor.
            Returns:
                torch.Tensor: The output tensor of shape (batch_size, seq_length, n_dim).
    """
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_padding_mask=None):
        assert x.ndim==3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))
        
        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)
       
        return output
    