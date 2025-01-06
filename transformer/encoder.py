import torch 
import torch.nn as nn
import math
from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock

class Encoder(nn.Module):
    """
    Encoder class for a Transformer model.
    Args:
        vocab_size (int): Size of the vocabulary.
        n_dim (int): Dimensionality of the embeddings and the model.
        dropout (float): Dropout rate to be applied.
        n_encoder_blocks (int): Number of encoder blocks.
        n_heads (int): Number of attention heads in each encoder block.
    Attributes:
        n_dim (int): Dimensionality of the embeddings and the model.
        embedding (nn.Embedding): Embedding layer for input tokens.
        positional_encoding (PositionalEncoding): Positional encoding layer.
        encoder_blocks (nn.ModuleList): List of encoder blocks.
    Methods:
        forward(x, padding_mask=None):
            Forward pass through the encoder.
            Args:
                x (Tensor): Input tensor of shape (batch_size, sequence_length).
                padding_mask (Tensor, optional): Padding mask for the input tensor.
            Returns:
                Tensor: Output tensor of shape (batch_size, sequence_length, n_dim).
    """
    def __init__(self, vocab_size: int, n_dim: int, dropout: float, n_encoder_blocks: int,n_heads: int):
        
        super(Encoder, self).__init__()
        self.n_dim = n_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim)
        self.positional_encoding = PositionalEncoding(d_model=n_dim, dropout=dropout)    
        self.encoder_blocks = nn.ModuleList([EncoderBlock(n_dim, dropout, n_heads) for _ in range(n_encoder_blocks)])
        
        
    def forward(self, x, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.n_dim)
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x=x, src_padding_mask=padding_mask)
        return x
