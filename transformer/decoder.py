import torch.nn as nn
from decoder_block import DecoderBlock
from positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    """
    Decoder class for a Transformer model.
    Args:
        vocab_size (int): Size of the vocabulary.
        n_dim (int): Dimensionality of the embeddings and the model.
        dropout (float): Dropout rate.
        n_decoder_blocks (int): Number of decoder blocks.
        n_heads (int): Number of attention heads in each decoder block.
    Attributes:
        embedding (nn.Embedding): Embedding layer for input tokens.
        positional_encoding (PositionalEncoding): Positional encoding layer.
        decoder_blocks (nn.ModuleList): List of decoder blocks.
    Methods:
        forward(tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
            Forward pass through the decoder.
            Args:
                tgt (Tensor): Target sequence of shape (T, B) where T is the target sequence length and B is the batch size.
                memory (Tensor): Memory from the encoder of shape (S, B, E) where S is the source sequence length and E is the embedding dimension.
                tgt_mask (Tensor, optional): Mask for the target sequence to prevent attending to future tokens.
                tgt_padding_mask (Tensor, optional): Mask to ignore padding tokens in the target sequence.
                memory_padding_mask (Tensor, optional): Mask to ignore padding tokens in the memory sequence.
            Returns:
                Tensor: Output of the decoder of shape (T, B, E).
    """
    def __init__(self, vocab_size: int, n_dim: int, dropout: float, n_decoder_blocks: int,n_heads: int):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim,padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model=n_dim, dropout=dropout)
          
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_dim, dropout, n_heads) for _ in range(n_decoder_blocks)])
        
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, memory, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=memory_padding_mask)
        return x