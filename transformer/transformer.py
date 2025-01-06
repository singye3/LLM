import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer Model
    This class implements a Transformer model for sequence-to-sequence tasks. It includes methods for encoding, decoding, and generating predictions.
    Attributes:
        vocab_size (int): Size of the vocabulary.
        model_dim (int): Dimension of the model.
        dropout (float): Dropout rate.
        n_encoder_block (int): Number of encoder blocks.
        n_decoder_block (int): Number of decoder blocks.
        n_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        PAD_IDX (int): Index for padding token.
    Methods:
        generate_square_subsequent_mask(size: int) -> torch.Tensor:
            Generates a square subsequent mask for the sequence.
        encode(x: torch.Tensor) -> torch.Tensor:
            Encodes the input sequence.
        decode(tgt: torch.Tensor, memory: torch.Tensor, memory_padding_mask=None) -> torch.Tensor:
            Decodes the target sequence using the encoded memory.
        forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            Forward pass through the Transformer model.
        predict(x: torch.Tensor, sos_idx: int=1, eos_idx: int=2, max_length: int=None) -> torch.Tensor:
            Generates predictions using greedy inference.
    """
    def __init__(self, **kwargs):
        """
        Initializes the Transformer model with the given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
                - vocab_size (int): Size of the vocabulary.
                - model_dim (int): Dimension of the model.
                - dropout (float): Dropout rate.
                - n_encoder_block (int): Number of encoder blocks.
                - n_decoder_block (int): Number of decoder blocks.
                - n_heads (int): Number of attention heads.
                - batch_size (int): Batch size.
                - pad_idx (int, optional): Padding index. Defaults to 0.
        """
        super(Transformer, self).__init__()
        self.vocab_size = kwargs.get('vocab_size')
        self.model_dim = kwargs.get('model_dim')
        self.dropout = kwargs.get('dropout')
        self.n_encoder_block = kwargs.get('n_encoder_block')
        self.n_decoder_block = kwargs.get('n_decoder_block')
        self.n_heads = kwargs.get('n_heads')
        self.batch_size = kwargs.get('batch_size')
        self.PAD_IDX = kwargs.get('pad_idx', 0)

        self.encoder = Encoder(self.vocab_size, self.model_dim, self.dropout, self.n_encoder_block, self.n_heads)
        self.decoder = Decoder(self.vocab_size, self.model_dim, self.dropout, self.n_decoder_block, self.n_heads)

        self.fc = nn.Linear(self.model_dim, self.vocab_size)
    
    @staticmethod
    def generate_square_subsequent_mask(size: int):
        # mask generation
        mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor using the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (B, S) where B is the batch size and S is the sequence length.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - encoder_output (torch.Tensor): The output tensor from the encoder of shape (B, S, E) where E is the embedding dimension.
                - encoder_padding_mask (torch.Tensor): The padding mask tensor of shape (B, S) with float values, where padding positions are filled with -inf.
        """
        mask = (x == self.PAD_IDX).float() 
        encoder_padding_mask = mask.masked_fill(mask == 1, float('-inf'))

        # (B, S, E)
        encoder_output = self.encoder(x, padding_mask=encoder_padding_mask)
        return encoder_output, encoder_padding_mask
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_padding_mask=None) -> torch.Tensor:
        """
        Decodes the target sequence using the provided memory tensor.

        Args:
            tgt (torch.Tensor): The target sequence tensor of shape (B, L).
            memory (torch.Tensor): The memory tensor from the encoder of shape (B, S, E).
            memory_padding_mask (torch.Tensor, optional): The padding mask for the memory tensor. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape (B, L, C) after decoding.
        """

        mask = (tgt == self.PAD_IDX).float()
        tgt_padding_mask = mask.masked_fill(mask == 1, float('-inf'))

        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)),
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask
        )
        output = self.fc(decoder_output) # (B,L,C)
        return output
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        encoder_output, encoder_padding_mask = self.encode(x)
        decoder_output = self.decode(
            tgt=y,
            memory=encoder_output,
            memory_padding_mask=encoder_padding_mask
        )
        return decoder_output
    
    def predict(self, x: torch.Tensor, sos_idx: int=1, eos_idx: int=2, max_length: int=None) -> torch.Tensor:
        """
        Generate greedy predictions (argmax) for the given input tensor using the transformer model.
        Args:
            x (torch.Tensor): The input tensor to generate predictions for.
            sos_idx (int, optional): The start-of-sequence token index. Defaults to 1.
            eos_idx (int, optional): The end-of-sequence token index. Defaults to 2.
            max_length (int, optional): The maximum length of the generated sequence. If None, it defaults to the length of the input tensor.
        Returns:
            torch.Tensor: The generated sequence tensor.
        """
        x = torch.cat([torch.tensor([sos_idx]), x, torch.tensor([eos_idx])]).unsqueeze(0)
        encoder_output, mask = self.transformer.encode(x) # (B, S, E)
        
        if not max_length:
            max_length = x.size(1)
        outputs = torch.ones((x.size()[0], max_length)).type_as(x).long() * sos_idx
        for step in range(1, max_length):
            y = outputs[:, :step]
            probs = self.tranformer.decode(y, encoder_output)
            output = torch.argmax(probs, dim=-1)

            if output[:, -1].detach().numpy() in (eos_idx, sos_idx):
                break
            outputs[:, step] = outputs[:, -1]

        return outputs
    




            
    

