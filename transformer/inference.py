import torch
import torch.nn as nn
from transformer import Transformer

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
class Translator(nn.Module):
    """
    Translator class that uses a transformer model to translate sentences.
    Args:
        transformer (nn.Module): The transformer model used for translation.
    Methods:
        str_to_tokens(s: str) -> List[int]:
            Converts a string to a list of token indices.
        tokens_to_str(tokens: List[int]) -> str:
            Converts a list of token indices back to a string.
        __call__(sentence: str, max_length: int = None, pad: bool = False) -> str:
            Translates the input sentence using the transformer model.
            Args:
                sentence (str): The input sentence to be translated.
                max_length (int, optional): The maximum length of the translated sentence. Defaults to None.
                pad (bool, optional): Whether to pad the sentence. Defaults to False.
            Returns:
                str: The translated sentence.
    """
    def __init__(self, transformer):
        super(Translator, self).__init__()
        self.transformer = transformer
    
    @staticmethod
    def str_to_tokens(s):
        return [ord(z)-97+3 for z in s]
    
    @staticmethod
    def tokens_to_str(tokens):
        return "".join([chr(x+94) for x in tokens])
    
    def __call__(self, sentence, max_length=None, pad=False):
        
        x = torch.tensor(self.str_to_tokens(sentence))
        x = torch.cat([torch.tensor([SOS_IDX]), x, torch.tensor([EOS_IDX])]).unsqueeze(0)
        
        encoder_output, mask = self.transformer.encode(x) # (B, S, E)
        
        if not max_length:
            max_length = x.size(1)
            
        outputs = torch.ones((x.size()[0], max_length)).type_as(x).long() * SOS_IDX
        
        for step in range(1, max_length):
            y = outputs[:, :step]
            probs = self.transformer.decode(y, encoder_output)
            output = torch.argmax(probs, dim=-1)
            if output[:, -1].detach().numpy() in (EOS_IDX, SOS_IDX):
                break
            outputs[:, step] = output[:, -1]
            
        
        return self.tokens_to_str(outputs[0])

# model parameters
args = {
    'vocab_size': 128,
    'model_dim': 128,
    'dropout': 0.1,
    'n_encoder_block': 1,
    'n_decoder_block': 1,
    'n_heads': 4
}
# Load the model
model = Transformer(**args)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=True))
translator = Translator(model)

# Reverese a sentence
sentence = "helloworld"
translated = translator(sentence)
print(f"Input: {sentence}")
print(f"Output: {translated}")