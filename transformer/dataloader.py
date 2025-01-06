import numpy as np
import torch
from torch.utils.data import Dataset

# Set random seed for reproducibility
np.random.seed(0)

# Function to generate a random string of length between 10 and 20
def generate_random_string():
    length = np.random.randint(10, 20)
    return "".join([chr(x) for x in np.random.randint(97, 97+26, length)])

class ReverseDataset(Dataset):
    """
    A custom dataset class for reversing strings.
    Args:
        n_samples (int): Number of samples to generate.
        pad_idx (int): Padding index.
        sos_idx (int): Start of sequence index.
        eos_idx (int): End of sequence index.
    Attributes:
        pad_idx (int): Padding index.
        sos_idx (int): Start of sequence index.
        eos_idx (int): End of sequence index.
        values (list): List of generated random strings.
        labels (list): List of reversed strings corresponding to `values`.
    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(index):
            Returns the transformed input and target tensors for the given index.
        text_transform(x):
            Converts a string to a tensor of indices, adding SOS and EOS tokens.
    """
    def __init__(self, n_samples, pad_idx, sos_idx, eos_idx):
        super(ReverseDataset, self).__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.values = [generate_random_string() for _ in range(n_samples)]
        self.labels = [x[::-1] for x in self.values]

    def __len__(self):
        return len(self.values)  # Number of samples in the dataset

    def __getitem__(self, index):
        return self.text_transform(self.values[index].rstrip("\n")), \
               self.text_transform(self.labels[index].rstrip("\n"))
        
    def text_transform(self, x):
        # Convert string to tensor of indices, adding SOS and EOS tokens
        return torch.tensor([self.sos_idx] + [ord(z) - 97 + 3 for z in x] + [self.eos_idx])
        


