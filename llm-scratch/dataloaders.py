# Reference : llm-from-scratch book by Sebastian Raschka
import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, i):
        return self.input_ids[i], self.target_ids[i]

def create_dataloader(txt, batch=4, max_length=128, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader