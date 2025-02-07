import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken

from dataloaders import create_dataloader
from gpt import GPTModel
from generate import *

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) 
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return total_loss
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, encoded, max_new_tokens=50, context_size=context_size) 
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace('\n', ' '))
    model.train()

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # training loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()

def main(gpt_config, settings):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load file
    file_path = 'the-verdict.txt'
    url = 'https://raw.githubusercontent.com/finetuneanon/data/main/the-verdict.txt'

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    # model initialization
    model = GPTModel(gpt_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])

    # dataloader
    train_ratio = 0.9
    split_idx = int(len(text_data) * train_ratio)

    train_loader = create_dataloader(
        text_data[:split_idx],
        batch=settings['batch_size'],
        max_length=gpt_config['context_length'],
        stride=gpt_config['context_length'],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader(
        text_data[split_idx:],
        batch=settings['batch_size'],
        max_length=gpt_config['context_length'],
        stride=gpt_config['context_length'],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # training
    tokenizer = tiktoken.get_encoding('gpt2')
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device, settings['num_epochs'], settings['eval_freq'], settings['eval_iter'], settings['start_context'], tokenizer
    )
    return train_losses, val_losses, tokens_seen, model

torch.cuda.empty_cache()
GPTConfig = {
    "vocab_size": 50257,    
    "context_length": 256,
    "emb_dim": 768,         
    "n_heads": 12,          
    "n_layers": 12,        
    "drop_rate": 0.1,      
    "qkv_bias": False       
}

settings = {
    "learning_rate": 5e-4,
    "num_epochs": 20,
    "batch_size": 32,
    "weight_decay": 0.1,
    "eval_freq": 5,
    "eval_iter": 1,
    "start_context": "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
}

train_losses, val_losses, tokens_seen, model = main(GPTConfig, settings)

# Plot the training and validation losses
epoch_tensor = torch.linspace(0, settings['num_epochs'], len(train_losses))
plot_losses(epoch_tensor, tokens_seen, train_losses, val_losses)
plt.savefig('loss.pdf')

# save model 
torch.save(model.state_dict(), "model.pth")
model = GPTModel(GPTConfig)
model.load_state_dict(torch.load("model.pth", weights_only=True))
 