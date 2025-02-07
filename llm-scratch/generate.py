import torch
import tiktoken

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, k_top=None, eos_id=None):
    # idx : (b, t)
    for _ in range(max_new_tokens):
        # crop the context to support the context size
        idx_cond = idx[:, -context_size:]
        #get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] # shape: (b, vocab_size)
        # k-top sampling
        if k_top is not None:
            top_logits, _ = torch.topk(logits, k_top)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf').to(logits.device)), logits)
        # temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # greedy sampling (argmax)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # shape: (b, 1)
        # auto-regressive
        idx = torch.cat([idx, idx_next], dim=-1) # (batch, n_tokens+1)

    return idx

