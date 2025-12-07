import torch
from transformers import GPT2Tokenizer
from transformer_gpt import GPTTransformer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTTransformer(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load('gpt_transformer/trained_model.pth', map_location=device))
model.eval()
input_ids = torch.tensor([tokenizer.encode('What he cannot')], device=device)
with torch.no_grad():
    generated = []
    for _ in range(30):
        if input_ids.size(1) >= 1024:
            input_ids = input_ids[:, -1023:]
        logits, loss = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / 0.8
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(tokenizer.decode([next_token.item()]))
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    print(''.join(generated))

        