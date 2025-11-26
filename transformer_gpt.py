import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers import GPT2Tokenizer

class GPTTransformer(nn.Module):
    def __init__(self, num_embeddings:int):
        super().__init__()
        self.num_embeddings = num_embeddings
        
        # embedding, vocabulary size from tokenizer, embedding length 512
        self.token_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=512)
        
        # position
        # 512 embedding dimension, 1024 max sequence length
        position_encoding = torch.zeros(1024, 512)
        position = torch.arange(0, 1024, dtype=torch.float)
        position = position.unsqueeze(1)
        # -math.log(10000.0)/512 = -0.018
        div_term = torch.exp(torch.arange(0, 512, 2).float() * (-math.log(10000.0)/512))
        position_encoding[:, 0::2] = torch.sin(position*div_term)
        position_encoding[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('position_encoding', position_encoding.unsqueeze(0))

        # attention
        self.w_q = nn.Linear(512, 512, bias=False)
        self.w_k = nn.Linear(512, 512, bias=False)
        self.w_v = nn.Linear(512, 512, bias=False)
        self.w_output_project_layer = nn.Linear(512, 512)
        self.attention_scale = math.sqrt(512/8) #64

        # feed forard from 512 2048
        self.feed_forward_linear1 = nn.Linear(512, 2048)
        self.feed_forward_linear2 = nn.Linear(2048, 512)
        
        # nomorlize
        self.layer_normalize1 = nn.LayerNorm(512)
        self.layer_normalize2 = nn.LayerNorm(512)

        #output layer
        self.final_normal_layer = nn.LayerNorm(512)
        self.lm_head = nn.Linear(512, num_embeddings, bias=False)

        self.dropout = nn.Dropout(0.1)
        
        #initialize weights
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w_q.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w_k.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w_output_project_layer.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.w_output_project_layer.bias)
        torch.nn.init.normal_(self.feed_forward_linear1.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.feed_forward_linear1.bias)
        torch.nn.init.normal_(self.feed_forward_linear2.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.feed_forward_linear2.bias)
        torch.nn.init.ones_(self.layer_normalize1.weight)
        torch.nn.init.zeros_(self.layer_normalize1.bias)
        torch.nn.init.ones_(self.layer_normalize2.weight)
        torch.nn.init.zeros_(self.layer_normalize2.bias)
        torch.nn.init.ones_(self.final_normal_layer.weight)
        torch.nn.init.zeros_(self.final_normal_layer.bias)
        
        # Tie weights after initialization
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, 
                input_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, 
                                                                 Optional[torch.Tensor]]:
        # train with many batches for efficiency, infer with only one.
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0) #(1,1,seq_length, seq_length)
        x = self.token_embedding(input_ids) * math.sqrt(512)
        x = x + self.position_encoding[:, :seq_length]
        x = self.dropout(x)

        transformer_block_output = x
        for _ in range(6):
            normalized_x = self.layer_normalize1(transformer_block_output)
            
            Q = self.w_q(normalized_x).view(batch_size, seq_length, 8, 64).transpose(1,2)
            K = self.w_k(normalized_x).view(batch_size, seq_length, 8, 64).transpose(1,2)
            V = self.w_v(normalized_x).view(batch_size, seq_length, 8, 64).transpose(1,2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
            scores = scores.masked_fill(mask==0, -1e4)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            context = torch.matmul(attention_weights, V)
            context = context.transpose(1,2).contiguous().view(batch_size, seq_length, 512)
            attended = self.w_output_project_layer(context)
            # attention
            attended = transformer_block_output + self.dropout(attended)

            feed_forwad_x = self.layer_normalize2(attended)
            feed_forwad_x = self.feed_forward_linear1(feed_forwad_x)
            feed_forwad_x = F.gelu(feed_forwad_x)
            feed_forwad_x = self.dropout(feed_forwad_x)
            # feed forward
            feed_forwad_x = self.feed_forward_linear2(feed_forwad_x)
            # output
            transformer_block_output = attended + self.dropout(feed_forwad_x)
        
        transformer_block_output = self.final_normal_layer(transformer_block_output)
        logits = self.lm_head(transformer_block_output)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1),
                ignore_index=-1
            )
        return logits, loss
    
def train():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size  # 50257
    with open('transformer_gpt.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Encode text in smaller chunks to avoid exceeding max_length
    chunk_size = 3000  # Process text in smaller chunks of characters
    tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunk_tokens = tokenizer.encode(chunk, truncation=True, max_length=1024)
        tokens.extend(chunk_tokens)
    
    sequences = []
    stride = 512  # Use stride of 512 instead of 1 for much faster training
    for i in range(0, len(tokens) - 1024, stride):
        sequences.append(tokens[i:i+1024 + 1])
    batches = []
    batch_size = 8  # Reduced batch size to fit in GPU memory
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i+batch_size]
        if len(batch) == batch_size:
            batch_tensor = torch.tensor(batch, dtype=torch.long)
            input_ids = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]
            batches.append((input_ids, targets))
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = GPTTransformer(vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters {total_params/1e6:.1f}M")
    print(f"Number of training batches: {len(batches)}")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    accumulation_steps = 16  # Simulate batch size of 128 (8 * 16)
    
    for epoch in range(100):
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (input_ids, targets) in enumerate(batches):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits, loss = model(input_ids, targets)
            loss = loss / accumulation_steps  # Scale loss for gradient accumulation
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
        
        # Apply any remaining gradients
        if len(batches) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss= total_loss/len(batches)
        print(f"Epoch {epoch}/100 - Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                prompt = "To be or not to be"
                input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

                for _ in range(30):
                    # Truncate if sequence gets too long
                    if input_ids.size(1) >= 1024:
                        input_ids = input_ids[:, -1023:]
                    
                    logits, _ = model(input_ids)
                    # Use temperature sampling for better diversity
                    next_token_logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                generated = tokenizer.decode(input_ids[0].tolist())
                print(f"\nGenerated text:\n{generated}\n")
            model.train()
        
    torch.save(model.state_dict(), 'gpt_model.pth')

if __name__ == "__main__":
    train()