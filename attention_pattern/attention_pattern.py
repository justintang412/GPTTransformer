import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import GPT2Tokenizer
from transformer_gpt import GPTTransformer

class GPTTransformerWithAttention(GPTTransformer):
    """Extended model that captures attention weights"""
    
    def forward_with_attention(self, input_ids: torch.Tensor):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        x = self.token_embedding(input_ids) * torch.sqrt(torch.tensor(512.0))
        x = x + self.position_encoding[:, :seq_length]
        x = self.dropout(x)
        
        transformer_block_output = x
        all_attention_weights = []  # Store attention from all layers
        
        for layer_idx in range(6):
            normalized_x = self.layer_normalize1(transformer_block_output)
            
            Q = self.w_q(normalized_x).view(batch_size, seq_length, 8, 64).transpose(1,2)
            K = self.w_k(normalized_x).view(batch_size, seq_length, 8, 64).transpose(1,2)
            V = self.w_v(normalized_x).view(batch_size, seq_length, 8, 64).transpose(1,2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
            scores = scores.masked_fill(mask==0, -1e4)
            attention_weights = F.softmax(scores, dim=-1)
            
            # Save attention weights before dropout
            all_attention_weights.append(attention_weights.detach().cpu())
            
            attention_weights = self.dropout(attention_weights)
            context = torch.matmul(attention_weights, V)
            context = context.transpose(1,2).contiguous().view(batch_size, seq_length, 512)
            attended = self.w_output_project_layer(context)
            attended = transformer_block_output + self.dropout(attended)
            
            feed_forwad_x = self.layer_normalize2(attended)
            feed_forwad_x = self.feed_forward_linear1(feed_forwad_x)
            feed_forwad_x = F.gelu(feed_forwad_x)
            feed_forwad_x = self.dropout(feed_forwad_x)
            feed_forwad_x = self.feed_forward_linear2(feed_forwad_x)
            transformer_block_output = attended + self.dropout(feed_forwad_x)
        
        transformer_block_output = self.final_normal_layer(transformer_block_output)
        logits = self.lm_head(transformer_block_output)
        
        return logits, all_attention_weights

def visualize_attention(text: str, image:str):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GPTTransformerWithAttention(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load('../trained_model.pth', map_location=device))
    model.eval()
    
    # Tokenize input
    input_ids = torch.tensor([tokenizer.encode(text)], device=device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    # Get attention weights
    with torch.no_grad():
        logits, attention_weights = model.forward_with_attention(input_ids)
    
    # attention_weights: list of 6 tensors, each (batch=1, heads=8, seq_len, seq_len)
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    # Visualize all layers and heads
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(20, 15))
    fig.suptitle(f'Attention Patterns: "{text}"', fontsize=16)
    
    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx][0]  # (8, seq_len, seq_len)
        
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            
            # Plot heatmap
            sns.heatmap(
                attn[head_idx].numpy(),
                ax=ax,
                cmap='viridis',
                xticklabels=tokens if len(tokens) < 20 else False,
                yticklabels=tokens if len(tokens) < 20 else False,
                cbar=True,
                square=True
            )
            
            if head_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)
            if layer_idx == 0:
                ax.set_title(f'Head {head_idx}', fontsize=10)
            
            ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.savefig(f'{image}_attention_all_layers_heads.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Average attention across heads for each layer
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Average Attention per Layer: "{text}"', fontsize=14)
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx // 3, layer_idx % 3]
        
        # Average across all heads
        avg_attn = attention_weights[layer_idx][0].mean(dim=0).numpy()
        
        sns.heatmap(
            avg_attn,
            ax=ax,
            cmap='YlOrRd',
            xticklabels=tokens if len(tokens) < 30 else False,
            yticklabels=tokens if len(tokens) < 30 else False,
            cbar=True,
            square=True
        )
        
        ax.set_title(f'Layer {layer_idx}', fontsize=12)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{image}_attention_avg_per_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show which tokens attend most to each position (last layer)
    last_layer_attn = attention_weights[-1][0].mean(dim=0).numpy()  # Average across heads
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        last_layer_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=len(tokens) < 15,  # Show values if sequence is short
        fmt='.2f',
        cbar=True,
        square=True
    )
    plt.title(f'Last Layer Attention Pattern\n"{text}"', fontsize=14)
    plt.xlabel('Attends To (Key)', fontsize=12)
    plt.ylabel('Token (Query)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{image}_attention_last_layer_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    test_text = "To be or not to be, that is the question"
    visualize_attention(test_text, 'tobe')
    
    # Try with a shorter phrase for clearer visualization
    visualize_attention("The cat sat on the mat", 'cat')