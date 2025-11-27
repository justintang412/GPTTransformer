import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer
from gpt_transformer.transformer_gpt import GPTTransformer

class TrainingMetrics:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.epoch_losses = []
        self.epoch_perplexities = []
        self.learning_rates = []
        self.batch_losses = []  # For detailed loss tracking
        
    def add_epoch(self, epoch, avg_loss, lr):
        """Add metrics for a completed epoch"""
        self.epoch_losses.append(avg_loss)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        self.epoch_perplexities.append(perplexity)
        self.learning_rates.append(lr)
        
    def add_batch_loss(self, loss):
        """Track individual batch losses"""
        self.batch_losses.append(loss)
        
    def plot_metrics(self, save_prefix='training'):
        """Generate and save all metric plots"""
        epochs = list(range(len(self.epoch_losses)))
        
        # 1. Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Epochs', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_loss_curve.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_prefix}_loss_curve.png")
        plt.close()
        
        # 2. Perplexity curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.epoch_perplexities, 'g-', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Perplexity', fontsize=12)
        plt.title('Perplexity Over Epochs', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_perplexity_curve.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_prefix}_perplexity_curve.png")
        plt.close()
        
        # 3. Learning rate schedule
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.learning_rates, 'r-', linewidth=2, marker='^', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for learning rate
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_learning_rate.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_prefix}_learning_rate.png")
        plt.close()
        
        # 4. Combined plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(epochs, self.epoch_losses, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Perplexity
        axes[0, 1].plot(epochs, self.epoch_perplexities, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, 'r-', linewidth=2, marker='^')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Batch losses (moving average)
        if self.batch_losses:
            window_size = 50
            smoothed_losses = []
            for i in range(len(self.batch_losses)):
                start_idx = max(0, i - window_size + 1)
                smoothed_losses.append(sum(self.batch_losses[start_idx:i+1]) / (i - start_idx + 1))
            
            axes[1, 1].plot(smoothed_losses, 'purple', linewidth=1, alpha=0.7)
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Loss (Moving Avg)')
            axes[1, 1].set_title('Batch-Level Loss (Smoothed)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_all_metrics.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_prefix}_all_metrics.png")
        plt.close()
        
    def save_to_file(self, filepath='training_metrics.txt'):
        """Save metrics to text file"""
        with open(filepath, 'w') as f:
            f.write("Epoch\tLoss\tPerplexity\tLearning Rate\n")
            for i, (loss, ppl, lr) in enumerate(zip(self.epoch_losses, 
                                                     self.epoch_perplexities, 
                                                     self.learning_rates)):
                f.write(f"{i}\t{loss:.4f}\t{ppl:.4f}\t{lr:.6f}\n")
        print(f"Saved metrics to: {filepath}")


def train_with_metrics():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size  # 50257
    
    with open('transformer_gpt.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Encode text in smaller chunks
    chunk_size = 3000
    tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunk_tokens = tokenizer.encode(chunk, truncation=True, max_length=1024)
        tokens.extend(chunk_tokens)
    
    sequences = []
    stride = 512
    for i in range(0, len(tokens) - 1024, stride):
        sequences.append(tokens[i:i+1024 + 1])
    
    batches = []
    batch_size = 8
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i+batch_size]
        if len(batch) == batch_size:
            batch_tensor = torch.tensor(batch, dtype=torch.long)
            input_ids = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]
            batches.append((input_ids, targets))
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = GPTTransformer(vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.1f}M")
    print(f"Number of training batches: {len(batches)}")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    
    # Alternative: Step decay scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Alternative: Reduce on plateau
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, verbose=True
    # )
    
    metrics = TrainingMetrics()
    model.train()
    accumulation_steps = 16
    
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (input_ids, targets) in enumerate(batches):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits, loss = model(input_ids, targets)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            batch_loss = loss.item() * accumulation_steps
            total_loss += batch_loss
            metrics.add_batch_loss(batch_loss)
        
        # Apply any remaining gradients
        if len(batches) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(batches)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step()
        # For ReduceLROnPlateau, use: scheduler.step(avg_loss)
        
        # Record metrics
        metrics.add_epoch(epoch, avg_loss, current_lr)
        
        perplexity = math.exp(min(avg_loss, 20))
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} - "
              f"Perplexity: {perplexity:.2f} - LR: {current_lr:.6f}")
        
        # Save plots periodically
        if epoch % 10 == 0 and epoch > 0:
            metrics.plot_metrics(save_prefix=f'training_epoch_{epoch}')
            
            # Generate sample text
            model.eval()
            with torch.no_grad():
                prompt = "To be or not to be"
                input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

                for _ in range(30):
                    if input_ids.size(1) >= 1024:
                        input_ids = input_ids[:, -1023:]
                    
                    logits, _ = model(input_ids)
                    next_token_logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                generated = tokenizer.decode(input_ids[0].tolist())
                print(f"\nGenerated text:\n{generated}\n")
            model.train()
    
    # Final plots and save
    metrics.plot_metrics(save_prefix='training_final')
    metrics.save_to_file('training_metrics.txt')
    torch.save(model.state_dict(), 'gpt_model_with_metrics.pth')
    print("Training complete!")

if __name__ == "__main__":
    train_with_metrics()
