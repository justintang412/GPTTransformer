# Attention Pattern Visualization

This module visualizes the attention patterns of the trained GPT Transformer model, providing deep insights into how the model processes and understands text.

## Overview

The `attention_pattern.py` script generates three types of visualizations to help you understand what your transformer model has learned:

### 1. All Layers and Heads Grid
**Output:** `*_attention_all_layers_heads.png`

A comprehensive 6×8 grid showing **each individual attention head separately**:
- 6 layers (rows) × 8 heads (columns) = 48 separate heatmaps
- Each heatmap shows how one specific attention head attends to different positions
- Allows you to observe head specialization patterns across the entire model

### 2. Average Per Layer
**Output:** `*_attention_avg_per_layer.png`

Shows 6 heatmaps (one per layer):
- Averages all 8 heads together for each layer
- Reveals the overall attention behavior at each layer
- Helps track how attention patterns evolve from early to late layers

### 3. Last Layer Detailed
**Output:** `*_attention_last_layer_detailed.png`

Focuses on the final layer's attention:
- Averaged across all heads in the last layer
- Shows the final representation before prediction
- Includes numerical values for short sequences

## Why Visualize Per-Head Attention?

Visualizing per-head attention helps you answer critical questions about your model:

### ❓ "Is my model learning real structure or just memorizing?"
**→ If heads specialize, your model is learning structure.**

Different heads should learn different patterns:
- Some heads may focus on adjacent tokens (local context)
- Others may capture long-range dependencies (distant relationships)
- Specialized heads indicate genuine pattern learning vs. memorization

### ❓ "Are some heads dead?"
**→ Important for debugging and training efficiency.**

Dead heads show uniform or random attention patterns:
- Indicates wasted capacity in your model
- May suggest training issues (learning rate, initialization, data)
- Candidates for pruning to improve efficiency

### ❓ "How do representations evolve across layers?"
**→ Per-head variation shows the internal process of comprehension.**

Track how attention patterns change from layer to layer:
- Early layers: Often focus on syntax and local structure
- Middle layers: Build semantic relationships
- Late layers: Form task-specific representations

### ❓ "Is the model building long-range dependencies?"
**→ Induction heads will reveal this.**

Induction heads are attention heads that can copy patterns:
- They attend to tokens that appeared after similar contexts earlier
- Critical for tasks requiring pattern matching and repetition
- Visible as diagonal or structured patterns in attention maps

### ❓ "Why does my model behave a certain way during inference?"
**→ Attention maps often answer this.**

When your model makes unexpected predictions:
- Check which tokens it's attending to
- Identify if certain heads are dominating the decision
- Understand the reasoning path through the network

## Usage

### Basic Visualization

```python
from attention_pattern import visualize_attention

# Visualize attention for a text sample
text = "To be or not to be, that is the question"
visualize_attention(text, 'tobe')
```

This generates three PNG files:
- `tobe_attention_all_layers_heads.png`
- `tobe_attention_avg_per_layer.png`
- `tobe_attention_last_layer_detailed.png`

### Analyzing Your Trained Model

Run the script directly to visualize example sentences:

```bash
python attention_pattern.py
```

The script processes two example texts by default:
1. "To be or not to be, that is the question"
2. "The cat sat on the mat"

## Advanced: Head Analysis and Optimization

### 1. Checking Heads of Trained Model

Identify head specialization patterns:

```python
import torch
import numpy as np
from attention_pattern import GPTTransformerWithAttention

def analyze_head_importance(model, test_texts, tokenizer):
    """Analyze which heads are most important"""
    head_activations = {f"L{l}H{h}": [] for l in range(6) for h in range(8)}
    
    for text in test_texts:
        input_ids = torch.tensor([tokenizer.encode(text)])
        with torch.no_grad():
            _, attn_weights = model.forward_with_attention(input_ids)
        
        for layer_idx in range(6):
            for head_idx in range(8):
                # Measure attention entropy (lower = more focused)
                attn = attn_weights[layer_idx][0, head_idx]
                entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1).mean()
                head_activations[f"L{layer_idx}H{head_idx}"].append(entropy.item())
    
    # Average importance across all test samples
    head_importance = {k: np.mean(v) for k, v in head_activations.items()}
    return head_importance
```

**What to look for:**
- **High entropy heads:** Diffuse attention (potentially dead or unused)
- **Low entropy heads:** Focused attention (actively learning patterns)
- **Consistent patterns:** Heads that behave similarly across different inputs

### 2. Head Pruning and Reset via Fine-Tuning

Once you've identified underperforming heads, you can optimize your model:

#### Head Pruning

Remove dead or redundant heads to reduce model size:

```python
def prune_heads(model, heads_to_prune):
    """
    Prune specified attention heads
    heads_to_prune: dict like {layer_idx: [head_idx1, head_idx2]}
    """
    for layer_idx, head_indices in heads_to_prune.items():
        # Create mask to zero out specific heads
        # This is a simplified example - full implementation would
        # restructure the weight matrices
        pass  # Implement based on your architecture
```

#### Head Reset and Fine-Tuning

Reset underperforming heads and retrain:

```python
def reset_heads(model, heads_to_reset):
    """
    Reset specific attention heads to random initialization
    heads_to_reset: dict like {layer_idx: [head_idx1, head_idx2]}
    """
    for layer_idx, head_indices in heads_to_reset.items():
        with torch.no_grad():
            # Reset Q, K, V projections for these heads
            # Assuming heads are organized in weight matrices
            for head_idx in head_indices:
                start_dim = head_idx * 64
                end_dim = start_dim + 64
                
                # Reset weights
                torch.nn.init.xavier_uniform_(
                    model.w_q.weight[:, start_dim:end_dim]
                )
                torch.nn.init.xavier_uniform_(
                    model.w_k.weight[:, start_dim:end_dim]
                )
                torch.nn.init.xavier_uniform_(
                    model.w_v.weight[:, start_dim:end_dim]
                )

def fine_tune_with_head_analysis(model, train_data, epochs=5):
    """
    Fine-tune model after head modifications
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in train_data:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
        
        # Re-analyze heads after each epoch
        head_importance = analyze_head_importance(model, test_texts, tokenizer)
        print(f"Epoch {epoch}: Head importance updated")
```

### Best Practices for Head Analysis

1. **Use diverse test samples:** Analyze heads across multiple types of text
2. **Look for specialization:** Healthy models have heads with distinct behaviors
3. **Monitor during training:** Track head development throughout training
4. **Prune conservatively:** Start by removing only the most dead heads
5. **Validate after changes:** Always test model performance after pruning/resetting

## Interpreting Attention Patterns

### Common Head Types

- **Previous Token Heads:** Attend primarily to the previous position
- **First Token Heads:** Attend to the first token (often used for aggregation)
- **Induction Heads:** Copy patterns from earlier in the sequence
- **Syntactic Heads:** Focus on grammatical relationships
- **Semantic Heads:** Capture meaning-based connections

### Red Flags

- **Uniform attention:** All positions get equal attention (dead head)
- **Single position:** Only attends to one position (potentially stuck)
- **No variation across inputs:** Same pattern regardless of text (not learning)

## Dependencies

- PyTorch
- matplotlib
- seaborn
- numpy
- transformers (for GPT2Tokenizer)

## Model Requirements

The visualization requires:
- A trained model saved as `../trained_model.pth`
- Model architecture matching `GPTTransformer` from `transformer_gpt.py`
- GPT-2 tokenizer vocabulary

## Output Format

All images are saved as high-resolution PNGs (150 DPI):
- Suitable for papers, presentations, and detailed analysis
- Color-coded heatmaps for easy interpretation
- Token labels when sequence length permits
