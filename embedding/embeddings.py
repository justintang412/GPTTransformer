import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the last hidden state.
    This is the method we implemented in embedding.py.
    """
    # The model_output is the full output from the transformer, we want the last_hidden_state
    token_embeddings = model_output.last_hidden_state
    
    # Expand the attention mask to match the dimensions of the token embeddings
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum the embeddings of non-padding tokens and divide by the number of non-padding tokens
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask

def cls_pooling(model_output):
    """
    Performs CLS pooling by simply taking the hidden state of the [CLS] token.
    The [CLS] token is always the first token in the sequence.
    """
    # The model_output is the full output, we want the last_hidden_state
    token_embeddings = model_output.last_hidden_state
    
    # The embedding for the entire sequence is the hidden state of the first token ([CLS])
    # Shape: (batch_size, hidden_size)
    return token_embeddings[:, 0]

def demonstrate_pooling_methods():
    """
    Demonstrates how to get a sentence embedding from a Hugging Face model
    using both Mean Pooling and CLS Pooling.
    """
    print("="*80)
    print("Demonstrating Mean Pooling vs. CLS Pooling")
    print("="*80)

    # --- 1. Load a Model and Tokenizer ---
    # We use a model from sentence-transformers, which is specifically fine-tuned
    # for creating high-quality sentence embeddings.
    # 'all-MiniLM-L6-v2' is a very popular and effective choice.
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # --- 2. Prepare Sentences ---
    sentences = [
        "The cat sat on the mat.",
        "Artificial intelligence is transforming the world."
    ]
    print(f"Input sentences:\n{sentences}\n")

    # --- 3. Tokenize the Sentences ---
    # The tokenizer will automatically add the [CLS] token at the beginning for us.
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Let's inspect the tokenized input for the first sentence
    first_sentence_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Tokens for first sentence: {first_sentence_tokens}")
    print("Notice the '[CLS]' token at the beginning and '[SEP]' at the end.\n")

    # --- 4. Get Model Output ---
    with torch.no_grad():
        # Get the full output from the model, which includes the last_hidden_state
        model_output = model(**inputs)

    print(f"Shape of last_hidden_state: {model_output.last_hidden_state.shape}")
    print("(batch_size, sequence_length, hidden_size)\n")

    # --- 5. Apply Pooling Methods ---

    # Method 1: Mean Pooling
    mean_pooled_embedding = mean_pooling(model_output, inputs['attention_mask'])
    
    # Method 2: CLS Pooling
    cls_pooled_embedding = cls_pooling(model_output)

    # --- 6. Inspect Final Embeddings ---
    print("--- Results ---")
    print(f"Shape of Mean Pooled Embedding: {mean_pooled_embedding.shape}")
    print(f"Shape of CLS Pooled Embedding:  {cls_pooled_embedding.shape}")
    print("\nBoth methods successfully created a single vector per sentence!")
    
    # --- 7. Normalize and Compare ---
    # For sentence-transformer models, it's standard to normalize the output.
    mean_pooled_embedding = torch.nn.functional.normalize(mean_pooled_embedding, p=2, dim=1)
    cls_pooled_embedding = torch.nn.functional.normalize(cls_pooled_embedding, p=2, dim=1)
    
    # Let's see how similar the embeddings are from the two different methods
    similarity = torch.nn.functional.cosine_similarity(mean_pooled_embedding, cls_pooled_embedding)
    print(f"\nCosine similarity between the two sentences (using mean pooling): {similarity[0]:.4f}")
    
    # Note: For models from sentence-transformers, mean pooling is often the recommended default.
    # However, both methods are widely used depending on how the model was trained.

if __name__ == "__main__":
    demonstrate_pooling_methods()
