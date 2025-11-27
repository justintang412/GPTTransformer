import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class EmbeddingModel(nn.Module):
    """
    A simple embedding model that uses a pre-trained transformer (like GPT-2)
    to generate sentence embeddings.
    """
    def __init__(self, model_name: str = 'gpt2'):
        """
        Initializes the model by loading a pre-trained transformer.
        
        Args:
            model_name (str): The name of the pre-trained model to use from Hugging Face.
        """
        super().__init__()
        # Load the pre-trained transformer model. We only need the core transformer,
        # not the language modeling head.
        self.transformer = GPT2Model.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate sentence embeddings.
        
        Args:
            input_ids (torch.Tensor): Token IDs of the input text, shape (batch_size, seq_length).
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding tokens.
        
        Returns:
            torch.Tensor: The sentence embedding, shape (batch_size, hidden_size).
        """
        # Get the hidden states from the base transformer model.
        # `outputs.last_hidden_state` contains the output of the final transformer layer.
        # Shape: (batch_size, sequence_length, hidden_size)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # To get a single vector representation for the entire sentence (pooling),
        # we will use mean pooling. We average the embeddings of all non-padding tokens.
        
        # Expand the attention mask to match the dimensions of the hidden state tensor
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum the embeddings of non-padding tokens
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        
        # Count the number of non-padding tokens
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        # Calculate the mean
        mean_pooled_embedding = sum_embeddings / sum_mask
        
        return mean_pooled_embedding

def demonstrate_embedding_generation():
    """
    A demonstration of how to use the EmbeddingModel to generate embeddings for a list of sentences.
    """
    print("="*60)
    print("Demonstrating Sentence Embedding Generation")
    print("="*60)
    
    # --- 1. Setup ---
    # Use CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the tokenizer and the embedding model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add a padding token to the tokenizer. This is crucial for batching sentences of different lengths.
    tokenizer.pad_token = tokenizer.eos_token
    
    model = EmbeddingModel(model_name='gpt2').to(device)
    model.eval()  # Set the model to evaluation mode
    
    # --- 2. Prepare Input Sentences ---
    sentences = [
        "The cat sat on the mat.",
        "A dog was chasing a ball in the park.",
        "Artificial intelligence is transforming the world.",
        "Natural language processing enables computers to understand text."
    ]
    sentences1 = [
        "The cat sleeps on the mat",
        "A rocket launched into the sky",
        "She baked a delicious chocolate cake",
        "The dog rests on the rug",
        "Mountains tower above the quiet valley"
    ]
    print(f"\nInput sentences:\n{sentences}")
    
    # --- 3. Tokenize and Batch ---
    # The tokenizer handles padding and creates an attention mask for us.
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    
    print(f"\nTokenized Input IDs Shape: {inputs['input_ids'].shape}")
    print(f"Attention Mask Shape: {inputs['attention_mask'].shape}")
    
    # --- 4. Generate Embeddings ---
    with torch.no_grad():  # Disable gradient calculation for inference
        embeddings = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    # --- 5. Inspect the Output ---
    print(f"\nGenerated Embeddings Shape: {embeddings.shape}")
    print("This shape means (batch_size, embedding_dimension).")
    
    # --- 6. (Optional) Calculate Similarity ---
    # We can use cosine similarity to see how semantically related the sentences are.
    # Normalize embeddings to unit length
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # Calculate cosine similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    
    print("\n" + "="*60)
    print("Cosine Similarity Matrix:")
    print("(Higher values mean more semantically similar)")
    print("="*60)
    print(similarity_matrix.cpu().numpy().round(3))
    print("\nNotice the high similarity between the two AI-related sentences (rows/cols 2 and 3).")


if __name__ == "__main__":
    demonstrate_embedding_generation()
