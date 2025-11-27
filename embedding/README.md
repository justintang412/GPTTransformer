# Embedding

## 1. Key Components: Tokenizer and Model

A key concept is that the **tokenizer** and the **transformer model** are different components, even though they come from the same "gpt2" family on Hugging Face.

*   **Tokenizer (`GPT2Tokenizer`):** This is **not a neural network**. It is a data-preparation component. Its job is to convert text into a sequence of integer IDs using a predefined vocabulary and a set of rules (like Byte-Pair Encoding).
*   **Model (`GPT2Model`):** This **is the neural network**. It contains the architecture (embedding layers, attention blocks) and all the learned weights. Its job is to take the integer IDs from the tokenizer and perform complex computations on them.

So, when we say they are "different components," we mean they are distinct software objects with different purposes: one for preparing data, the other for processing it.

## 2. How Sentence Embeddings are Calculated

The process transforms a batch of sentences into a batch of fixed-size embedding vectors. This involves several steps where the shape of the data tensor changes.

*   **Initial Input:** A list of sentences.
*   **Tokenization:** The `GPT2Tokenizer` converts the sentences into integer IDs.
    *   **`input_ids` Shape:** `(batch_size, seq_length)`
*   **Transformer Processing:** The `input_ids` are fed into the `GPT2Model`. The model's attention mechanism processes the tokens in context with each other.
    *   **`last_hidden_state` Shape:** `(batch_size, seq_length, hidden_size)`
    *   This tensor contains a unique, contextualized vector for every token in every sentence.
*   **Mean Pooling (Summarization):** To get a single vector for each sentence, the token vectors are averaged.
    1.  **Masking:** An `attention_mask` is used to mathematically ignore padding tokens during the calculation.
    2.  **Summation:** The contextualized vectors for all non-padding tokens in a sentence are summed together.
        *   **`sum_embeddings` Shape:** `(batch_size, hidden_size)`
    3.  **Division:** The summed vector is divided by the number of real tokens to get the final average.
        *   **`mean_pooled_embedding` Shape:** `(batch_size, hidden_size)`

## 3. How Similarity is Calculated with Linear Algebra

The goal is to efficiently compute the **cosine similarity** between every pair of sentence embeddings. This is achieved through two key algebraic principles.

*   **The Principle of L2 Normalization:**
    *   An embedding vector can be thought of as an arrow in a high-dimensional space. Its **direction** represents its semantic meaning, and its **length** is its magnitude.
    *   The **L2 Norm** is the standard Euclidean length of this vector (calculated as `√(v₁² + v₂² + ...)`).
    *   **Normalization** is the process of dividing a vector by its L2 norm. This forces the vector to have a length of **1** while keeping its direction identical. The result is called a **unit vector**.
    *   This is critical because the cosine similarity formula `(A · B) / (||A|| * ||B||)` simplifies to just the dot product `A · B` when the lengths `||A||` and `||B||` are 1. Normalization isolates the directional (semantic) aspect of the vectors for a pure similarity comparison.

*   **The Convenience of Matrix Multiplication:**
    *   To calculate the dot product between all pairs of sentence embeddings, a single, highly optimized matrix multiplication is used: `torch.matmul(embeddings, embeddings.T)`.
    *   This operation takes the `(batch_size, hidden_size)` matrix of embeddings and multiplies it by its transpose `(hidden_size, batch_size)`.
    *   The result is a `(batch_size, batch_size)` **similarity matrix** where the value at `matrix[i, j]` is the cosine similarity between sentence `i` and sentence `j`. This leverages the power of GPUs to perform a complex set of comparisons in one fast and scalable step.

## Output
Input sentences:
['The cat sat on the mat.', 'A dog was chasing a ball in the park.', 'Artificial intelligence is transforming the world.', 'Natural language processing enables computers to understand text.']

Tokenized Input IDs Shape: torch.Size([4, 10])
Attention Mask Shape: torch.Size([4, 10])

Generated Embeddings Shape: torch.Size([4, 768])
This shape means (batch_size, embedding_dimension).

============================================================
Cosine Similarity Matrix:
(Higher values mean more semantically similar)
============================================================
[[1.    0.999 0.994 0.994]
 [0.999 1.    0.995 0.995]
 [0.994 0.995 1.    0.999]
 [0.994 0.995 0.999 1.   ]]

Another test:
============================================================
Demonstrating Sentence Embedding Generation
============================================================
Using device: cpu

Input sentences:
['The cat sleeps on the mat', 'A rocket launched into the sky', 'She baked a delicious chocolate cake', 'The dog rests on the rug', 'Mountains tower above the quiet valley']

Tokenized Input IDs Shape: torch.Size([5, 7])
Attention Mask Shape: torch.Size([5, 7])

Generated Embeddings Shape: torch.Size([5, 768])
This shape means (batch_size, embedding_dimension).

============================================================
Cosine Similarity Matrix:
(Higher values mean more semantically similar)
============================================================
[[1.    0.997 0.997 0.999 0.999]
 [0.997 1.    0.997 0.997 0.997]
 [0.997 0.997 1.    0.996 0.996]
 [0.999 0.997 0.996 1.    0.997]
 [0.999 0.997 0.996 0.997 1.   ]]