import torch
from gpt_transformer.transformer_gpt import GPTTransformer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode 