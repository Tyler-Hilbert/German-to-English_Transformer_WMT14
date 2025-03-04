# Test trained model with string

from Model import Transformer

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Test STR
test_str = "Resumption of the session"

# Hyperparameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 160
dropout = 0.1
vocab_size = 30000 # TODO - use exact embeddings size

# Other values
model_path = 'models/transformer_wmt14_epoch106.pt'
en_end_token = 102
de_end_token = 4

# Tokenize
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
en = en_tokenizer(test_str).input_ids
padding_length = max_seq_length - len(en)
en += [102] * padding_length
en = torch.tensor(en)
en = en[:1 * max_seq_length].view(1, max_seq_length)

de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
de = de_tokenizer("").input_ids
padding_length = max_seq_length - len(de)
de += [4] * padding_length
de = torch.tensor(de)
de = de[:1 * max_seq_length].view(1, max_seq_length)

# Load model
transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])

# Test output
transformer.eval()
output = transformer(en, de[:, :-1])
print (output)

output_ids = output.argmax(dim=-1).squeeze(0).tolist()
print("Generated Token IDs:", output_ids)

decoded_text = de_tokenizer.decode(output_ids, skip_special_tokens=True)
print("Decoded Output:", decoded_text)
