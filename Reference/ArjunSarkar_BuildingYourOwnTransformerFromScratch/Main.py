# I have modified the original script to use wmt14.
# Reference: https://medium.com/towards-data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

from Model import Transformer

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Hyperparameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Load tokens
# TODO - improve tokenization process
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
en_tokens_full = []
with open('data/data_en.txt', 'r') as file:
    for line in file:
        en_tokens_full += en_tokenizer(line.strip()).input_ids

de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
de_tokens_full = []
with open('data/data_de.txt', 'r') as file:
    for line in file:
        de_tokens_full += de_tokenizer(line.strip()).input_ids

assert (len(en_tokens_full) > 7500 and len(en_tokens_full) < 7600)
#assert (len(de_tokens_full) > 7600 and len(en_tokens_full) < 7700)

# Trim token to fit
en_tokens = en_tokens_full[:7500]
en_tensor = torch.tensor(en_tokens)
en_tensor = en_tensor[:75 * 100].view(75, 100)
print (en_tensor.shape)

de_tokens = de_tokens_full[:7500]
de_tensor = torch.tensor(de_tokens)
de_tensor = de_tensor[:75 * 100].view(75, 100)
print (de_tensor.shape)

#print (max(en_tokens))
#print (max(de_tokens))

# Model
# TODO - don't hardcode vocab size
transformer = Transformer(30000, 30000, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Train
transformer.train()
for epoch in range(2000):
    optimizer.zero_grad()
    output = transformer(en_tensor, de_tensor[:, :-1])
    loss = criterion(output.contiguous().view(-1, 30000), de_tensor[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")