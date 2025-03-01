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
epochs = 350
lr = 0.0001
num_sequences = 204 # The number of training sequences. Will be multiplied by `max_sequence_length` to get number of tokens.
tokens_minimum = 20480 # Minimum number of tokens used for training. Should be within 100 of `num_sequences*max_sequence_length`.
vocab_size = 30000 # TODO - use exact embeddings size

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

#assert (len(en_tokens_full) > 33800 and len(en_tokens_full) < 33900)
#assert (len(de_tokens_full) > 34000 and len(en_tokens_full) < 35000)
assert (len(en_tokens_full) > tokens_minimum)
assert (len(de_tokens_full) > tokens_minimum)

# Trim token to fit (and in this case to reduce training set size)
num_tokens = num_sequences*max_seq_length
print (num_tokens)
# En tensor
en_tokens = en_tokens_full[:num_tokens]
en_tensor = torch.tensor(en_tokens)
en_tensor = en_tensor[:num_sequences * max_seq_length].view(num_sequences, max_seq_length)
print (en_tensor.shape)
# De tensor
de_tokens = de_tokens_full[:num_tokens]
de_tensor = torch.tensor(de_tokens)
de_tensor = de_tensor[:num_sequences * max_seq_length].view(num_sequences, max_seq_length)
print (de_tensor.shape)

#print (max(en_tokens))
#print (max(de_tokens))

# Model
transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# Train
transformer.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = transformer(en_tensor, de_tensor[:, :-1])
    loss = criterion(output.contiguous().view(-1, vocab_size), de_tensor[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")