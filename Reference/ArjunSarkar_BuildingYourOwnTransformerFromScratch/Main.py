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
epochs = 750
lr = 0.0001
num_sequences = 100 # The number of training sequences. Will be multiplied by `max_sequence_length` to get number of tokens.
vocab_size = 30000 # TODO - use exact embeddings size

# Other values
model_path = 'models/transformer_wmt14' # Path to save model
en_end_token = 102
de_end_token = 4

####### UTILS ###############################################

# Convert `file_path`, a file of space separated sentences, to tokens using `tokenizer`
def tokenize(file_path, tokenizer, padding, max_seq_length):
    tokens = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens += tokenizer(line.strip()).input_ids
    return tokens

# Batchify python list and convert to torch.tensor
def batchify(tokens, num_sequences, max_seq_length):
    num_tokens = num_sequences*max_seq_length
    tokens = tokens[:num_tokens]
    tokens_tensor = torch.tensor(tokens)
    tokens_tensor = tokens_tensor[:num_sequences * max_seq_length].view(num_sequences, max_seq_length)
    print (tokens_tensor.shape)
    return tokens_tensor

def save_model(model_path, epoch, transformer, optimizer, loss):
    full_model_path = model_path + '_epoch' + str(epoch) + '.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        },
        full_model_path
    )

######################################################

# Load tokens
# TODO - improve tokenization process
#   EN tokens
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
en_tokens_train = tokenize('data/data_en_train.txt', en_tokenizer, en_end_token, max_seq_length)
#   DE tokens
de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
de_tokens_train = tokenize('data/data_de_train.txt', de_tokenizer, de_end_token, max_seq_length)

# Batchify EN and DE lists
en_tensor_train = batchify(en_tokens_train, num_sequences, max_seq_length)
de_tensor_train = batchify(de_tokens_train, num_sequences, max_seq_length)

# Model
transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# Train
transformer.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = transformer(en_tensor_train, de_tensor_train[:, :-1])
    loss = criterion(output.contiguous().view(-1, vocab_size), de_tensor_train[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # Save model
    if epoch % 10 == 0:
        save_model(model_path, epoch, transformer, optimizer, loss)


# Save final model
save_model(model_path, epoch, transformer, optimizer, loss)