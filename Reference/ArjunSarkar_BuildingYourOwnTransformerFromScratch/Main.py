# I have modified the original script to use wmt14.
# Reference: https://medium.com/towards-data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

from Model import Transformer

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Import hyperparameters
from Params import *
# Other values
model_path = 'models/transformer_wmt14' # Path to save model

####### UTILS ###############################################

# Convert `file_path`, a file of space separated sentences, to tokens using `tokenizer`
def tokenize(file_path, tokenizer):
    tokens = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens += tokenizer(line.strip()).input_ids
            break # FIXME this is just for testing
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
en_tokens_train = tokenize(train_dataset_path_input, en_tokenizer)
en_tokens_train += [0]*num_sequences*max_seq_length # FIXME
print (en_tokens_train)
#   DE tokens
de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
de_tokens_train = tokenize(train_dataset_path_expected_outputs, de_tokenizer)
de_tokens_train += [0]*num_sequences*max_seq_length # FIXME
print (de_tokens_train)

# Batchify EN and DE lists
en_tensor_train = batchify(en_tokens_train, num_sequences, max_seq_length)
print (en_tensor_train)
de_tensor_train = batchify(de_tokens_train, num_sequences, max_seq_length)
print (de_tensor_train)

# Model
transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# Train
transformer.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = transformer(de_tensor_train, en_tensor_train[:, :-1])
    loss = criterion(output.contiguous().view(-1, vocab_size), en_tensor_train[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # Save model
    if epoch % 25 == 0:
        save_model(model_path, epoch, transformer, optimizer, loss)


# Save final model
save_model(model_path, epoch, transformer, optimizer, loss)