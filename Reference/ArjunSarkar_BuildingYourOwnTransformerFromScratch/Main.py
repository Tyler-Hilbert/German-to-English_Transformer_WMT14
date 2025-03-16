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

###############################################

# Main
def train(model_path):
    # Load dataset and tokenize
    en_tokens_train = load_file_and_tokenize(train_dataset_path_input, 'bert-base-uncased')
    de_tokens_train = load_file_and_tokenize(train_dataset_path_expected_outputs, 'bert-base-german-cased')

    # Batchify (move to PyTorch tensor)
    en_tensor_train = batchify(en_tokens_train, num_sequences, max_seq_length)
    de_tensor_train = batchify(de_tokens_train, num_sequences, max_seq_length)

    # Init model
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

# Convert `file_path`, a file of space separated sentences using AutoTokenizer `tokenizer_name`
def load_file_and_tokenize(file_path, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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

# Save model to disk
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

if __name__ == "__main__":
    train(model_path)