# I have modified the original script to use wmt14.
# Reference: https://medium.com/towards-data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

from Model import Transformer
from DE_EN_Dataset import DE_EN_Dataset
import torch
import torch.nn as nn
import torch.optim as optim

# Import hyperparameters
from Params import *
# Other values
model_path = 'models/transformer_wmt14' # Path to save model

###############################################

# Main
def train(model_path):
    # Load dataset
    training_data = DE_EN_Dataset(
        train_dataset_path_de, 
        train_dataset_path_en, 
        de_tokenizer_name, 
        en_tokenizer_name,
        max_seq_length
    )

    # Init model
    transformer = Transformer(
        vocab_size, 
        vocab_size, 
        d_model, 
        num_heads, 
        num_layers, 
        d_ff, 
        max_seq_length, 
        dropout
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    transformer.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_num in range(training_data.__len__()):
            optimizer.zero_grad()
            de_tokens, en_tokens = training_data.__getitem__(batch_num)

            #print ('de_tokens', de_tokens)
            #print ('en_tokens', en_tokens)

            de_tensor_train = to_torch_tensor(de_tokens, 1, max_seq_length) # TODO -- larger batch size
            en_tensor_train = to_torch_tensor(en_tokens, 1, max_seq_length)

            output = transformer(de_tensor_train, en_tensor_train[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size), en_tensor_train[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print end of epoch stats
        avg_epoch_loss = epoch_loss / training_data.__len__()
        print(f"Epoch: {epoch+1}, Loss: {avg_epoch_loss}")

        # Save model
        if epoch % 10 == 0:
            save_model(model_path, epoch, transformer, optimizer)

    # Save final model
    save_model(model_path, epoch, transformer, optimizer)

# Convert list to torch.tensor
def to_torch_tensor(tokens, num_sequences, max_seq_length):
    num_tokens = num_sequences*max_seq_length
    tokens = tokens[:num_tokens]
    tokens_tensor = torch.tensor(tokens)
    tokens_tensor = tokens_tensor[:num_sequences * max_seq_length].view(num_sequences, max_seq_length)
    #print (tokens_tensor.shape)
    return tokens_tensor

# Save model to disk
def save_model(model_path, epoch, transformer, optimizer):
    full_model_path = model_path + '_epoch' + str(epoch) + '.pt'
    # TODO - add loss
    torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        full_model_path
    )

if __name__ == "__main__":
    train(model_path)