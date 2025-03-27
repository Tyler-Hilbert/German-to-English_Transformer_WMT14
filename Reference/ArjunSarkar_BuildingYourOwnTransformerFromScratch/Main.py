# I have modified the original script to use wmt14.

from Model import Transformer
from SentencePairDataset import SentencePairDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import hyperparameters
from Params import *
# Other values
model_path = 'models/transformer_wmt14' # Path to save model

###############################################

# Main
def train(model_path):
    # Select CUDA, mps or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load dataset
    training_data = SentencePairDataset(
        train_dataset_path_de, 
        train_dataset_path_en, 
        de_tokenizer_name, 
        en_tokenizer_name,
        max_seq_length
    )

    # Data loader
    # TODO -- optimize parameters
    training_generator = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
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
        dropout,
        device
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    transformer.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_de_tokens, batch_en_tokens in training_generator:
            batch_de_tokens = batch_de_tokens.to(device)
            batch_en_tokens = batch_en_tokens.to(device)

            optimizer.zero_grad()

            #print ('batch_de_tokens\n', batch_de_tokens)
            #print ('batch_en_tokens\n', batch_en_tokens)

            output = transformer(batch_de_tokens, batch_en_tokens[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size), batch_en_tokens[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            #print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        # Print end of epoch stats
        avg_epoch_loss = epoch_loss / training_data.__len__()
        print(f"Epoch: {epoch+1}, Loss: {avg_epoch_loss}")

        # Save model
        if epoch % 5 == 0:
            save_model(model_path, epoch, transformer, optimizer)

    # Save final model
    save_model(model_path, epoch, transformer, optimizer)

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