# The original transformer model proposed in `Attention Is All You Need`.
# Implemented in PyTorch using WMT14 (DE to EN).

from Model import Transformer
from transformers import AutoTokenizer # TODO -- remove when implemented in custom dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datasets import load_dataset

# Import hyperparameters
from Params import *
# Other values
model_path = 'models/wmt14_de-en_full' # Path to save model

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

    # Load data
    training_data = load_dataset('wmt/wmt14', 'de-en', split='train')
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
        start_time = time.time()
        epoch_loss = 0

        for step, data in enumerate(training_generator):
            # DE tokenization
            data_de = data['translation']['de']
            de_tokenizer = AutoTokenizer.from_pretrained(de_tokenizer_name)
            batch_de_tokens = torch.tensor(
                de_tokenizer(
                    data_de,
                    truncation=True,
                    padding='max_length',
                    max_length=max_seq_length
                ).input_ids
            )
            batch_de_tokens = batch_de_tokens.to(device)

            # EN tokenization
            data_en = data['translation']['en']
            en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
            batch_en_tokens = torch.tensor(
                en_tokenizer(
                    data_en,
                    truncation=True,
                    padding='max_length',
                    max_length=max_seq_length
                ).input_ids
            )
            batch_en_tokens = batch_en_tokens.to(device)

            # Training
            optimizer.zero_grad()
            output = transformer(batch_de_tokens, batch_en_tokens[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size), batch_en_tokens[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print (f'Step {step}, Loss (last training example) {loss.item()}, Time elapse since start of Epoch {time.time() - start_time}')
            if step % 1000 == 0:
                save_model(model_path, epoch, transformer, optimizer)

        # Print end of epoch stats
        avg_epoch_loss = epoch_loss / training_data.__len__()
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}, Loss: {avg_epoch_loss}, Epoch Time: {epoch_time}s")

        # Save model
        if epoch % 1 == 0:
            save_model(model_path, epoch, transformer, optimizer)

    # Save final model
    save_model(model_path, epoch, transformer, optimizer)

# Save model to disk
def save_model(model_path, epoch, transformer, optimizer):
    # TODO - epoch off by 1
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