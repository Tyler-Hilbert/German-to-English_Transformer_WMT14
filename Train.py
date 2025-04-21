# The original transformer model proposed in `Attention Is All You Need`.
# Implemented in PyTorch using WMT14 (DE to EN).

from Model import Transformer
from Config import config # THIS MUST BE LOADED FOR FUNCTIONS TO WORK!!!
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Main
def train(model_path):
    device = get_device()

    # Load tokenizer, dataset, dataloader, model, criterion and optimizer
    de_tokenizer = AutoTokenizer.from_pretrained(config.de_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(config.en_tokenizer_name)
    training_data = get_dataset()
    training_generator = get_dataloader(training_data)
    transformer = get_model(de_tokenizer, en_tokenizer, device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = get_optimizer(transformer)

    # Training loop
    transformer.train()
    for epoch in range(config.epochs):
        start_time = time.time()
        epoch_loss = 0

        for step, data in enumerate(training_generator):
            batch_de_tokens = get_batch_tokens(data['translation']['de'], de_tokenizer, device)
            batch_en_tokens = get_batch_tokens(data['translation']['en'], en_tokenizer, device)

            # Training
            optimizer.zero_grad()
            output = transformer(batch_de_tokens, batch_en_tokens[:, :-1])
            loss = criterion(output.contiguous().view(-1, en_tokenizer.vocab_size), batch_en_tokens[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            # Training stats
            epoch_loss += loss.item()
            if step % 100 == 0:
                print (f'Step {step}, Loss (last training example) {loss.item()}, Time elapse since start of Epoch {time.time() - start_time}')

            # Save model
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


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_dataset():
    return load_dataset('wmt/wmt14', 'de-en', split='train')

def get_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

def get_model(de_tokenizer, en_tokenizer, device):
    return Transformer(
        src_vocab_size= de_tokenizer.vocab_size,
        tgt_vocab_size= en_tokenizer.vocab_size,
        d_model=        config.d_model,
        num_heads=      config.num_heads,
        num_layers=     config.num_layers,
        d_ff=           config.d_ff,
        max_seq_length= config.max_seq_length,
        dropout=        config.dropout,
        device=         device
    ).to(device)

def get_optimizer(transformer):
    return optim.Adam(
        transformer.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

def get_batch_tokens(data, tokenizer, device):
    return torch.tensor(
        tokenizer(
            data,
            truncation=True,
            padding='max_length',
            max_length=config.max_seq_length
        ).input_ids
    ).to(device)

# Save model to disk
def save_model(model_path, epoch, transformer, optimizer):
    full_model_path = model_path + '_epoch' + str(epoch+1) + '.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        full_model_path
    )

if __name__ == "__main__":
    train(config.model_path)