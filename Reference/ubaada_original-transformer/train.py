import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import wandb  
import kagglehub
import os
import pandas as pd
import random

from model import Transformer
from utils import generate_text, load_last_checkpoint, get_tokenizer

# ========================================================
# Download dataset and/or set path
# ========================================================
path = kagglehub.dataset_download("mohamedlotfy50/wmt-2014-english-german")

train_file = path + "/wmt14_translate_de-en_train.csv"
val_file = path + "/wmt14_translate_de-en_validation.csv"
test_file = path + "/wmt14_translate_de-en_test.csv"

# ========================================================
# Initalise tokenizer
# ========================================================
print("Loading Tokenizer...")
tokenizer = get_tokenizer(train_file=train_file)
print(tokenizer.encode("this is my cat").tokens)

# ========================================================
# Dataset Loaders
# ========================================================

def process_batch(batch, tokenizer, device):
    # batch is a list of tuples (en, de)
    # tokenizing collectively for padding
    en_obj = tokenizer.encode_batch([x[0] for x in batch], add_special_tokens=True)
    en = torch.tensor([x.ids for x in en_obj]).to(device,non_blocking=True)
    de_obj = tokenizer.encode_batch([x[1] for x in batch], add_special_tokens=True)
    de = torch.tensor([x.ids for x in de_obj]).to(device, non_blocking=True)
    en_padding_mask = torch.tensor([x.attention_mask for x in en_obj]).to(device,non_blocking=True)

    #prepare for training
    de_input = de[:, :-1]  # remove the last token
    de_target = de[:, 1:]  # remove the [BOS] token
    return {
        "en": en,
        "de_input": de_input,
        "de_target": de_target,
        "en_padding_mask": en_padding_mask
    }

class WMT14Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, lineterminator="\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Data Augmentation to randomly drop punctuation (, . ? !) just from input
        if random.random() < 0.2:
            row["en"] = row["en"].replace(",", "").replace(".", "").replace("?", "").replace("!", "")

        en = "[BOS]" + row["en"] + "[EOS]" # encoder input, not sure if we need BOS and EOS
        de = "[BOS]" + row["de"] + "[EOS]" # decoder input
        return en, de

print("Loading Datasets...")
train_dataset = WMT14Dataset(train_file)
val_dataset = WMT14Dataset(val_file)
test_dataset = WMT14Dataset(test_file)



# ========================================================
# Model Init (base model from paper)
# ========================================================
N = 6
dmodel = 512
dff = 2048
h = 8
dk = dmodel // h
enc_vocab_size = 37000
dec_vocab_size = 37000
dropout = 0.1

model = Transformer(N, N, dmodel, h, enc_vocab_size, dec_vocab_size, dff, dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Initialised Model. Params:", sum(p.numel() for p in model.parameters()))


# ========================================================
# Training
# ========================================================

print("Starting training...")
wandb.init(project="scratch-transformer")
# Or resume from a previous run id
#wandb.init(project="scratch-transformer", id="<run-id>", resume="must")
checkpoint_folder = "./model_weights/"

optimizer = optim.Adam(model.parameters(), lr=0.0001, foreach=True)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

# Training parameters
torch.set_float32_matmul_precision("medium")  # Precision of matrix multiplication
start_epoch = load_last_checkpoint(model, checkpoint_folder)
num_epochs = 25
batch_size = 16
accumulation_steps = 8  # Number of batches to accumulate gradients

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: process_batch(x, tokenizer, device))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: process_batch(x, tokenizer, device))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: process_batch(x, tokenizer, device))

for epoch in tqdm(range(start_epoch+1, num_epochs), desc="Epochs"):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()  # Zero gradients before starting accumulation

    batch_iterator = tqdm(data_loader, desc=f"Training Epoch {epoch}", leave=True)
    
    for step, batch in enumerate(batch_iterator):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                dec_x=batch["de_input"],
                enc_x=batch["en"],
                enc_padding_mask=batch["en_padding_mask"]
            )["logits"]
            loss = criterion(outputs.view(-1, dec_vocab_size), batch["de_target"].reshape(-1))

        # For printing avg loss in the epoch. Nothing to do with grad accum
        total_loss += loss.item()
        
        # Scale the loss by accumulation_steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update tqdm description to show current batch loss
        batch_iterator.set_postfix(batch_loss=loss.item() * accumulation_steps)
        
        # Perform optimizer step every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"batch_loss": loss.item() * accumulation_steps, "epoch": epoch})

        # print the sample translations
        if (step % 20_000)==0:
            s1= generate_text("this is my cat", model, tokenizer, max_len=10)
            s2 = generate_text("The quick brown fox jumps over the lazy dog, swiftly avoiding the deep, mysterious forest that lies ahead.", model, tokenizer, max_len=70)
            print(s1)
            print(s2 + "\n")
            wandb.log({"example_sentence_1": s1, "example_sentence_2": s2})

    
    # Perform an optimizer step for remaining gradients
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    average_loss = total_loss / len(data_loader)
    wandb.log({"epoch_loss": average_loss, "epoch": epoch})
    print(f"Epoch {epoch}, Average Loss: {average_loss:.4f}")

    # -------------------------------
    # Validation Phase   (per epoch)
    # -------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            outputs = model(
                dec_x=val_batch["de_input"],
                enc_x=val_batch["en"],
                enc_padding_mask=batch["en_padding_mask"]
            )["logits"]
            loss = criterion(outputs.view(-1, dec_vocab_size), val_batch["de_target"].reshape(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    wandb.log({"val_loss": val_loss, "epoch": epoch})
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
    
    # Save the model checkpoint and upload it to wandb
    model_path = f"{checkpoint_folder}/transformer_epoch_{epoch}.pt"
    torch.save(model.state_dict(), model_path)
    if (epoch % 5) == 0:
        wandb.save(model_path)




# ========================================
# Test Phase (after all epochs)
# ========================================
model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_batch in test_loader:
        outputs = model(
            dec_x=test_batch["de_input"],
            enc_x=test_batch["en"],
            enc_padding_mask=batch["en_padding_mask"]
        )["logits"]
        loss = criterion(outputs.view(-1, dec_vocab_size), test_batch["de_target"].reshape(-1))
        test_loss += loss.item()

test_loss /= len(test_loader)
wandb.log({"test_loss": test_loss})
print(f"Final Test Loss: {test_loss:.4f}")
