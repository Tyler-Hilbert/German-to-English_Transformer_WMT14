from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers, decoders
import pandas as pd
import torch
import re
import os

_MAX_LEN = 512
_DEFAULT_TOKENIZER_FILE = "wmt14_de_en_tokenizer.json"

# ========================================================
# train tokenizer
# ========================================================
def create_tokenizer(tokenizer_pth, train_file):
    def get_training_corpus(train_file):
        i = 0
        for chunk in pd.read_csv(train_file, chunksize=1000, usecols=["de", "en"], lineterminator="\n"):
            # Drop rows with NaN values in either column
            chunk = chunk.dropna(subset=["de", "en"])
            # Convert all entries to strings to avoid type errors
            combined_text = chunk["de"].astype(str).tolist() + chunk["en"].astype(str).tolist()
            print("Done:", i, "rows", end="\r")
            i += 1000
            yield combined_text
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=37000, special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"])
    
    # Train the tokenizer with data yielded from get_training_corpus
    tokenizer.train_from_iterator(get_training_corpus(train_file), trainer=trainer)
    tokenizer.save(tokenizer_pth)

# ========================================================
# get tokenizer
# ========================================================
def get_tokenizer(tokenizer_pth=None, train_file=None):
    if tokenizer_pth is None:
        tokenizer_pth = _DEFAULT_TOKENIZER_FILE
    if not os.path.exists(tokenizer_pth):
        if train_file is None:
            raise ValueError("Tokenizer file does not exist. You can create a new tokenizer by providing a training file.")
        create_tokenizer(tokenizer_pth, train_file) # create new tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_pth)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=_MAX_LEN)
    tokenizer.decoder = decoders.ByteLevel()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return tokenizer

# ========================================================
# greedy decoding
# ========================================================
def generate_text(enc_text, model, tokenizer=None, max_len=10):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    enc_inp = tokenizer.encode(enc_text)
    enc_ids = torch.tensor(enc_inp.ids).to(device)
    # ! enc input must also have BOS and EOS tokens
    if enc_ids[0] != tokenizer.token_to_id("[BOS]"):
        enc_ids = torch.cat([torch.tensor([tokenizer.token_to_id("[BOS]")]).to(device), enc_ids])
    if enc_ids[-1] != tokenizer.token_to_id("[EOS]"):
        enc_ids = torch.cat([enc_ids, torch.tensor([tokenizer.token_to_id("[EOS]")]).to(device)])

    # Add batch dimension if needed
    if len(enc_ids.shape) == 1:
        enc_ids = enc_ids.unsqueeze(0)

    # Initialize token history with BOS token ID
    bos_token_id = tokenizer.token_to_id("[BOS]")
    gen_ids = [bos_token_id]
    memory = None

    # Start generation loop
    for i in range(max_len):
        prev_dec_inputs = torch.tensor(gen_ids).unsqueeze(0).to(device)

        if memory is None:
            out = model(dec_x=prev_dec_inputs, enc_x=enc_ids)
            memory = out["memory"]
        else:
            out = model(dec_x=prev_dec_inputs, memory=memory)

        # Get logits of the last token
        logits = out["logits"]
        # Get probabilities
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        # Get the token with the highest probability
        next_token_id = torch.argmax(probs).item()
        # Add the token ID to the history
        gen_ids.append(next_token_id)

        # Check if the next token is EOS token
        if next_token_id == tokenizer.token_to_id("[EOS]"):
            break

    print("Generated token IDs:", gen_ids)
    # Decode the token IDs to get the generated text
    generated_text = tokenizer.decode(gen_ids)
    return generated_text


# ========================================================
# Load last check point (if any)
# ========================================================
def load_last_checkpoint(model, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return 0
    checkpoints = [f for f in os.listdir(folder) if re.match(r'transformer_epoch_\d+.pt', f)]
    if not checkpoints:
        return 0
    latest = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
    model.load_state_dict(torch.load(folder + latest, map_location=torch.device('cpu')))
    last_epoch = int(re.search(r'\d+', latest).group())
    print(f"Loaded checkpoint {latest} at epoch {last_epoch}")
    return last_epoch
    
