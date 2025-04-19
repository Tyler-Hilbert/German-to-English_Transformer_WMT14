# Deprecated

# Test trained model from validation dataset on disk

from Model import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu

# Import hyperparameters
from Config import *
# Other values
model_path = 'models/model1_de-en.pt'
verbose = False

######################################################

def validation():
    # Select CUDA, mps or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load model

    # FIXME -- vocab size
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
    de_tokenizer = AutoTokenizer.from_pretrained(de_tokenizer_name)
    transformer = Transformer(
        30000,
        30522,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        device
    ).to(device)
    transformer.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])
    transformer.eval()

    # Load datasets
    # TODO - convert this to load from dataset instead of file
    en_dataset = load_file_by_line(validation_dataset_path_en)
    de_dataset = load_file_by_line(validation_dataset_path_de)

    for en_text, de_text in zip(en_dataset, de_dataset):
        en_tokens = tokenize(en_text, en_tokenizer, device)
        de_tokens = tokenize(de_text, de_tokenizer, device)
        de_tokens = de_tokens.to(device)

        if verbose:
            print ('en_tokens', en_tokens)
            print ('de_tokens', de_tokens)

        # Inference
        output = transformer(de_tokens, en_tokens[:, :-1])
        output_ids = output.argmax(dim=-1).squeeze(0).tolist()
        #print ('output_ids', output_ids)
        decoded_text = en_tokenizer.decode(output_ids, skip_special_tokens=True)
        #print ('decoded_text', decoded_text)

        # Score
        if verbose:
            print ('Input:', de_text)
        print ('Decoded output  :', decoded_text)
        print ('Expected output :', en_text)
        decoded_text = decoded_text.lower().split(' ')
        expected_output = en_text.lower().split(' ')
        bleu_score = sentence_bleu([expected_output], decoded_text)
        print ('BLEU score', bleu_score)
        print ('\n\n')

# Loads a single file line by line
def load_file_by_line(dataset):
    sentences = []
    with open(dataset, 'r') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences

# Tokenizes and returns a torch tensor
def tokenize(sentence, tokenizer, device):
    tokens = tokenizer(sentence).input_ids
    return torch.tensor([tokens]).to(device)

if __name__ == "__main__":
    validation()