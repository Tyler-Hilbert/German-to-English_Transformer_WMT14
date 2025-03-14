# Test trained model with string

from Model import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu

# Import hyperparameters
from Params import *
# Other values
model_path = 'models/transformer_wmt14_epoch1999.pt'
validation_dataset_path_input = '../data/data_en_train.txt'
validation_dataset_path_expected_outputs = '../data/data_de_train.txt'

######################################################

# Load model
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])
transformer.eval()

# Load dataset
inputs = []
expected_outputs = []
with open(validation_dataset_path_input, 'r') as file:
    for line in file:
        inputs.append(line.strip())
with open(validation_dataset_path_expected_outputs, 'r') as file:
    for line in file:
        expected_outputs.append(line.strip())

# Test dataset
for input, expected_output in zip(inputs, expected_outputs):
    # Tokenize
    en = en_tokenizer(input).input_ids
    padding_length = max_seq_length - len(en)
    en += [0] * padding_length
    en = torch.tensor(en)
    en = en[:1 * max_seq_length].view(1, max_seq_length)

    de = de_tokenizer("").input_ids
    padding_length = max_seq_length - len(de)
    de += [0] * padding_length
    de = torch.tensor(de)
    de = de[:1 * max_seq_length].view(1, max_seq_length)

    # Inference
    output = transformer(en, de[:, :-1])
    output_ids = output.argmax(dim=-1).squeeze(0).tolist()
    decoded_text = de_tokenizer.decode(output_ids, skip_special_tokens=True)

    # Score
    print ('Input:', input)
    print('Decoded output:', decoded_text)
    print ('expected output', expected_output)
    bleu_score = sentence_bleu(decoded_text, expected_output)
    print ('BLEU score', bleu_score)
    print ('\n\n')

