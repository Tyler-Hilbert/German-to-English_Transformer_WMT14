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
model_path = 'models/transformer_wmt14_epoch99.pt'

######################################################

def validation():
    # Load model
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
    de_tokenizer = AutoTokenizer.from_pretrained(de_tokenizer_name)
    transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])
    transformer.eval()

    # Load datasets
    en_dataset = load_file_by_line(train_dataset_path_en)
    de_dataset = load_file_by_line(train_dataset_path_de)

    for en_text, de_text in zip(en_dataset, de_dataset):
        en_tokens = tokenize(en_text, en_tokenizer)
        de_tokens = tokenize(de_text, de_tokenizer)

        print ('en_tokens', en_tokens)
        print ('de_tokens', de_tokens)

        # Inference
        output = transformer(de_tokens, en_tokens[:, :-1])
        output_ids = output.argmax(dim=-1).squeeze(0).tolist()
        #print ('output_ids', output_ids)
        decoded_text = en_tokenizer.decode(output_ids, skip_special_tokens=True)
        #print ('decoded_text', decoded_text)

        # Score
        print ('Input:', de_text)
        print('Decoded output:', decoded_text)
        print ('expected output', en_text)
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
def tokenize(sentence, tokenizer):
    tokens = tokenizer(sentence).input_ids
    return torch.tensor([tokens])

if __name__ == "__main__":
    validation()