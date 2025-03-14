# Test trained model with string

from Model import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu

# Test STR
test_str = "A Republican strategy to counter the re-election of Obama"
expected_output = "Eine republikanische Strategie gegen die Wiederwahl Obamas"

# Hyperparameters
d_model = 64
num_heads = 1
num_layers = 2
d_ff = 512
max_seq_length = 160
dropout = 0.1
##epochs = 2000
##lr = 0.0001
##num_sequences = 10 # The number of training sequences. Will be multiplied by `max_sequence_length` to get number of tokens.
vocab_size = 30000 # TODO - use exact embeddings size

# Other values
model_path = 'models/transformer_wmt14_epoch1999.pt'

# Tokenize
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
en = en_tokenizer(test_str).input_ids
padding_length = max_seq_length - len(en)
en += [0] * padding_length
en = torch.tensor(en)
en = en[:1 * max_seq_length].view(1, max_seq_length)

de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
de = de_tokenizer("").input_ids
padding_length = max_seq_length - len(de)
de += [0] * padding_length
de = torch.tensor(de)
de = de[:1 * max_seq_length].view(1, max_seq_length)

# Load model
transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])

# Test output
transformer.eval()
output = transformer(en, de[:, :-1])
output_ids = output.argmax(dim=-1).squeeze(0).tolist()
decoded_text = de_tokenizer.decode(output_ids, skip_special_tokens=True)

# Score
print("Decoded Output:", decoded_text, "\n")
print ("expected output", expected_output)
bleu_score = sentence_bleu(decoded_text, expected_output)

