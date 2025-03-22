import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Stores a German English sentence pair
class DE_EN_Dataset(Dataset):
    def __init__(self, de_file_path, en_file_path, de_tokenizer_name, en_tokenizer_name, max_seq_length):
        # Tokenizers
        en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_name)
        de_tokenizer = AutoTokenizer.from_pretrained(de_tokenizer_name)

        # Load data and tokenize
        self.en_token_sentences = []
        with open(en_file_path, 'r') as file:
            for line in file:
                tokens = en_tokenizer(line.strip()).input_ids
                tokens += [0] * (max_seq_length-len(tokens))
                self.en_token_sentences.append(tokens)

        self.de_token_sentences = []
        with open(de_file_path, 'r') as file:
            for line in file:
                tokens = de_tokenizer(line.strip()).input_ids
                tokens += [0] * (max_seq_length-len(tokens))
                self.de_token_sentences.append(tokens)

        # Verify length of sentence pairs
        assert(len(self.en_token_sentences) == len(self.de_token_sentences))
        self.len = len(self.en_token_sentences)
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.de_token_sentences[idx], self.en_token_sentences[idx]