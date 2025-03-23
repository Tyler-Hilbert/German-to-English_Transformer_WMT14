import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Stores a sentence in two languages
class SentencePairDataset(Dataset):
    def __init__(self, src_language_file_path, tgt_language_file_path, src_tokenizer_name, tgt_tokenizer_name, max_seq_length):
        # Tokenizers
        tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_tokenizer_name)
        src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)

        # Load data and tokenize
        self.tgt_token_sentences = []
        with open(tgt_language_file_path, 'r') as file:
            for line in file:
                tokens = tgt_tokenizer(line.strip()).input_ids
                tokens += [0] * (max_seq_length-len(tokens))
                self.tgt_token_sentences.append(tokens)

        self.src_token_sentences = []
        with open(src_language_file_path, 'r') as file:
            for line in file:
                tokens = src_tokenizer(line.strip()).input_ids
                tokens += [0] * (max_seq_length-len(tokens))
                self.src_token_sentences.append(tokens)

        # Verify length of sentence pairs
        assert(len(self.src_token_sentences) == len(self.tgt_token_sentences))
    
    def __len__(self):
        return len(self.src_token_sentences)

    def __getitem__(self, idx):
        return self.src_token_sentences[idx], self.tgt_token_sentences[idx]