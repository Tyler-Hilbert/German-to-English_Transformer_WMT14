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
        self.tgt_token_sentences = self.file_to_sentence_tokens(
            tgt_language_file_path,
            tgt_tokenizer,
            max_seq_length
        )

        self.src_token_sentences = self.file_to_sentence_tokens(
            src_language_file_path,
            src_tokenizer,
            max_seq_length
        )

        # Verify length of sentence pairs
        assert(len(self.src_token_sentences) == len(self.tgt_token_sentences))
    
    def __len__(self):
        return len(self.src_token_sentences)

    def __getitem__(self, idx):
        return self.src_token_sentences[idx], self.tgt_token_sentences[idx]

    # Returns token from a file where each line is a sentence, padded to `max_seq_length`
    def file_to_sentence_tokens(self, file_path, tokenizer, max_seq_length):
        sentence_tokens = []
        with open(file_path, 'r') as file:
            for line in file:
                tokens = tokenizer(
                    line.strip(),
                    truncation=True,
                    padding='max_length',
                    max_length=max_seq_length
                ).input_ids
                sentence_tokens.append(tokens)
        return torch.tensor(sentence_tokens)