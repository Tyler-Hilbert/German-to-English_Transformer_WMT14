config = {
    # Path to save model
    'model_path': 'models/wmt14_de-en_model1-1',

    # Hyperparameters
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 6,
    'd_ff': 2048,

    'max_seq_length': 128,

    'epochs': 5,
    'batch_size': 64,
    'lr': 0.0001,
    'dropout': 0.1,

    # Tokenizers
    'en_tokenizer_name': 'bert-base-uncased',
    'de_tokenizer_name': 'bert-base-german-cased',
    'en_end_token': 102,
    'de_end_token': 4,

    # Validation Dataset
    'validation_dataset_path_de': 'data/data_de_validation.txt',
    'validation_dataset_path_en': 'data/data_en_validation.txt'
}

from types import SimpleNamespace
config = SimpleNamespace(**config)