# Hyperparameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 301
dropout = 0.1
epochs = 2000
lr = 0.0001
# FIXME - Decreased for debugging purposes
num_sequences = 1 # The number of training sequences. Will be multiplied by `max_sequence_length` to get number of tokens.
vocab_size = 30000 # TODO - use exact embeddings size
en_end_token = 102
de_end_token = 4

# Dataset
train_dataset_path_de = '../../data/data_de_train.txt'
train_dataset_path_en = '../../data/data_en_train.txt'