# Hyperparameters
d_model = 64
num_heads = 1
num_layers = 2
d_ff = 512
max_seq_length = 160
dropout = 0.1
epochs = 2000
lr = 0.0001
num_sequences = 10 # The number of training sequences. Will be multiplied by `max_sequence_length` to get number of tokens.
vocab_size = 30000 # TODO - use exact embeddings size
en_end_token = 102
de_end_token = 4
