# Model
This is a German to English translation model that uses the same architecture and dataset as "Attention is all you Need".  
I implemented the training code, CUDA/Mps support, dataset, tokenizer and evaluation script to a base PyTorch transformer model I found [here](https://medium.com/towards-data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb).  
#### Differences
The differences from the paper are that:  
- I translated DE to EN while the paper translates EN to DE.
- I truncated sentences at 128 tokens.
- I only used the WMT14 DE-EN sentence pairs rather than the entire dataset.
- I use separate source and target vocabularies, tokenized with BERT, instead of a shared vocabulary.