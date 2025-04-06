# Model
This is a German to English translation model that uses the same architecture and dataset as "Attention is all you Need".  
I implemented the training code, CUDA/Mps support, dataset, tokenizer and evaluation script to a base PyTorch transformer model I found [here](https://medium.com/towards-data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb).  
## Differences
The few minor differences from the paper are that I tokenized with BERT, truncated sentences at 128 sentences and only used the DE-EN data from WMT14.  