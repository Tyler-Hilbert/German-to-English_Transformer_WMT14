# Model
This is a German to English translation model that uses the same architecture and dataset as "Attention is all you Need".  
#### The differences from the paper are that:  
- I translate DE to EN while the paper translates EN to DE.
- I truncate sentences at 128 tokens.
- I trained with 65,535 tokens per batch on 1 H200 instead of 25,000 tokens per batch on 8 P100s.
- I use separate source and target vocabularies, tokenized with BERT, instead of a shared BPE vocabulary.
- I only used the WMT14 DE-EN sentence pairs rather than the entire dataset.

# Usage
#### Step 1 -- Config
Adjust `batch_size` in `Config.py` depending on your hardware.
#### Step 2 -- Train
Train the model using the command:  
`$python3 Train.py`  
(if you get a memory error that means you need to decrease `batch_size` in `Config.py`)
#### Step 3 -- Test Model
Test the trained model on the validation dataset using the command:  
`$python3 TestModel.py`  
(update `model_path` in `Config.py` as the filename changes depending on how long the model is trained)
