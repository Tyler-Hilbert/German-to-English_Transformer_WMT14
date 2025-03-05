
<p align="center">
  <img src="https://github.com/user-attachments/assets/f95d8fec-9793-4d91-a8d0-7caaa607de8d" />
</p>

# Scratch Transformer
Reimplementation of the original 2017 "Attention is all you need" transformer from 2017. For a detailed overview, see [my long blog post](https://www.ubaada.com/post/fc9c5fc3).

The encoder-deocder base model with apprx. 63M was trained on WMT-14 English-to-German dataset. I tried to match the original paper where possible.
## Usage
### Method 1: Using huggingface transformer library (Recommended): 
I have ported the model.py and the last training checkpoint to 🤗 hub. Use this method to automatically download all the files including model weights and tokenizer.json.
``` python
model = AutoModel.from_pretrained("ubaada/original-transformer", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ubaada/original-transformer")
text = 'This is my cat'
output = model.generate(**tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=100))
tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
# # ' Das ist meine Katze.'
```

### Method 2: Manual
- Clone the repo.
- Download pytorch_model.bin from [🤗 repo](https://huggingface.co/ubaada/original-transformer) under ./model_weights/.
- Download tokenizer.json and place it in root.
``` bash
python utils.py generate
# Enter text to translate or press Enter to exit.
# >>> 
```
## Training

| Parameter            | Value                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------|
| Dataset              | WMT14-de-en                                                                                     |
| Translation Pairs    | 4.5M (135M tokens total)                                                                         |
| Epochs               | 24                                                                                              |
| Batch Size           | 16                                                                                              |
| Accumulation Batch   | 8                                                                                               |
| Effective Batch Size | 128 (16 * 8)                                                                                    |
| Training Script      | [train.py](https://github.com/ubaada/scratch-transformer/blob/main/train.py)             |
| Optimiser            | Adam (learning rate = 0.0001)                                                                   |
| Loss Type            | Cross Entropy |
| Final Test Loss      | 1.87 |
| GPU.                 | RTX 4070 (12GB) |

<img src="https://github.com/user-attachments/assets/e533e35b-0236-4856-81d8-7f0b949478f9" width="500"/>


## Results
<img src="https://github.com/user-attachments/assets/6a9e8714-95f5-4c9f-a24a-472a7726feff" width="500" />


## Training Yourself
To train, simply run:
``` bash
python train.py
```
It will:
1. Download the WMT14-de-en dataset from kagglehub.
2. Train a BPE tokenizer and save it to 'wmt14_de_en_tokenizer.json'
3. Load the downloaded dataset.
4. Load the base with base config.
5. Load any checkpoint from previous epoch if found in 'model_weights'
6. Start training the model while logging to wandb.


The defaults model config is hardcoded for base model. The default training params are optimised for RTX 4070 12GB. Change the params according to your GPU before running. On this setup, it achieves approximately 60-80% GPU utilisation. 

The training script is using a simple PyTorch dataloader. It can be improved upon by implementing a bucket loader which batches only similar length inputs. It is not currently implemented.
