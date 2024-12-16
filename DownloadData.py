# Script to download the WMT 2014 English-German dataset
#   Source: https://huggingface.co/datasets/wmt/wmt14


from datasets import load_dataset

# Load dataset
ds = load_dataset("wmt/wmt14", "de-en")
# Save dataset to disk
for split, split_dataset in ds.items():
    split_dataset.to_json(f"Data/wmt_de-en_{split}.jsonl")