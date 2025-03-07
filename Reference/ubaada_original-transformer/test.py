from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ubaada/original-transformer", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ubaada/original-transformer")
text = 'This is my cat'
output = model.generate(**tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=100))

print (tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
# # ' Das ist meine Katze.'