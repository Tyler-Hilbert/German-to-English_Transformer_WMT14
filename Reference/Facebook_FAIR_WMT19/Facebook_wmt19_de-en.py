# Reference https://huggingface.co/facebook/wmt19-de-en

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = "Maschinelles Lernen ist großartig, oder?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) # Machine learning is great, isn't it?