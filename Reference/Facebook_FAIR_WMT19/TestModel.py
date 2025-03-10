# Reference https://huggingface.co/facebook/wmt19-de-en

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from nltk.translate.bleu_score import sentence_bleu

# Dataset
validation_dataset_path_input = '../data/data_en_validation.txt'
validation_dataset_path_expected_outputs = '../data/data_de_validation.txt'

# Load model and tokenizer
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

# Load validation dataset
inputs = []
expected_outputs = []
with open(validation_dataset_path_input, 'r') as file:
    for line in file:
        inputs.append(line.strip())
with open(validation_dataset_path_expected_outputs, 'r') as file:
    for line in file:
        expected_outputs.append(line.strip())

# Run validation data
print (f'Testing dataset at {validation_dataset_path_input}:')
print ('format: [input] [output] [BLEU score]')
print ('----------------------------------------------')
for input, expected_output in zip(inputs, expected_outputs):
        input_ids = tokenizer.encode(input, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bleu_score = sentence_bleu(input.strip(), expected_output.strip()) # TODO - Is there a better way to score?
        
        print (input)
        print (decoded)
        print (bleu_score)
        print ('')