from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import random
from tqdm import tqdm
import json



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

ds = load_dataset("lecslab/story_cloze")
tokenizer.pad_token = tokenizer.eos_token


def generate_text(sentence, temp = 1):
    inputs = tokenizer(sentence,  padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample = True ,max_length=2*(inputs['input_ids'].size()[1]), temperature = temp, repetition_penalty=2.0)  # Adjust max_length as needed
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




with_generated_texts = []


# Process the dataset
random_examples = random.sample(range(len(ds['train'])), 150)

for i in tqdm(random_examples):
    example = ds['train'][i]
    story = example['prompt']
    generated_text_1 = generate_text(story, temp = 0.3)[len(story):]
    generated_text_2 = generate_text(story, temp = 0.3)[len(story):]
    instance = {'story': story, 'generated_text_1': generated_text_1, 'generated_text_2': generated_text_2, 'better_option': 0}
    with_generated_texts.append(instance)




with open('generated_texts.json', 'w') as f:
    json.dump(with_generated_texts, f, indent=4)