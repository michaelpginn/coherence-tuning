from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import random
from tqdm import tqdm
import json
from datasets import Dataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir="/scratch/alpine/sema4648/transformer_cache/")

ds = load_dataset("lecslab/story_cloze")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate_text(sentence, temp: float = 1):
    inputs = tokenizer(sentence,  padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample = True ,max_length=2*(inputs['input_ids'].size()[1]), temperature = temp, repetition_penalty=2.0)  # Adjust max_length as needed
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




with_generated_texts = []


# Process the dataset

loading_from_ds = False ## if choosing random instances for a model. If working with previous examples, should load teh file instead
previous_file_path = 'generated_texts_gpt2_0.3.json' ## set only if loading_from_ds is False


if loading_from_ds:
    random_indices = random.sample(range(len(ds['train'])), 150)
    random_examples = [{"story": ds['train'][i]['prompt']} for i in random_indices]
else:
    with open(previous_file_path, 'r') as f:
        random_examples = json.load(f)


for example in tqdm(random_examples):
    story = example['story']
    generated_text_1 = generate_text(story, temp = 0.3)[len(story):]
    generated_text_2 = generate_text(story, temp = 0.3)[len(story):]
    instance = {'story': story, 'generated_text_1': generated_text_1, 'generated_text_2': generated_text_2, 'better_option': 0}
    with_generated_texts.append(instance)



with open('generated_texts_llama3.2_0.3.json', 'w') as f:
    json.dump(with_generated_texts, f, indent=4)



dataset = Dataset.from_list(with_generated_texts)


dataset.push_to_hub("lecslab/generated_cloze_llama3.2_0.3", private=False)
