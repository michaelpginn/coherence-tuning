'''Merges the results of multiple annotators for the Llama3 outputs'''

import datasets
import json

# Maria annotated 1-50 and 100-150
with open("data/generated_texts_gpt2_0.3.maria.json") as f:
    texts_maria = json.load(f)

# Ali annotated 1-50
# Michael annotated 50-100
with open("data/generated_texts_gpt2_0.3.michael.json") as f:
    texts_michael_ali = json.load(f)

if (len(texts_maria) != len(texts_michael_ali)):
    raise ValueError()

merged_data = []
for item1, item2, idx in zip(texts_maria, texts_michael_ali, range(len(texts_maria))):
    item1 = {**item1, "mic_chosen": None, "mar_chosen": None, "ali_chosen": None}
    if idx in range(51) or idx in range(100, 150):
        item1["mar_chosen"] = item1["better_option"]
    if idx in range(50):
        item1["mic_chosen"] = item2["better_option"]
    if idx in range(50, 100):
        item1["ali_chosen"] = item2["better_option"]
    merged_data.append(item1)

data = datasets.Dataset.from_list(merged_data)
# data = data.train_test_split(test_size=0.3)
data = data.remove_columns(["better_option"])
data.push_to_hub("lecslab/porc-llama3_1_1b-v1")

data.to_json("data/porc-llama3_1_1b-merged.json", lines=False)
