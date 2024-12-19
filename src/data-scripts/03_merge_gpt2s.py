'''Merges the results of multiple annotators for the GPT2 outputs'''
import datasets
import json

# Maria annotated 1-50 and 100-150
with open("data/generated_texts_gpt2_0.3.maria.json") as f:
    texts_maria = json.load(f)

# Michael annotated 1-50
# Ali annotated 50-100
with open("data/generated_texts_gpt2_0.3.json") as f:
    texts_michael_ali = json.load(f)

if (len(texts_maria) != len(texts_michael_ali)):
    raise ValueError()

merged_data = []
for item1, item2, idx in zip(texts_maria, texts_michael_ali, range(len(texts_maria))):
    item1 = {**item1, "mic_chosen": None, "mar_chosen": None, "ali_chosen": None}
    if idx in range(50) or idx in range(100, 150):
        item1["mar_chosen"] = item1["better_option"]
    if idx in range(50):
        item1["ali_chosen"] = item2["better_option"]
    if idx in range(50, 100):
        item1["mic_chosen"] = item2["better_option"]
    merged_data.append(item1)

data = datasets.Dataset.from_list(merged_data)
# data = data.train_test_split(test_size=0.3)
data = data.remove_columns(["better_option"])
data.push_to_hub("lecslab/porc-gpt2-v1-all")

# data.to_json("data/porc-gpt2-merged.json", lines=False)
