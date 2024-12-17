import datasets
import json

with open("data/generated_texts_gpt2_0.3.json") as f:
    texts = json.load(f)

data = datasets.Dataset.from_list(texts)

def relabel(row, idx):
    if row["better_option"] == 1:
        stories = (row["generated_text_1"], row["generated_text_2"])
    elif row["better_option"] == 2:
        stories = (row["generated_text_2"], row["generated_text_1"])
    else:
        raise Exception("Invalid option for `better_option`", row["better_option"])

    base_row = {
        "prompt": row["story"],
        "chosen": stories[0], "rejected":stories[1],
        "ali_chosen": None, "mic_chosen": None, "mar_chosen": None
    }
    if idx in range(51):
        return {**base_row, "ali_chosen": row["better_option"]}
    elif idx in range(51, 100):
        return {**base_row, "mic_chosen": row["better_option"]}
    else:
        return {**base_row, "mar_chosen": row["better_option"]}

data = data.map(relabel, with_indices=True)
data = data.train_test_split(test_size=0.3)
data = data.remove_columns(["better_option", "story"])
data.push_to_hub("lecslab/porc-gpt2-v1")
