"""Convert the raw ROCStories data into a HuggingFace dataset"""

from typing import cast

import datasets

data = datasets.load_dataset(
    "csv", data_files=["./raw/val_2016.csv", "./raw/test_2016.csv"]
)
data = cast(datasets.DatasetDict, data)

# Make splits
data = data["train"].train_test_split(test_size=0.25, seed=0)
eval_test_splits = data["test"].train_test_split(test_size=0.5, seed=0)
data["eval"] = eval_test_splits["train"]
data["test"] = eval_test_splits["test"]


# Format our data with a "prefix" and "chosen"/"rejected" final sentences, for DPO
def process(row):
    row["prompt"] = " ".join(row[f"InputSentence{n}"] for n in range(1, 5))
    if row["AnswerRightEnding"] == 1:
        row["chosen"] = row["RandomFifthSentenceQuiz1"]
        row["rejected"] = row["RandomFifthSentenceQuiz2"]
    else:
        row["chosen"] = row["RandomFifthSentenceQuiz2"]
        row["rejected"] = row["RandomFifthSentenceQuiz1"]
    return row


data = data.map(
    process,
    batched=False,
    remove_columns=[
        "InputStoryid",
        "InputSentence1",
        "InputSentence2",
        "InputSentence3",
        "InputSentence4",
        "RandomFifthSentenceQuiz1",
        "RandomFifthSentenceQuiz2",
        "AnswerRightEnding",
    ],
)

data.push_to_hub("lecslab/story_cloze")
