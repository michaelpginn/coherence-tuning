"""Evaluate a trained LLM using the test set."""

import argparse
from typing import cast

import datasets
import transformers


def evaluate_model(model_key: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    model = transformers.AutoModel.from_pretrained(model_key)
    dataset = datasets.load_dataset("lecslab/story_cloze")
    dataset = cast(datasets.DatasetDict, dataset)

    # TODO: For each example in the test set, compute
    # the log likelihood of the chosen and rejected sentences.
    #
    # We can then compute metrics such as:
    # - F1 score (how often does the model assign higher probability to `chosen`)
    # - Average difference between P(chosen) - P(rejected)

    for row in dataset["test"]:
        breakpoint()

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_key",
        help="A HuggingFace model key",
        default="openai-community/gpt2",
    )
    args = parser.parse_args()
    evaluate_model(args.model_key)
