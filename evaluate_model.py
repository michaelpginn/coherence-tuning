"""Evaluate a trained LLM using the test set."""

import argparse
import typing
from typing import cast

import datasets
import torch
import transformers


def evaluate_model(model_key: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_key)
    dataset = datasets.load_dataset("lecslab/story_cloze")
    dataset = cast(datasets.DatasetDict, dataset)

    # TODO: For each example in the test set, compute
    # the average log likelihood of the chosen and rejected sentences.
    #
    # We can then compute metrics such as:
    # - F1 score (how often does the model assign higher probability to `chosen`)
    # - Average difference between P(chosen) - P(rejected)

    for row in dataset["test"]:
        row = cast(typing.Mapping, row)

        # FIXME: This is a bit inefficient, because we recompute logits for
        # the same prefix twice. We should instead input the prefix
        # to the model with `use_cache=True` to get the KV values,
        # then provide these as `past_key_values` for the chosen/rejected inputs
        prefix_len = tokenizer(row["prefix"], return_tensors="pt")["input_ids"].size(-1)

        chosen_inputs = tokenizer(row["prefix"] + row["chosen"], return_tensors="pt")
        reject_inputs = tokenizer(row["prefix"] + row["rejected"], return_tensors="pt")

        chosen_logits = model(**chosen_inputs).logits
        reject_logits = model(**reject_inputs).logits

        # Slice just the logits corresponding to the chosen/rejected sentence
        chosen_logits = chosen_logits[:, prefix_len - 1 : -1, :]
        reject_logits = reject_logits[:, prefix_len - 1 : -1, :]

        log_softmax = torch.nn.LogSoftmax(dim=-1)
        chosen_probs, reject_probs = (
            log_softmax(chosen_logits),
            log_softmax(reject_logits),
        )

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
