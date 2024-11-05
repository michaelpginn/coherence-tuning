"""Evaluate a trained LLM using the test set."""

import transformers


def evaluate_model(model_key: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    model = transformers.AutoModel.from_pretrained(model_key)

    # TODO: For each example in the test set, compute
    # the log likelihood of the chosen and rejected sentences.
    #
    # We can then compute metrics such as:
    # - F1 score (how often does the model assign higher probability to `chosen`)
    # - Average difference between P(chosen) - P(rejected)

    return {}
