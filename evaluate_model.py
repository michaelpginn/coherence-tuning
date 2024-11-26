"""Evaluate a trained LLM using the test set."""

import argparse
import math
import typing
from typing import cast

import datasets
import torch
import transformers
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def evaluate_model(
    test_dataset: datasets.Dataset,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
):
    model = model.to(device)
    # For each example in the test set, compute
    # the average negative log likelihood (aka crossentropy) of the chosen and
    # rejected sentences.
    #
    # We can then compute metrics such as:
    # - Average score (how often does the model assign higher probability to `chosen`)
    # - Mean difference between P(chosen) - P(rejected)

    wins = 0  # How many times does the model prefer the chosen completion?
    margins = []  # Average nll_reject - nll_chosen, wider margin is better

    for row in tqdm(test_dataset, desc="Running inference..."):
        row = cast(typing.Mapping, row)

        # FIXME: This is a bit inefficient, because we recompute logits for
        # the same prefix twice. We should instead input the prefix
        # to the model with `use_cache=True` to get the KV values,
        # then provide these as `past_key_values` for the chosen/rejected inputs
        prefix_inputs = tokenizer(row["prompt"], return_tensors="pt").to(device)
        prefix_len = prefix_inputs["input_ids"].size(-1)

        # Check what the model would generate on its own
        # original_completion = tokenizer.batch_decode(
        #     model.generate(
        #         **prefix_inputs,
        #         generation_config=transformers.GenerationConfig(max_new_tokens=32),
        #     )
        # )

        chosen_inputs = tokenizer(
            row["prompt"] + " " + row["chosen"], return_tensors="pt"
        ).to(device)
        reject_inputs = tokenizer(
            row["prompt"] + " " + row["rejected"], return_tensors="pt"
        ).to(device)

        chosen_logits = model(
            **chosen_inputs
        ).logits  # [batch_size (1), seq_len, vocab_size]
        reject_logits = model(**reject_inputs).logits

        # Slice just the logits corresponding to the chosen/rejected sentence
        chosen_logits = chosen_logits[:, prefix_len - 1 : -1, :]
        reject_logits = reject_logits[:, prefix_len - 1 : -1, :]
        chosen_input_ids = chosen_inputs["input_ids"][:, prefix_len - 1 : -1]
        reject_input_ids = reject_inputs["input_ids"][:, prefix_len - 1 : -1]

        log_softmax = torch.nn.LogSoftmax(dim=-1)
        chosen_probs: torch.Tensor = log_softmax(chosen_logits)
        reject_probs: torch.Tensor = log_softmax(reject_logits)

        # Gather log-probs for actual tokens in chosen/rejected sequences
        chosen_model_probs = torch.gather(
            chosen_probs, -1, chosen_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        reject_model_probs = torch.gather(
            reject_probs, -1, reject_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        nll_chosen = -chosen_model_probs.mean(dim=-1)
        nll_reject = -reject_model_probs.mean(dim=-1)

        # Sanity check
        gold_fn = torch.nn.CrossEntropyLoss(reduction="none")
        correct_loss = gold_fn(chosen_logits.permute(0, 2, 1), chosen_input_ids)
        if not torch.all(torch.isclose(correct_loss, -chosen_model_probs, atol=1e-3)):
            raise Exception(
                f"⛔️ Mismatch!!! Correct: {correct_loss}, predicted: {-chosen_model_probs}"
            )

        wins += torch.sum(nll_chosen < nll_reject).item()
        margins.extend((nll_reject - nll_chosen).tolist())

    mean_margin = math.exp(
        -1 * sum(margins) / len(margins)
    )  # Convert back to probability

    return {
        "average": wins / len(test_dataset),
        "mean_margin": mean_margin,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_key",
        help="A HuggingFace model key",
        default="openai-community/gpt2",
    )
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_key)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_key)

    dataset = datasets.load_dataset("lecslab/story_cloze")
    dataset = cast(datasets.DatasetDict, dataset)

    print(
        evaluate_model(test_dataset=dataset["test"], model=model, tokenizer=tokenizer)
    )
