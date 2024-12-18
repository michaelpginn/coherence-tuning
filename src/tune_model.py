"""Run DPO tuning on a pretrained model"""

import argparse
from typing import cast

import datasets
import transformers
import trl

import wandb
from evaluate_model import evaluate_model


def tune_model(
    output_dir: str,
    dataset: datasets.DatasetDict,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    loss_fn: str,
):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    training_config = trl.DPOConfig(
        label_smoothing=0.5 if loss_fn == "robust" else 0,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        num_train_epochs=200,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir=output_dir,
        report_to="wandb",
        loss_type=loss_fn, # type:ignore
        save_total_limit=3,
    )
    trainer = trl.DPOTrainer(
        model=model,
        args=training_config,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if "eval" in dataset else dataset["test"],
    )
    trainer.train()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        default="checkpoints",
    )
    parser.add_argument(
        "-m",
        "--model_key",
        help="A HuggingFace model key",
        default="openai-community/gpt2",
    )
    parser.add_argument(
        "-l",
        "--loss_fn",
        help="See https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions",
        default="sigmoid",
    )
    parser.add_argument("-d", "--dataset", default="lecslab/story_cloze")
    args = parser.parse_args()

    wandb.init(
        entity="lecs-general",
        project="coherence-tuning",
        config={"model_key": args.model_key, "loss_fn": args.loss_fn, "dataset": args.dataset},
    )

    dataset = datasets.load_dataset(args.dataset)
    dataset = cast(datasets.DatasetDict, dataset)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_key)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_key)

    wandb.log(
        evaluate_model(test_dataset=dataset["test"], model=model, tokenizer=tokenizer)
    )

    trained_model = tune_model(
        output_dir=args.output_dir,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        loss_fn=args.loss_fn,
    )

    # Run another evaluation
    print("Final evaluation:")
    wandb.log(
        evaluate_model(
            test_dataset=dataset["test"], model=trained_model, tokenizer=tokenizer
        )
    )
