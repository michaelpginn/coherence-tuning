"""Run DPO tuning on a pretrained model"""

import argparse
from typing import cast

import datasets
import transformers
import trl
import peft

import wandb
from evaluate_model import evaluate_model


def tune_model(
    output_dir: str,
    dataset: datasets.DatasetDict,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    loss_fn: str,
    use_lora: bool,
    label_smoothing_p: float = 0,
):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    training_config = trl.DPOConfig(
        label_smoothing=label_smoothing_p,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        num_train_epochs=200 if not use_lora else 400,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir=output_dir,
        report_to="wandb",
        loss_type=loss_fn, # type:ignore
        save_total_limit=3,
        learning_rate=1e-6 if not use_lora else 2e-4
    )

    if use_lora:
        lora_config = peft.LoraConfig(r=8) # type:ignore
        model = peft.get_peft_model(model, lora_config) # type:ignore


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
    parser.add_argument(
        "--use_lora",
        action='store_true',
    )
    parser.add_argument("-d", "--dataset", default="lecslab/story_cloze")
    parser.add_argument("--label_smoothing", default=0, type=float)
    args = parser.parse_args()

    wandb.init(
        entity="lecs-general",
        project="coherence-tuning",
        config=vars(args),
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
        use_lora=args.use_lora,
        label_smoothing_p=float(args.label_smoothing)
    )

    # Run another evaluation
    print("Final evaluation:")
    eval_result = evaluate_model(
        test_dataset=dataset["test"], model=trained_model, tokenizer=tokenizer
    )
    preds = eval_result["test/preds"]
    del eval_result["test/preds"]
    wandb.log(
        eval_result
    )
    data = [dataset["test"]["story"], dataset["test"]["chosen"], dataset["test"]["rejected"], preds]
    preds_table = wandb.Table(
        columns=["story", "chosen", "rejected", "correct?"],
        data=[list(i) for i in zip(*data)]
    )
    wandb.log({"predictions": preds_table})
