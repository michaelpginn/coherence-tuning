"""Run DPO tuning on a pretrained model"""

import transformers
import datasets
import trl
from typing import cast
import argparse
import wandb

from evaluate_model import evaluate_model

def tune_model(
    dataset: datasets.DatasetDict,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    training_config = trl.DPOConfig(
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        num_train_epochs=50,
        output_dir="dpo-model",
        report_to="wandb"
    )
    trainer = trl.DPOTrainer(
        model=model,
        args=training_config,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval']
    )
    trainer.train()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_key",
        help="A HuggingFace model key",
        default="openai-community/gpt2",
    )
    args = parser.parse_args()

    wandb.init(entity="lecs-general", project="coherence-tuning")

    dataset = datasets.load_dataset("lecslab/story_cloze")
    dataset = cast(datasets.DatasetDict, dataset)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_key)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_key)

    trained_model = tune_model(dataset=dataset, model=model, tokenizer=tokenizer)

    # Run another evaluation
    print("Final evaluation:")
    test_metrics = evaluate_model(test_dataset=dataset["test"], model=trained_model, tokenizer=tokenizer)
    wandb.log(test_metrics)
