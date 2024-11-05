"""Run DPO tuning on a pretrained model"""

import transformers


def tune_model(base_model_key: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    model = transformers.AutoModel.from_pretrained(model_key)

    # TODO: Run DPO training with the base model.
    # DPO should be implemented in HuggingFace already,
    # but the key idea is to compute the logits for
    # the rejected and chosen sentence, given the prefix text.
    #
    # Then, the DPO loss is computed with the two sets of logits,
    # attempting to maximize the difference in probability
    # P(chosen) - P(rejected).
    #
    # During training, we should just report evaluation loss and maybbbbe F1 score.
