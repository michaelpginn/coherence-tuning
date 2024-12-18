import datasets
from typing import cast

for key in ["lecslab/porc-llama3_1_1b-v1", "lecslab/porc-gpt2-v1"]:
    data = cast(datasets.DatasetDict, datasets.load_dataset(key))

    def assign_labels(row):
        """Adds a final label based on the annotations

        For now, if the two annotators disagree, throw out the row.
        Later, we'll improve this.
        """
        annots = [row["mic_chosen"], row["mar_chosen"], row["ali_chosen"]] # type:ignore
        annots = [a for a in annots if a]
        if len(annots) == 1:
            label = annots[0]
        elif len(annots) == 2:
            label = annots[0] if annots[0] == annots[1] else 0
        else:
            raise ValueError()
        texts = [row["generated_text_1"], row["generated_text_2"]]

        chosen = texts[label-1] if label != 0 else None
        rejected = texts[2-label] if label != 0 else None
        return {"chosen": chosen, "rejected": rejected, "prompt": row["story"]}

    data = data.map(assign_labels, batched=False)
    data = data.filter(lambda row: row["chosen"] is not None)
    data = data["train"].train_test_split(test_size=0.3)
    data.push_to_hub(key)
