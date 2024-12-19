import argparse
from collections import defaultdict
from math import sqrt
import datasets
from typing import cast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="lecslab/porc-llama3_1_1b-v1-all",
    )
    args = parser.parse_args()
    dataset = cast(datasets.DatasetDict, datasets.load_dataset(args.dataset))

    annotator_stats = defaultdict(lambda: {"count": 0, "matched": 0})
    for row in dataset['train']:
        annots = [row["mic_chosen"], row["mar_chosen"], row["ali_chosen"]] # type:ignore
        annotator_names = frozenset((name for name, a in zip(["mic", "mar", "ali"], annots) if a))
        annots = [a for a in annots if a]
        if len(annots) == 2:
            annotator_stats[annotator_names]["count"] += 1
            if len(set((a for a in annots if a))) == 1:
                annotator_stats[annotator_names]["matched"] += 1
        elif len(annots) > 2:
            raise ValueError(annotator_names)

    for pair, stats in annotator_stats.items():
        print(pair)
        print("Dual annotated: ", stats["count"])
        accuracy = stats["matched"] / stats["count"]
        print("Accuracy: ", accuracy)
        print("Cohen's kappa: ", ((accuracy) - 0.5) / 0.5)
        c = 1-(accuracy)
        p_annot_err = min((2 - sqrt(4 - 8*c)) / 4, (2 + sqrt(4 - 8*c)) / 4)
        est_err = (p_annot_err * p_annot_err) / (p_annot_err * p_annot_err + (1-p_annot_err) * (1-p_annot_err))
        print("Estimated error: ", est_err)


if __name__ == "__main__":
    main()
