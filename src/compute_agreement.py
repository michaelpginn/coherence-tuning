import argparse
import datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="lecslab/porc-llama3_1_1b-v1",
    )
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.dataset)

    count = 0
    matched = 0
    for row in dataset['train']:
        annots = [row["mic_chosen"], row["mar_chosen"], row["ali_chosen"]]
        annots = [a for a in annots if a]
        if len(annots) == 2:
            count += 1
            if annots[0] == annots[1]:
                matched += 1
        elif len(annots) > 2:
            raise ValueError()

    print("Dual annotated: ", count)
    print("Accuracy: ", matched / count)
    print("Cohen's kappa: ", ((matched / count) - 0.5) / 0.5)


if __name__ == "__main__":
    main()
