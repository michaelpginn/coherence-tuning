import matplotlib.pyplot as plt
import numpy as np

models = ("GPT-2", "Llama 3.2 1b", "Qwen 2.5 0.5b")
scores = {
    "Before DPO": (46.7, 44.2, 46.6),
    "After DPO": (65.6, 92.5, 83.5)
}

colors = ["#299D8F", "#E76F51"]

x = np.arange(len(models))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')

for idx, (attribute, measurement) in enumerate(scores.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[idx])
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xticks(x + width * (len(scores) - 1) / 2, models)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 100)

plt.savefig("story_cloze_bar_chart.pdf", bbox_inches='tight')
