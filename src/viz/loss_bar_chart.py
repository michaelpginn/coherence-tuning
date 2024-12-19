import matplotlib.pyplot as plt
import numpy as np

models = ("GPT-2", "Llama 3.2 1b")
scores = {
    "Base": (48.3, 42.9),
    "DPO": (62.1, 32.1),
    "cDPO": (55.2, 42.9),
    "Robust": (65.5, 46.4),
}

colors = ["#299D8F", "#E76F51", "#254653", "#43E0D8"]

x = np.arange(len(models))
width = 0.2
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
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 100)

plt.savefig("losses_bar_chart.pdf", bbox_inches='tight')
