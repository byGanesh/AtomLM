from pathlib import Path
import matplotlib.pyplot as plt

# Experiment data
epochs = [1, 2, 3]
loss = [2.7389, 2.10, 1.9828]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss,
    marker="o",
    linewidth=2,
    markersize=8
)

# Add values above points
for x, y in zip(epochs, loss):
    plt.text(
        x,
        y + 0.05,
        f"{y:.4f}",
        ha="center",
        fontsize=10
    )

plt.xlabel("Epoch")
plt.ylabel("Average Loss")

plt.title(
    "AtomLM Stage 1 Experiment 002\nTraining Loss Curve",
    fontsize=13
)

plt.xticks(epochs)
plt.grid(True, alpha=0.3)

plt.tight_layout()

path = "assets/atomlm_loss_curve.png"

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(path)
