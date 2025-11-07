import pandas as pd
import matplotlib.pyplot as plt

# === Set your CSV path ===
csv_path = r"RNN_log.csv"  # replace with your path

# --- Load data ---
df = pd.read_csv(csv_path)
epochs = df["epoch"]
loss = df["loss"]
acc = df["acc"] * 100  # convert to percentage

# --- Plot ---
plt.close('all')
fig, ax1 = plt.subplots(figsize=(8,5))
ax2 = ax1.twinx()

ax1.plot(epochs, loss, color='red', label='Loss', linewidth=2)
ax2.plot(epochs, acc, color='blue', linestyle='--', label='Accuracy (%)', linewidth=2)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color='red')
ax2.set_ylabel("Accuracy (%)", color='blue')
ax1.set_title("RNN")
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.show()
plt.savefig("RNN_loss.png", dpi=300)