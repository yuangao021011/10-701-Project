import pandas as pd
import matplotlib.pyplot as plt

# === Set your CSV path ===
csv_path = r"RNN.csv"  # replace with your path

# --- Load data ---
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# Check if columns exist (handles both old and new log formats safely-ish)
# But assuming you ran the new training script, these columns will be there:
train_loss = df["train_loss"]
val_loss = df["val_loss"]

# --- Plot ---
plt.close('all')
fig, ax1 = plt.subplots(figsize=(8,5))

# Plot Train Loss
ax1.plot(epochs, train_loss, color='red', label='Train Loss', linewidth=2)

# Plot Val Loss
ax1.plot(epochs, val_loss, color='blue', linestyle='--', label='Val Loss', linewidth=2)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("2 Hidden Layers RNN")
ax1.grid(True, alpha=0.3)

# Standard legend
ax1.legend(loc='best')

plt.tight_layout()
plt.show()
plt.savefig("RNN_loss_comparison.png", dpi=300)