import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configuration ===
# Map your filenames to the legend labels
# Format: (filename, label)
trials = [
    ("RNN_log.csv", "1 Hidden Layer"),
    ("deepGRU_log.csv", "2 Hidden Layers"),
    ("layer3GRU_log.csv", "3 Hidden Layers"),
    ("layer4GRU_log.csv", "4 Hidden Layers")
]

plt.figure(figsize=(10, 6))
SMOOTHING_WINDOW = 3  # Number of epochs for moving average
# === Plotting ===
# Matplotlib automatically cycles through its default colors if we don't specify 'c='
for filename, label in trials:
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        
        if "val_loss" in df.columns and "epoch" in df.columns:
            # Apply sliding window smoothing
            # min_periods=1 ensures we still plot the first few points (optional)
            # If you strictly want 5-epoch averages, remove min_periods=1
            smoothed_loss = df["val_loss"].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            
            plt.plot(df["epoch"], smoothed_loss, label=label, linewidth=2)
        else:
            print(f"Warning: 'val_loss' or 'epoch' column missing in {filename}")
    else:
        # If you don't have the files yet, this prints a warning
        print(f"Warning: File {filename} not found. Skipping.")

# === Styling ===
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs. Number of Hidden Layers")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig("layer_sweep.png", dpi=300)
# plt.savefig("val_loss_comparison.png", dpi=300) # Uncomment to save