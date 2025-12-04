import torch
from simpleRNN import SimpleRNN
from MidiDataset_v3 import MidiDataset
from torch.utils.data import DataLoader, random_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches

#Hyper
INPUT_SIZE = 2
HIDDEN_SIZE = 512
NUM_LAYERS = 2
OUTPUT_SIZE = 64
BATCH_SIZE = 32   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

# Load checkpoint 
ckpt_path = "checkpoints_v5/epoch_080.pt"
ckpt = torch.load(ckpt_path, map_location=device)

model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint from {ckpt_path}, acc={ckpt['acc']:.3f}")

DATA_PATH = 'maestro-v3.0.0'
TRAIN_SAMPLING_RATE = 10      # Hz
STRIDE = 10                   # 1 second step at 10 Hz
WINDOW_SIZE = TRAIN_SAMPLING_RATE * 10   # 10 seconds → 100 samples
NUM_FILES = 100

dataset = MidiDataset(
    root_dir=DATA_PATH,
    stride=STRIDE,
    window_size=WINDOW_SIZE,
    train_sampling_rate=TRAIN_SAMPLING_RATE,
    n_files=NUM_FILES
)

total_len = len(dataset)
test_len = int(total_len * 0.10)
rest_len = total_len - test_len
train_len = int(rest_len * 0.80)
val_len = rest_len - train_len

# split 
train_ds, val_ds, test_ds = random_split(
    dataset, 
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False, # No shuffle for validation
    drop_last=False,
    num_workers=2,
    pin_memory=True
)



sample = next(iter(test_loader))
print(sample.size())


MIN_PITCH = 34
MAX_PITCH = 96
N_PITCHES = MAX_PITCH - MIN_PITCH + 1  # 63 pitches 0...62
REST_INDEX = N_PITCHES                 # 63 rest pitch

def label_to_midi(idx):
    if idx == REST_INDEX:
        return None   
    return idx + MIN_PITCH

# predicted_pitches = None
# gt_pitches = None
with torch.no_grad():
    pitches = sample[5, :, 0]
    vels = sample[5, :, 1]         # (T,)

    # input_t = [pitch_{t-1}, vel_t],   
    # label_t = pitch_t
    prev_pitches = pitches[:-1]                # (T-1,)
    curr_vels    = vels[1:] / 127.0           # (T-1,)

    # (1, T-1, 2) 
    inputs = torch.stack([prev_pitches, curr_vels], dim=-1)  # (T-1, 2)
    inputs = inputs.unsqueeze(0).to(device)                  # (1, T-1, 2)

    logits = model(inputs)               # (1, T-1, 129)
    preds  = logits.argmax(dim=-1)[0]    # (T-1,)

    predicted_pitches = preds.cpu().tolist()
    MIDI_pred = [label_to_midi(p) for p in predicted_pitches]
    print("predicted:",predicted_pitches)

    gt_pitches = pitches[1:].cpu().tolist()
    MIDI_gt = [label_to_midi(p) for p in gt_pitches]
    print("ground truth:",gt_pitches)


def plot_piano_roll_comparison(gt, pred):
    """
    gt: list of ground truth MIDI pitches
    pred: list of predicted MIDI pitches
    """
    t_steps = range(len(gt))
    
    # Create a figure with two subplots (Piano Roll on top, Error Strip on bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                   gridspec_kw={'height_ratios': [4, 1]}, 
                                   sharex=True)
    
    # --- 1. PIANO ROLL OVERLAY ---
    # Plot Ground Truth as Blue circles
    ax1.scatter(t_steps, gt, c='blue', label='Ground Truth', alpha=0.6, s=80, edgecolors='none')
    
    # Plot Prediction as Red X's (or smaller dots)
    ax1.scatter(t_steps, pred, c='red', label='Prediction', alpha=0.7, s=50, marker='x')

    # Draw connecting lines for ground truth to visualize the "melody"
    ax1.plot(t_steps, gt, c='blue', alpha=0.2) 
    
    ax1.set_ylabel('MIDI Pitch')
    ax1.set_title(f'Ground Truth vs Prediction (acc: {ckpt["acc"]:.3f})')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 2. MATCH/MISMATCH STRIP (BARCODE) ---
    # Create a color map: Green for match (0), Red for error (1)
    matches = [1 if g == p else 0 for g, p in zip(gt, pred)] # 1=Match, 0=Error for visualization
    
    # We plot this as a simple "barcode"
    # We create a matrix of shape (1, T) to use imshow
    match_matrix = np.array(matches).reshape(1, -1)
    
    # Plot heatmap: Green (match) vs Red (mismatch)
    # We use a custom colormap: 0->Red, 1->Green
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#ff9999', '#99ff99']) # Light Red, Light Green
    
    ax2.imshow(match_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Add text labels inside the strip
    ax2.set_yticks([])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Time Step')
    
    # Add a custom legend for the strip
    red_patch = mpatches.Patch(color='#ff9999', label='Mismatch')
    green_patch = mpatches.Patch(color='#99ff99', label='Match')
    ax2.legend(handles=[green_patch, red_patch], loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.savefig('piano_roll_comparison.png', dpi=300)

# Run the plotter
# Filter out None values if your label_to_midi returns None for rests
# or handle them by assigning a specific value (e.g., 0) for plotting
clean_gt = [x if x is not None else 0 for x in MIDI_gt]
clean_pred = [x if x is not None else 0 for x in MIDI_pred]

plot_piano_roll_comparison(clean_gt, clean_pred)


print("\nEvaluating on the full Test Set...")

criterion = torch.nn.CrossEntropyLoss()
running_loss = 0.0
running_correct = 0
total_tokens = 0

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        # Extract data
        pitches = batch[:, :, 0]
        vels    = batch[:, :, 1] / 127.0
        
        # Prepare inputs (t-1) and targets (t)
        prev_pitches = pitches[:, :-1]
        curr_vels    = vels[:, 1:]

        inputs = torch.stack([prev_pitches, curr_vels], dim=-1)
        labels = pitches[:, 1:].long()

        # Forward pass
        logits = model(inputs)  # (B, T-1, C)
        
        # Reshape for loss calculation
        B, T_eff, C = logits.shape
        loss = criterion(logits.reshape(B * T_eff, C), labels.reshape(B * T_eff))

        # Accumulate stats
        running_loss += loss.item() * (B * T_eff)
        preds = logits.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        total_tokens += (B * T_eff)

test_loss = running_loss / total_tokens
test_acc = running_correct / total_tokens

print(f"Test Results -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

