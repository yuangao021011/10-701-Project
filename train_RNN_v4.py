from torch.utils.data import DataLoader, random_split
from simpleRNN import SimpleRNN
from MidiDataset_v3 import MidiDataset
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import csv


path = 'maestro-v3.0.0'
TRAIN_SAMPLING_RATE = 10      # Hz
STRIDE = 10                   # 1 second step at 10 Hz
WINDOW_SIZE = TRAIN_SAMPLING_RATE * 10   # 10 seconds → 100 samples

NUM_FILES = 100

BATCH_SIZE = 32               
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4           # mild regularization

# ====== MODEL SETTINGS ======
INPUT_SIZE = 2                # we feed both velocity and pitch
HIDDEN_SIZE = 512
NUM_LAYERS = 1
OUTPUT_SIZE = 64             # pitch classes [0..127]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is", device)


dataset = MidiDataset(
    root_dir=path,
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

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False, # No shuffle for validation
    drop_last=False,
    num_workers=2,
    pin_memory=True
)


model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


log_path = "RNN_log.csv"
os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

ckpt_dir = "checkpoints_v4"
os.makedirs(ckpt_dir, exist_ok=True)

# write header once
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, running_correct, total_tokens = 0.0, 0, 0

    for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        batch = batch.to(device)                     # (B,T,2)

        pitches = batch[:, :, 0]               # (B, T)
        vels    = batch[:, :, 1] / 127.0       # (B, T)
        
        prev_pitches = pitches[:, :-1]          # (B, T-1)
        curr_vels    = vels[:, 1:]     

        inputs  = torch.stack([prev_pitches, curr_vels], dim=-1)
        labels  = pitches[:, 1:].long()

        optimizer.zero_grad()
        logits = model(inputs)                       # (B,T,128)

        # Predict through timestep
        B, T_eff, C = logits.shape                  # T_eff = T-1
        loss = criterion(
            logits.reshape(B * T_eff, C),
            labels.reshape(B * T_eff)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * B * T_eff
        preds = logits.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        total_tokens += B * T_eff

    train_loss = running_loss / total_tokens
    train_acc = running_correct / total_tokens

    # Validation
    model.eval()
    val_running_loss, val_running_correct, val_total_tokens = 0.0, 0, 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            pitches = batch[:, :, 0]
            vels    = batch[:, :, 1] / 127.0
            
            prev_pitches = pitches[:, :-1]
            curr_vels    = vels[:, 1:]

            inputs  = torch.stack([prev_pitches, curr_vels], dim=-1)
            labels  = pitches[:, 1:].long()

            logits = model(inputs)
            
            B, T_eff, C = logits.shape
            loss = criterion(
                logits.reshape(B * T_eff, C),
                labels.reshape(B * T_eff)
            )

            val_running_loss += loss.item() * B * T_eff
            preds = logits.argmax(dim=-1)
            val_running_correct += (preds == labels).sum().item()
            val_total_tokens += B * T_eff

    val_loss = val_running_loss / val_total_tokens
    val_acc = val_running_correct / val_total_tokens

    # === LOGGING ===
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc]) 

    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": train_loss,
        "acc": train_acc,
        "val_loss": val_loss, # Useful to save this too
        "val_acc": val_acc
    }, ckpt_path)

    print(f"Saved checkpoint to {ckpt_path}")