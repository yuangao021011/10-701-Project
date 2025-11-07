from torch.utils.data import DataLoader
from simpleRNN import SimpleRNN
from MidiDataset import MidiDataset
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

BATCH_SIZE = 32               # use 32 if VRAM allows; else 8/16
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4           # mild regularization

# ====== MODEL SETTINGS ======
INPUT_SIZE = 1                # we feed only velocity
HIDDEN_SIZE = 512
NUM_LAYERS = 1
OUTPUT_SIZE = 129             # pitch classes [0..127]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is", device)

dataset = MidiDataset(
    root_dir=path,
    stride=STRIDE,
    window_size=WINDOW_SIZE,
    train_sampling_rate=TRAIN_SAMPLING_RATE,
    n_files=NUM_FILES
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True
)


model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


log_path = "RNN_log.csv"
os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

# write header once
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "acc"])


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, running_correct, total_tokens = 0.0, 0, 0

    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)                     # (B,T,2)
        labels = batch[:, :, 0].long()               # (B,T) pitches
        inputs = (batch[:, :, 1] / 127.0).unsqueeze(-1)  # (B,T,1) velocities

        optimizer.zero_grad()
        logits = model(inputs)                       # (B,T,128)

        # Flatten to compute CE across all timesteps
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B*T, C), labels.reshape(B*T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * B * T
        preds = logits.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        total_tokens += B * T

    avg_loss = running_loss / total_tokens
    acc = running_correct / total_tokens
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  loss: {avg_loss:.4f}  acc: {acc:.3f}")
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_loss, acc])