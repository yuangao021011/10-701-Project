from torch.utils.data import Dataset
import torch
import numpy as np
import math
import os
from mido import MidiFile

TRAIN_SAMPLING_RATE = 10  # samples per second
REST_PITCH = 128
NUM_FILES = 100
path = 'maestro-v3.0.0'

class MidiDataset(Dataset):
    def __init__(self, root_dir, stride, window_size, train_sampling_rate=TRAIN_SAMPLING_RATE, n_files=NUM_FILES):
        self.root_dir = root_dir
        self.stride = stride
        self.train_sampling_rate = train_sampling_rate
        self.window_size = window_size
        self.n_files = n_files
        self._build_index()

    def obtain_midi_sampling_frequency(self, mid_file):
        """
        Calculates number of ticks per second for the file.
        """
        ticks_per_beat = mid_file.ticks_per_beat

        # Always read from track 0 (MAESTRO convention)
        tempo = 500000  # default 120 BPM
        for msg in mid_file.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        # seconds per tick = (μs per beat) / (1e6 * ticks per beat)
        tick_duration_sec = tempo / (1_000_000 * ticks_per_beat)

        # ticks per second = 1 / (seconds per tick)
        return 1.0 / tick_duration_sec

    def resolution_from_frequency(self, file_sampling_rate, train_sampling_rate):
        """
        ticks per sample = (ticks/sec) / (samples/sec)
        """
        resolution = file_sampling_rate / train_sampling_rate
        return max(1, round(resolution))

    def _build_index(self):
        midi_files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith((".mid", ".midi")):
                    midi_files.append(os.path.join(dirpath, filename))

        if self.n_files is not None:
            used_files = midi_files[:self.n_files]   # <--
        else:
            used_files = midi_files

        self.sample_list = []
        total_steps = 0

        for f in used_files:
            mid = MidiFile(f)
            file_ticks_per_sec = self.obtain_midi_sampling_frequency(mid)
            resolution = self.resolution_from_frequency(file_ticks_per_sec, self.train_sampling_rate)

            track = mid.tracks[1]

            rows = []
            acc = 0
            cur_note, cur_vel = 0, 0  # start silent

            for msg in track:
                acc += getattr(msg, "time", 0)   # count time for ALL messages
                if msg.type == "note_on":
                    cur_note, cur_vel = msg.note, msg.velocity

                while acc >= resolution:
                    # if silent, emit REST_PITCH; else emit current note
                    note_out = REST_PITCH if cur_vel == 0 else cur_note
                    rows.append([note_out, cur_vel])
                    acc -= resolution

            midi_df = np.asarray(rows, dtype=np.int16)
            steps = (midi_df.shape[0] - self.window_size) / self.stride
            steps = max(0, math.floor(steps) + 1)
            total_steps += steps

            for step in range(steps):
                start = step * self.stride
                end = start + self.window_size
                sample = midi_df[start:end]
                self.sample_list.append(sample)

        print(f"Total samples: {len(self.sample_list)}  (from {len(used_files)} MIDI files)")
        print(f"Total windows: {total_steps}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sample_list[idx]).float()
dataset = MidiDataset(root_dir=path,
                      stride=5,
                      window_size=100,
                      train_sampling_rate=10)

print(len(dataset))
print(dataset[0].shape)