import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class SimpleRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(SimpleRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.GRU(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True
    )
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # Shape: (num_layers, batch_size, hidden_size)
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    # rnn_out shape: (batch_size, window_size, hidden_size)
    rnn_out, _ = self.rnn(x, h0) # We only pass in h0

    # logits shape: (batch_size, window_size, output_size)
    logits = self.fc(rnn_out)

    return logits