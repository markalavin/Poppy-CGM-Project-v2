import torch
import torch.nn as nn


class PoppyLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.2):
        super(PoppyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The LSTM Layer
        # batch_first=True means our data is shaped [Batch, Time, Features]
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer to prevent overfitting to specific days
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        # This takes the "thought" from the LSTM and turns it into one number (Glucose)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # We only care about the very last hidden state (the summary of the 4 hours)
        out = out[:, -1, :]

        # Apply dropout and the final linear layer
        out = self.dropout(out)
        out = self.fc(out)

        return out