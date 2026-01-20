import torch
import torch.nn as nn


class PoppyPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        """
        input_size=2 because we are feeding it:
        1. Glucose values
        2. Pre-calculated Metabolic Pressure (Karo/Meals)
        """
        super(PoppyPredictor, self).__init__()

        # The LSTM processes the sequence
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # The fully connected layer turns the LSTM output into a single glucose prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x comes in as [batch, 2, 160] from the Dataset
        # LSTM expects [batch, sequence_length, features], so we transpose
        x = x.transpose(1, 2)

        # Pass through LSTM
        # out shape: [batch, 160, 64]
        out, _ = self.lstm(x)

        # We only care about the very last time step (the prediction point)
        last_time_step = out[:, -1, :]

        # Final prediction
        prediction = self.fc(last_time_step)
        return prediction