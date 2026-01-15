import torch.nn as nn
# Import only from the parameters file
from Application_Parameters import PREDICTION_SAMPLES, HIDDEN_SIZE, NUM_LAYERS


class PoppyLSTMModel(nn.Module):
    def __init__(self, input_size=7):
        super(PoppyLSTMModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)

        # Output layer (24 samples)
        self.fc = nn.Linear(HIDDEN_SIZE, PREDICTION_SAMPLES)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out