import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricStomach(nn.Module):
    """The 'Stomach' that turns Karo impulses into a Sugar Curve."""

    def __init__(self, window_size=160):
        super().__init__()
        self.window_size = window_size
        # 3-Parameter Essence
        self.raw_height = nn.Parameter(torch.tensor([1.0]))
        self.raw_peak = nn.Parameter(torch.tensor([30.0]))
        self.raw_scale = nn.Parameter(torch.tensor([20.0]))

    def forward(self, x):
        # Guardrails
        height = F.softplus(self.raw_height)
        peak = torch.clamp(self.raw_peak, min=10.0, max=60.0)
        scale = F.softplus(self.raw_scale) + 5.0

        # Generate Kernel
        t = torch.arange(float(self.window_size)).to(x.device)
        kernel = height * (t / peak) ** 2 * torch.exp(-t / scale)

        # Convolve
        kernel = kernel.view(1, 1, -1)
        return F.conv1d(x, kernel, padding=self.window_size - 1)[:, :, :self.window_size]


class PoppyPredictor(nn.Module):
    """The 'Brain' that combines CGM and Sugar Curves."""

    def __init__(self, window_size=160):
        super().__init__()
        self.stomach = ParametricStomach(window_size)
        # 2 Inputs: Glucose Channel + Sugar Curve Channel
        self.rnn = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, 2, 160]
        glucose_chan = x[:, 0:1, :]
        karo_impulses = x[:, 1:2, :]

        # Create COB (Carbs on Board) channel
        cob_curve = self.stomach(karo_impulses)

        # Merge and process
        combined = torch.cat([glucose_chan, cob_curve], dim=1).transpose(1, 2)
        lstm_out, _ = self.rnn(combined)
        return self.fc(lstm_out[:, -1, :])