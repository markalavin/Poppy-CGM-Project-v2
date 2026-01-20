import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysiologyLayer(nn.Module):
    """
    A trainable 1D Convolutional layer that acts as an
    impulse response filter for metabolic events.
    """

    def __init__(self, kernel_size=72):
        super(PhysiologyLayer, self).__init__()
        self.kernel_size = kernel_size

        # We use two independent filters: one for Insulin, one for Meals
        self.insulin_conv = nn.Conv1d(in_channels=1, out_channels=1,
                                      kernel_size=kernel_size, bias=False)
        self.meal_conv = nn.Conv1d(in_channels=1, out_channels=1,
                                   kernel_size=kernel_size, bias=False)

        # Initialize with realistic baselines
        self.initialize_kernels()

    def initialize_kernels(self):
        with torch.no_grad():
            # Create a more significant "V-shape" or downward slope
            # This gives the model a clear signal to start with
            size = self.kernel_size

            # Start with a small negative constant instead of just random noise
            # This ensures the model starts by believing insulin DROPS sugar
            self.insulin_conv.weight.fill_(-0.05)

            # (Keep your meal initialization as is, or set to +0.05)
            self.meal_conv.weight.fill_(0.05)
    def forward(self, x):
        """
        Expects x of shape [batch, seq_len, features]
        Assumes feature index 1 is Insulin and 2 is Meals.
        """
        # 0. THE GUARDRAILS: Apply these BEFORE the convolution math
        with torch.no_grad():
            # Force insulin to stay negative (glucose-lowering)
            self.insulin_conv.weight.clamp_(max=0.0)
            # Force meals to stay positive (glucose-raising)
            self.meal_conv.weight.clamp_(min=0.0)

        # 1. Extract and Pad (Causal Padding)
        ins_raw = x[:, :, 1].unsqueeze(1)
        meal_raw = x[:, :, 2].unsqueeze(1)

        ins_padded = F.pad(ins_raw, (self.kernel_size - 1, 0))
        meal_padded = F.pad(meal_raw, (self.kernel_size - 1, 0))

        # 2. Convolve to create IOB and COB
        iob = self.insulin_conv(ins_padded).squeeze(1)
        cob = self.meal_conv(meal_padded).squeeze(1)

        # 3. Return the processed columns
        return iob, cob


import matplotlib.pyplot as plt


def plot_learned_kernels(model_path):
    # We import these here to avoid 'circular imports' at the top of the file
    from Model_Architecture import PoppyLSTMModel
    from Application_Parameters import HIDDEN_SIZE, NUM_LAYERS

    # 1. Load the model
    # X.shape[2] from your training was 7, so we use 7 here
    model = PoppyLSTMModel(input_size=7, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

    # Load weights (mapping to CPU in case you trained on GPU)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Extract weights from the Physiology Layer
    with torch.no_grad():
        # .detach().cpu() ensures we can convert to numpy regardless of where the model is
        ins_kernel = model.phys_layer.insulin_conv.weight.squeeze().cpu().numpy()
        meal_kernel = model.phys_layer.meal_conv.weight.squeeze().cpu().numpy()

    # 3. Create the Plot
    plt.figure(figsize=(12, 5))

    # Plot Insulin (IOB)
    plt.subplot(1, 2, 1)
    plt.plot(ins_kernel, color='red', lw=2)
    plt.title("Learned Insulin Action Curve\n(Poppy's IOB Shape)")
    plt.xlabel("Time (5-min increments)")
    plt.ylabel("Impact Weight")
    plt.grid(True, alpha=0.3)

    # Plot Meals (COB)
    plt.subplot(1, 2, 2)
    plt.plot(meal_kernel, color='green', lw=2)
    plt.title("Learned Meal Digestion Curve\n(Poppy's COB Shape)")
    plt.xlabel("Time (5-min increments)")
    plt.ylabel("Impact Weight")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# This allows you to run this file directly to see the plot
if __name__ == "__main__":
    plot_learned_kernels("poppy_model_phys_latest.pth")