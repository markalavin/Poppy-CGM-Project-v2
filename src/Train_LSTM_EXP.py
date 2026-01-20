import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Model_Architecture import PoppyPredictor
from Poppy_Dataset import PoppyDataset


def plot_metabolic_curve(model, window_size=160):
    """Generates a plot of the learned Karo response curve."""
    model.eval()
    with torch.no_grad():
        p = model.stomach
        height = torch.nn.functional.softplus(p.raw_height).cpu()
        peak = torch.clamp(p.raw_peak, 10, 60).cpu()
        scale = (torch.nn.functional.softplus(p.raw_scale) + 5.0).cpu()

        t = torch.arange(float(window_size))
        curve = height * (t / peak) ** 2 * torch.exp(-t / scale)

        plt.figure(figsize=(10, 5))
        plt.plot(t.numpy(), curve.numpy(), label='Learned Karo Response', color='orange', lw=3)
        plt.fill_between(t.numpy(), curve.numpy(), color='orange', alpha=0.2)
        plt.title(f"Poppy's Learned Metabolic Essence\nPeak: {peak.item():.1f}m | Magnitude: {height.item():.2f}")
        plt.xlabel("Minutes after Dose")
        plt.ylabel("Glucose Pressure (mg/dL)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()


def train_model(use_cuda=True):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Settings
    WINDOW_SIZE = 160
    dataset = PoppyDataset(cgm_file='../data/Poppy CGM.csv', events_file='../data/Poppy Reports.csv')

    # Speed Demon: Large batch size to keep the GPU busy
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

    model = PoppyPredictor(window_size=WINDOW_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"🚀 Training starting on {len(dataset)} samples...")

    for epoch in range(100):
        model.train()
        epoch_loss = 0

        for i, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print a dot for every batch to show progress
            print(".", end="", flush=True)

        p = model.stomach
        peak = torch.clamp(p.raw_peak, 10, 60).item()
        avg_loss = epoch_loss / len(loader)
        print(f"\n✅ Epoch {epoch + 1}/100 | Loss: {avg_loss:.2f} | Peak: {peak:.1f}m")

    # Final Visualization & Save
    plot_metabolic_curve(model, WINDOW_SIZE)
    torch.save(model.state_dict(), "poppy_model_cob.pth")
    print("📂 Model saved to poppy_model_cob.pth")


# --- THE IGNITION ---
if __name__ == "__main__":
    train_model(use_cuda=True)