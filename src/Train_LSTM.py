import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from Model_Architecture import PoppyPredictor
from Poppy_Dataset import PoppyDataset
from Application_Parameters import TRAINING_EPOCHS


def train_model(until_date='2026-01-10'):
    # 1. Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset Initialization
    # The until_date ensures we don't train on the last 9 days of "silent" records
    WINDOW_SIZE = 160
    dataset = PoppyDataset(
        cgm_file='../data/Poppy CGM.csv',
        events_file='../data/Poppy Reports.csv',
        window_size=WINDOW_SIZE,
        until_date=until_date
    )

    # High batch size for GPU efficiency
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # 3. Model Setup (input_size=2 for Glucose + Metabolic Pressure)
    model = PoppyPredictor(input_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"🚀 Training starting on {len(dataset)} samples...")

    # 4. Training Loop
    # 50 epochs is a good baseline for the refined dataset
    for epoch in range( TRAINING_EPOCHS ):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"✅ Epoch {epoch + 1}/{TRAINING_EPOCHS} | Loss: {avg_loss:.2f}")

    # 5. Save the result
    model_name = "poppy_model_v2.pth"
    torch.save(model.state_dict(), model_name)
    print(f"📂 Training Complete. Model saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Poppy's Metabolic Predictor")
    # Change '2024-01-10' to '2026-01-10', end date of original tranche of data:
    parser.add_argument('--until', type=str, default='2026-01-10',
                        help='Cutoff date for training (YYYY-MM-DD)')

    args = parser.parse_args()
    train_model(until_date=args.until)