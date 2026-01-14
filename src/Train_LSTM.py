import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Model_Architecture import PoppyLSTM
from Merge_Poppy_Data import merge_poppy_data
from Process_CGM_Data import processLibreViewCSV
from Process_Report_Data import processLibreReportsCSV
from Check_Input_Tensors import construct_input_tensors
from Check_Input_Tensors import normalize_tensors

import torch.optim as optim


def train_model(X, y, epochs=200, batch_size=32, lr=0.001):
    """
    Trains the PoppyLSTM model using a Learning Rate Scheduler
    and saves the best performing version of the weights.
    """
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Data Preparation
    # Ensure y is 2D [Batch, 1] and everything is a Float on the GPU
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    X_train = X.to(device).float()
    y_train = y.to(device).float()

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Model Initialization
    # input_size=7 matches your features: Glucose, Insulin, 3 Carbs, 2 Time
    model = PoppyLSTM(input_size=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Scheduler & Tracking Logic
    # patience=10: wait 10 epochs before cutting LR in half
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_loss = float('inf')
    model_path = "poppy_model_best.pth"

    print(f"Starting training for {epochs} epochs...")

    # 5. The Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)  # Calculate error
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Step the scheduler based on the average loss
        scheduler.step(avg_loss)

        # Save if this is the best version we've seen so far
        save_msg = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            save_msg = "--> Best Model Saved!"

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}, LR: {current_lr} {save_msg}')

    print(f"\nTraining Complete!")
    print(f"Final Best Loss: {best_loss:.6f}")
    print(f"Final best RMSE re-scaled: {math.sqrt( best_loss ) * ( 400 - 50 ) }")

    # Load the best weights back into the model before returning
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    X, y = construct_input_tensors()
    # 0.1 Normalize the training data so most features lie in the range 0 - 1:
    X_scaled, y_scaled = normalize_tensors(X, y)

    # 0.2 Verify normalization (check if max is ~1.0)
    print(f"Normalization check:  Max X Glucose: {X_scaled[:, :, 0].max()}")
    print(f"Normalization check:  Max y Glucose: {y_scaled.max()}")

    # 1. Grab the first window and move to CPU
    # Format: [window_index, all_time_steps, all_features]
    first_window = X_scaled[0].cpu()

    # 2. Print the first 5 time steps of all 7 features
    print("--- First 5 Time Steps of Window 0 ---")
    print(first_window[:5, :])

    # 3. Specifically check Glucose (Column 0) and Insulin (Column 1)
    glucose_column = first_window[:, 0]
    insulin_column = first_window[:, 1]

    print(f"\nGlucose Range in this window: {glucose_column.min():.4f} to {glucose_column.max():.4f}")
    print(f"Insulin Range in this window: {insulin_column.min():.4f} to {insulin_column.max():.4f}")

    train_model( X_scaled, y_scaled, epochs = 200 )

if __name__ == "__main__":
    main()
