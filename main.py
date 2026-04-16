# main.py
import argparse
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import UCIHARDataLoader
from src.partition import UserPartitioner
from src.centralized_data import CentralizedDataBuilder
from src.model import HAR_CNN
from src.model_utils import reshape_for_cnn
from src.client import FLClient
from src.server import FederatedServer


def train(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluates the model on the provided dataloader."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')

    return acc, f1, all_preds, all_labels


def run_centralized_training(client_splits, partitioner, device):
    """Runs the centralized training pipeline."""
    print("\n--- Starting Centralized Training ---")
    
    # CENTRALIZED DATA
    central_builder = CentralizedDataBuilder(client_splits)
    global_data = central_builder.combine_all_clients(
        add_client_id=False,
        shuffle=True
    )

    # GLOBAL TEST SET
    X_global_test_raw, y_global_test = partitioner.get_global_test()

    # NORMALIZATION (GLOBAL)
    X_train_raw = global_data["X_train"]
    global_mean = X_train_raw.mean(axis=0)
    global_std  = X_train_raw.std(axis=0).replace(0, 1)

    X_train_norm = (X_train_raw       - global_mean) / global_std
    X_test_norm  = (X_global_test_raw - global_mean) / global_std

    # RESHAPE FOR CNN
    X_train = reshape_for_cnn(X_train_norm)
    X_test  = reshape_for_cnn(X_test_norm)

    y_train = torch.tensor(global_data["y_train"].values.squeeze() - 1, dtype=torch.long)
    y_test  = torch.tensor(y_global_test.values.squeeze() - 1, dtype=torch.long)

    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")

    # DATALOADER
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64, shuffle=False)

    # MODEL
    model = HAR_CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # METRIC TRACKING
    history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "f1_score": []
    }

    # TRAINING LOOP
    epochs = 10
    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, f1, _, _ = evaluate(model, test_loader, device)

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["f1_score"].append(f1)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # SAVE RESULTS
    os.makedirs("results", exist_ok=True)

    # Save metrics
    df = pd.DataFrame(history)
    df.to_csv("results/centralized_metrics.csv", index=False)

    # Save model
    torch.save(model.state_dict(), "results/centralized_model.pth")

    print("\nMetrics saved to results/centralized_metrics.csv")
    print("Model saved to results/centralized_model.pth")
    
    return model


def setup_federated_clients(client_splits, device):
    """Initializes Federated Learning clients & global model."""
    print("\n--- Setting up Federated Learning Clients ---")
    
    # Initialize global model (server side)
    global_model = HAR_CNN().to(device)
    global_weights = global_model.state_dict()

    clients = {}

    # Create clients
    for client_id in client_splits.keys():
        client = FLClient(client_id, device)

        # Receive global model
        client.receive_global_model(global_weights)

        client.print_status()

        clients[client_id] = client
        
    return clients, global_model

def run_federated_round(clients, client_splits):
    """
    Executes ONE federated learning round:
    - Each client trains locally
    - Returns updated weights
    - Collects all client updates for server
    """

    print("\n--- Federated Round 1: Client Training & Weight Return ---")

    client_updates = []

    for client_id, client in clients.items():
        data = client_splits[client_id]

        print(f"\n[Client {client_id}] Starting local training...")

        # LOCAL TRAINING
        updated_weights = client.local_train(
            data["X_train"],
            data["y_train"],
            epochs=2
        )

        # RETURN MODEL WEIGHTS
        client_updates.append({
            "client_id": client_id,
            "weights": updated_weights,          # model.state_dict()
            "num_samples": len(data["X_train"])  # required for FedAvg
        })

        print(f"[Client {client_id}] Weights sent to server")

    print("\n--- All Client Weights Collected ---")

    return client_updates

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Human Activity Recognition - FL & Centralized Baseline")
    parser.add_argument('--mode', type=str, default='both', choices=['centralized', 'federated', 'both'],
                        help="Execution mode: run 'centralized' training, setup 'federated' test, or 'both'.")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # LOAD DATA
    print("\nLoading and partitioning dataset...")
    loader = UCIHARDataLoader("data/UCI_HAR")
    train_df, test_df = loader.load_full_dataset(use_inertial_signals=True)

    # PARTITION (CLIENT SIMULATION)
    partitioner = UserPartitioner(train_df, test_df)
    client_datasets = partitioner.create_clients()
    client_splits   = partitioner.split_clients(client_datasets)

    print(f"FL Clients (train subjects): {len(client_splits)}")

    # Execute Centralized Training Phase
    if args.mode in ['centralized', 'both']:
        run_centralized_training(client_splits, partitioner, device)

    # Execute Federated Clients Setup Phase
    if args.mode in ['federated', 'both']:
        clients, global_model = setup_federated_clients(client_splits, device)

        print("\n--- Running Federated Learning Round ---")

        # RUN FEDERATED ROUND (CLIENT SIDE)
        client_updates = run_federated_round(clients, client_splits)

        # COLLECT CLIENT UPDATES (SERVER SIDE)
        server = FederatedServer()

        collected_updates = server.collect_updates(client_updates)

        global_weights = server.fedavg_aggregate()

        print("\n--- Global Model Aggregated Successfully ---")

        # DEBUG: Inspect payload (Privacy Validation)
        print("\n--- Inspecting Client Payloads ---")

        for update in collected_updates:
            weights = update["weights"]
            first_tensor = list(weights.values())[0]

            print(
                f"Client {update['client_id']} -> "
                f"Samples: {update['num_samples']} | "
                f"Tensor Shape: {tuple(first_tensor.shape)}"
            )

        # Update the global model with the aggregated weights from FedAvg
        global_model.load_state_dict(global_weights)
        print("\n[SERVER] Global model successfully updated with aggregated weights.")

if __name__ == "__main__":
    main()