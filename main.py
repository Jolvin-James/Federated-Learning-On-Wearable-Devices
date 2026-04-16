# main.py
import argparse
import random
import os
import copy
import torch
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
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return acc, f1, all_preds, all_labels

# CENTRALIZED TRAINING
def run_centralized_training(client_splits, partitioner, device):
    print("\n========== CENTRALIZED TRAINING ==========")

    builder = CentralizedDataBuilder(client_splits)

    global_data = builder.combine_all_clients(
        add_client_id=False,
        shuffle=True
    )

    X_test_raw, y_test_raw = partitioner.get_global_test()

    X_train_raw = global_data["X_train"]

    mean = X_train_raw.mean()
    std = X_train_raw.std().replace(0, 1)

    X_train_norm = (X_train_raw - mean) / std
    X_test_norm = (X_test_raw - mean) / std

    X_train = reshape_for_cnn(X_train_norm)
    X_test = reshape_for_cnn(X_test_norm)

    y_train = torch.tensor(
        global_data["y_train"].values.squeeze() - 1,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    model = HAR_CNN().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "f1_score": []
    }

    epochs = 10
    best_acc = 0.0

    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)

        acc, f1, _, _ = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["f1_score"].append(f1)
        
        # ---------- Best Checkpoint ----------
        if acc > best_acc:
            best_acc = acc
            os.makedirs("results", exist_ok=True)
            torch.save(
                model.state_dict(),
                "results/centralized_model.pth"
            )
            print("Best centralized model saved.")

    pd.DataFrame(history).to_csv(
        "results/centralized_metrics.csv",
        index=False
    )

    print(f"\nCentralized training complete. Best Accuracy: {best_acc:.4f}")

    # Load and return the best evaluating model
    model.load_state_dict(torch.load("results/centralized_model.pth"))

    return model


# FEDERATED CLIENT SETUP
def setup_federated_clients(client_splits, device):
    print("\n========== SETTING UP CLIENTS ==========")

    global_model = HAR_CNN().to(device)

    global_weights = copy.deepcopy(
        global_model.state_dict()
    )

    clients = {}

    for client_id in client_splits.keys():
        client = FLClient(client_id, device)

        client.receive_global_model(global_weights)
        client.print_status()

        clients[client_id] = client

    return clients, global_model

# ONE FEDERATED ROUND
def run_federated_round(
    clients,
    client_splits,
    round_num=1
):
    print(f"\n========== ROUND {round_num} ==========")

    client_updates = []

    for client_id, client in clients.items():

        print(
            f"\n[Client {client_id}] "
            f"Starting Local Training..."
        )

        data = client_splits[client_id]

        updated_weights = client.local_train(
            data["X_train"],
            data["y_train"],
            epochs=2,
            batch_size=32,
            lr=0.001
        )

        client_updates.append({
            "client_id": client_id,
            "weights": updated_weights,
            "num_samples": len(data["X_train"])
        })

        print(
            f"[Client {client_id}] "
            f"Sent weights to server"
        )

    return client_updates

# FULL FEDERATED TRAINING LOOP
def run_federated_training(
    client_splits,
    partitioner,
    device,
    rounds=30,
    fraction=0.6,
    patience=6
):
    print("\n========== ADVANCED FEDERATED TRAINING ==========")

    # Better dropout
    global_model = HAR_CNN(dropout_rate=0.30).to(device)

    clients = {}
    global_weights = copy.deepcopy(global_model.state_dict())

    for client_id in client_splits.keys():
        client = FLClient(client_id, device)
        client.receive_global_model(global_weights)
        clients[client_id] = client

    server = FederatedServer(model=global_model)

    # ---------- Global Test Set ----------
    X_test_raw, y_test_raw = partitioner.get_global_test()

    all_train = []
    for cid in client_splits:
        all_train.append(client_splits[cid]["X_train"])

    X_train_all = pd.concat(all_train)

    mean = X_train_all.mean()
    std = X_train_all.std().replace(0, 1)

    # Normalize client data to prevent training on unnormalized data
    for cid in client_splits:
        client_splits[cid]["X_train"] = (client_splits[cid]["X_train"] - mean) / std

    X_test_norm = (X_test_raw - mean) / std
    X_test = reshape_for_cnn(X_test_norm)

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    # ---------- Tracking ----------
    history = {
        "round": [],
        "accuracy": [],
        "f1_score": [],
        "clients_used": [],
        "lr": []
    }

    best_acc = 0
    wait = 0

    base_lr = 0.001

    for round_num in range(1, rounds + 1):

        print(f"\n========== ROUND {round_num} ==========")

        # LR Scheduling
        current_lr = base_lr * (0.95 ** (round_num - 1))

        # Client Sampling
        all_clients = list(clients.keys())

        num_selected = max(
            2,
            int(len(all_clients) * fraction)
        )

        selected_ids = random.sample(
            all_clients,
            num_selected
        )

        print("Selected Clients:", selected_ids)

        client_updates = []

        # ---------- Local Training ----------
        for cid in selected_ids:

            client = clients[cid]
            data = client_splits[cid]

            updated_weights = client.local_train(
                data["X_train"],
                data["y_train"],
                epochs=3,
                batch_size=32,
                lr=current_lr
            )

            client_updates.append({
                "client_id": cid,
                "weights": updated_weights,
                "num_samples": len(data["X_train"])
            })

        # ---------- Server ----------
        server.collect_updates(client_updates)

        global_weights = server.fedavg_aggregate()

        server.update_global_model(global_weights)

        server.broadcast_global_model(clients)

        # ---------- Evaluate ----------
        acc, f1, _, _ = evaluate(
            server.get_global_model(),
            test_loader,
            device
        )

        print(
            f"[ROUND {round_num}] "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f}"
        )

        history["round"].append(round_num)
        history["accuracy"].append(acc)
        history["f1_score"].append(f1)
        history["clients_used"].append(num_selected)
        history["lr"].append(current_lr)

        # ---------- Best Checkpoint ----------
        if acc > best_acc:
            best_acc = acc
            wait = 0

            torch.save(
                server.get_global_model().state_dict(),
                "results/best_federated_model.pth"
            )

            print("Best model saved.")

        else:
            wait += 1

        # ---------- Early Stopping ----------
        if wait >= patience:
            print(
                f"\nEarly stopping at round {round_num}"
            )
            break

        server.clear_updates()

    os.makedirs("results", exist_ok=True)

    pd.DataFrame(history).to_csv(
        "results/federated_metrics.csv",
        index=False
    )

    print("\nFederated training complete.")
    print("Best Accuracy:", best_acc)

def main():
    parser = argparse.ArgumentParser(
        description="HAR Centralized + Federated"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=[
            "centralized",
            "federated",
            "both"
        ]
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using Device:", device)

    loader = UCIHARDataLoader(
        "data/UCI_HAR"
    )

    train_df, test_df = loader.load_full_dataset(
        use_inertial_signals=True
    )

    partitioner = UserPartitioner(
        train_df,
        test_df
    )

    client_datasets = partitioner.create_clients()

    client_splits = partitioner.split_clients(
        client_datasets
    )

    print(
        f"\nTotal FL Clients: "
        f"{len(client_splits)}"
    )

    # CENTRALIZED
    if args.mode in ["centralized", "both"]:
        run_centralized_training(
            client_splits,
            partitioner,
            device
        )

    # FEDERATED
    if args.mode in ["federated", "both"]:
        run_federated_training(
            client_splits,
            partitioner,
            device,
            rounds=args.rounds
        )


if __name__ == "__main__":
    main()