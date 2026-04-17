# main.py

import argparse
import random
import os
import copy
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

from src.data_loader import UCIHARDataLoader
from src.partition import UserPartitioner
from src.centralized_data import CentralizedDataBuilder
from src.model import HAR_CNN
from src.model_utils import reshape_for_cnn
from src.client import FLClient
from src.server import FederatedServer
from src.comparison import ModelComparator

# Normalisation Helper
def compute_norm_stats(client_splits):
    """
    Compute global mean and std from all clients' RAW training data.
    Call this ONCE before any in-place normalisation so that the
    same stats can be reused for test-set scaling.
    """
    all_train = [client_splits[cid]["X_train"] for cid in client_splits]
    X_all = pd.concat(all_train)
    mean = X_all.mean()
    std = X_all.std().replace(0, 1)
    return mean, std

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
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            loss = criterion(outputs, y)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)

    acc = accuracy_score(all_labels, all_preds)

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    return avg_loss, acc, f1, all_preds, all_labels

# Centralized Training
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    best_acc = 0.0

    for epoch in range(10):

        loss = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        _, acc, f1, _, _ = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch+1}/10 | "
            f"Loss: {loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

        if acc > best_acc:
            best_acc = acc

            os.makedirs("results", exist_ok=True)

            torch.save(
                model.state_dict(),
                "results/centralized_best.pth"
            )

    model.load_state_dict(
        torch.load("results/centralized_best.pth", weights_only=True)
    )

    print(f"\nBest Centralized Accuracy: {best_acc:.4f}")

    return model, mean, std

# Quick Global Test
def quick_global_test(model, partitioner, mean, std, device):
    """
    Evaluate model on the global test set.

    Args:
        mean, std: pre-computed from training data BEFORE any
                   in-place normalisation occurs.
    """
    X_test_raw, y_test_raw = partitioner.get_global_test()

    X_test_norm = (X_test_raw - mean) / std

    X_test = reshape_for_cnn(X_test_norm)

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    _, acc, f1, _, _ = evaluate(model, loader, device)

    return acc, f1


# Federated Training
def run_federated_training(
    client_splits,
    partitioner,
    device,
    rounds=15
):

    print("\n========== FEDERATED TRAINING ==========")

    global_model = HAR_CNN().to(device)

    clients = {}

    global_weights = copy.deepcopy(
        global_model.state_dict()
    )

    for cid in client_splits.keys():

        client = FLClient(cid, device)

        client.receive_global_model(global_weights)

        clients[cid] = client

    server = FederatedServer(model=global_model)

    # Compute stats BEFORE normalising so evaluation functions
    # can reuse the same scale without re-reading mutated data.
    mean, std = compute_norm_stats(client_splits)

    for cid in client_splits:
        client_splits[cid]["X_train"] = (
            client_splits[cid]["X_train"] - mean
        ) / std

    best_acc = 0.0

    for round_num in range(1, rounds + 1):

        print(f"\n========== ROUND {round_num} ==========")

        selected = random.sample(
            list(clients.keys()),
            max(2, int(len(clients) * 0.6))
        )

        updates = []

        for cid in selected:

            client = clients[cid]

            data = client_splits[cid]

            updated_weights = client.local_train(
                data["X_train"],
                data["y_train"],
                epochs=3,
                batch_size=32,
                lr=0.001
            )

            updates.append({
                "client_id": cid,
                "weights": updated_weights,
                "num_samples": len(data["X_train"])
            })

        server.collect_updates(updates)

        global_weights = server.fedavg_aggregate()

        server.update_global_model(global_weights)

        server.broadcast_global_model(clients)

        acc, f1 = quick_global_test(
            server.get_global_model(),
            partitioner,
            mean,
            std,
            device
        )

        print(
            f"[ROUND {round_num}] "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f}"
        )

        if acc > best_acc:

            best_acc = acc

            os.makedirs("results", exist_ok=True)

            torch.save(
                server.get_global_model().state_dict(),
                "results/federated_best.pth"
            )

        server.clear_updates()

    global_model.load_state_dict(
        torch.load("results/federated_best.pth", weights_only=True)
    )

    print(f"\nBest Federated Accuracy: {best_acc:.4f}")

    return global_model, mean, std

# Global Evaluation
def run_global_evaluation(
    model,
    partitioner,
    mean,
    std,
    device,
    model_name="Model"
):
    """
    Args:
        mean, std: norm stats from the corresponding training run.
    """

    print(
        f"\n========== GLOBAL EVALUATION : {model_name} =========="
    )

    X_test_raw, y_test_raw = partitioner.get_global_test()

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

    loss, acc, f1, preds, labels = evaluate(
        model,
        test_loader,
        device
    )

    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    cm = confusion_matrix(labels, preds)

    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        labels,
        preds
    )

    print("\nClassification Report:")
    print(report)

    os.makedirs("results", exist_ok=True)

    pd.DataFrame(cm).to_csv(
        f"results/{model_name.lower()}_cm.csv",
        index=False
    )

    with open(
        f"results/{model_name.lower()}_report.txt",
        "w"
    ) as f:
        f.write(report)

# Final Metrics
def run_metrics_comparison(
    centralized_model,
    federated_model,
    partitioner,
    c_mean,
    c_std,
    f_mean,
    f_std,
    device
):

    print("\n========== FINAL METRICS ==========")

    c_acc, c_f1 = quick_global_test(
        centralized_model, partitioner, c_mean, c_std, device
    )

    f_acc, f_f1 = quick_global_test(
        federated_model, partitioner, f_mean, f_std, device
    )

    metrics = pd.DataFrame({
        "Model": ["Centralized", "Federated"],
        "Accuracy": [c_acc, f_acc],
        "F1 Score": [c_f1, f_f1]
    })

    print(metrics)

    acc_gap = c_acc - f_acc
    f1_gap = c_f1 - f_f1

    print("\nPerformance Gap:")
    print(f"Accuracy Drop : {acc_gap:.4f}")
    print(f"F1 Drop       : {f1_gap:.4f}")

    os.makedirs("results", exist_ok=True)

    metrics.to_csv(
        "results/final_metrics.csv",
        index=False
    )

    # Bar Chart
    plt.figure(figsize=(8, 5))

    x = np.arange(2)

    plt.bar(
        x - 0.15,
        metrics["Accuracy"],
        width=0.3,
        label="Accuracy"
    )

    plt.bar(
        x + 0.15,
        metrics["F1 Score"],
        width=0.3,
        label="F1 Score"
    )

    plt.xticks(
        x,
        metrics["Model"]
    )

    plt.ylim(0.80, 1.00)

    plt.title(
        "Centralized vs Federated Performance"
    )

    plt.ylabel("Score")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/model_comparison.png"
    )

    plt.show()

    print("\nMetrics saved in results/")

def main():

    parser = argparse.ArgumentParser()

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

    client_data = partitioner.create_clients()

    client_splits = partitioner.split_clients(
        client_data
    )

    central_model = None
    fed_model = None
    c_mean = c_std = None
    f_mean = f_std = None
    c_time = f_time = 0.0

    # Centralized
    if args.mode in ["centralized", "both"]:

        t0 = time.time()
        central_model, c_mean, c_std = run_centralized_training(
            client_splits,
            partitioner,
            device
        )
        c_time = time.time() - t0

        run_global_evaluation(
            central_model,
            partitioner,
            c_mean,
            c_std,
            device,
            model_name="Centralized"
        )

    # Federated
    if args.mode in ["federated", "both"]:

        t0 = time.time()
        fed_model, f_mean, f_std = run_federated_training(
            client_splits,
            partitioner,
            device
        )
        f_time = time.time() - t0

        run_global_evaluation(
            fed_model,
            partitioner,
            f_mean,
            f_std,
            device,
            model_name="Federated"
        )

    # Metrics
    if args.mode == "both":

        run_metrics_comparison(
            central_model,
            fed_model,
            partitioner,
            c_mean,
            c_std,
            f_mean,
            f_std,
            device
        )

        comparator = ModelComparator()

        # Fetch test set once; normalise separately for each model
        X_test_raw, y_test_raw = partitioner.get_global_test()
        y_test_tensor = torch.tensor(
            y_test_raw.values.squeeze() - 1, dtype=torch.long
        )

        # Centralized scores
        c_loss, c_acc, c_f1, _, _ = evaluate(
            central_model,
            DataLoader(
                TensorDataset(
                    reshape_for_cnn((X_test_raw - c_mean) / c_std),
                    y_test_tensor
                ),
                batch_size=64
            ),
            device
        )

        # Federated scores
        f_loss, f_acc, f_f1, _, _ = evaluate(
            fed_model,
            DataLoader(
                TensorDataset(
                    reshape_for_cnn((X_test_raw - f_mean) / f_std),
                    y_test_tensor
                ),
                batch_size=64
            ),
            device
        )

        comparator.add_result(
            model_name="Centralized",
            accuracy=c_acc,
            f1_score=c_f1,
            loss=c_loss,
            training_time=round(c_time, 1),
            communication_rounds=0,
            privacy="Low"
        )

        comparator.add_result(
            model_name="Federated",
            accuracy=f_acc,
            f1_score=f_f1,
            loss=f_loss,
            training_time=round(f_time, 1),
            communication_rounds=15,
            privacy="High"
        )

        comparator.run_all()


if __name__ == "__main__":
    main()