# main.py

import argparse
import copy
import os
import pickle
import random
import socket
import struct
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import UCIHARDataLoader
from src.partition import UserPartitioner
from src.centralized_data import CentralizedDataBuilder
from src.model import HAR_CNN
from src.model_utils import reshape_for_cnn
from src.client import FLClient
from src.server import FederatedServer
from src.comparison import ModelComparator
from privacy_comparison_demo import run_demo as run_privacy_comparison_demo

HOST = "127.0.0.1"
PORT = 5000
BUFFER = 4096
FL_PACKET_DEMO_DELAY_SEC = 0.15


def emit_dashboard_event(event_hook, event_type, title, **fields):
    if event_hook is not None:
        event_hook(event_type, title, **fields)

def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(min(BUFFER, n - len(data)))
        if not packet:
            return None
        data += packet
    return data


def send_payload(payload):
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(">Q", len(raw))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(header + raw)
    s.close()


def receive_payload(conn):
    header = recvall(conn, 8)

    if not header:
        return None

    msg_len = struct.unpack(">Q", header)[0]
    body = recvall(conn, msg_len)

    if not body:
        return None

    return pickle.loads(body)

def plot_training_curves(history, model_name):
    os.makedirs("results", exist_ok=True)

    if "loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(history["loss"], marker="o")
        plt.title(f"{model_name} Loss Curve")
        plt.xlabel("Epoch / Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{model_name.lower()}_loss.png")
        plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], marker="o")
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epoch / Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{model_name.lower()}_accuracy.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history["f1"], marker="o")
    plt.title(f"{model_name} F1 Score Curve")
    plt.xlabel("Epoch / Round")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{model_name.lower()}_f1.png")
    plt.show()

def compute_norm_stats(client_splits):
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

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

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


# QUICK GLOBAL TEST
def quick_global_test(model, partitioner, mean, std, device):
    X_test_raw, y_test_raw = partitioner.get_global_test()

    X_test = reshape_for_cnn((X_test_raw - mean) / std)

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


# CENTRALIZED TRAINING
def run_centralized_training(client_splits, partitioner, device):
    print("\n========== CENTRALIZED TRAINING ==========")

    start_time = time.time()

    builder = CentralizedDataBuilder(client_splits)

    global_data = builder.combine_all_clients(
        add_client_id=False,
        shuffle=True
    )

    X_train_raw = global_data["X_train"]
    y_train_raw = global_data["y_train"]

    X_test_raw, y_test_raw = partitioner.get_global_test()

    mean = X_train_raw.mean()
    std = X_train_raw.std().replace(0, 1)

    X_train = reshape_for_cnn((X_train_raw - mean) / std)
    X_test = reshape_for_cnn((X_test_raw - mean) / std)

    y_train = torch.tensor(
        y_train_raw.values.squeeze() - 1,
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = torch.nn.CrossEntropyLoss()

    history = {
        "loss": [],
        "accuracy": [],
        "f1": []
    }

    best_acc = 0.0

    for epoch in range(10):

        loss = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, acc, f1, _, _ = evaluate(
            model,
            test_loader,
            device
        )

        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["f1"].append(f1)

        print(
            f"Epoch {epoch+1}/10 | "
            f"Loss={loss:.4f} | "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            os.makedirs("results", exist_ok=True)
            torch.save(
                model.state_dict(),
                "results/centralized_best.pth"
            )

    model.load_state_dict(
        torch.load("results/centralized_best.pth")
    )

    plot_training_curves(history, "Centralized")

    total_time = time.time() - start_time

    final_loss, final_acc, final_f1, _, _ = evaluate(
        model,
        test_loader,
        device
    )

    return model, mean, std, total_time, final_loss, final_acc, final_f1

# FEDERATED TRAINING
def run_federated_training(
    client_splits,
    partitioner,
    device,
    rounds=15,
    event_hook=None
):
    print("\n========== FEDERATED TRAINING ==========")
    print("Wireshark Filter -> tcp.port == 5000")

    start_time = time.time()

    global_model = HAR_CNN().to(device)
    server = FederatedServer(model=global_model)

    clients = {}

    history = {
        "accuracy": [],
        "f1": []
    }

    global_weights = copy.deepcopy(
        global_model.state_dict()
    )

    for cid in client_splits:
        client = FLClient(cid, device)
        client.receive_global_model(global_weights)
        clients[cid] = client

    emit_dashboard_event(
        event_hook,
        "clients",
        "Federated clients initialized",
        client_count=len(clients),
        client_ids=sorted(int(cid) for cid in clients.keys()),
    )

    mean, std = compute_norm_stats(client_splits)

    for cid in client_splits:
        client_splits[cid]["X_train"] = (
            client_splits[cid]["X_train"] - mean
        ) / std

        client_splits[cid]["X_test"] = (
            client_splits[cid]["X_test"] - mean
        ) / std

    best_acc = 0.0

    for round_num in range(1, rounds + 1):

        print(f"\n========== ROUND {round_num} ==========")

        selected = random.sample(
            list(clients.keys()),
            max(2, int(len(clients) * 0.6))
        )

        emit_dashboard_event(
            event_hook,
            "round_started",
            f"Federated round {round_num} started",
            round=round_num,
            selected_clients=sorted(int(cid) for cid in selected),
            selected_count=len(selected),
        )

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        sock.bind((HOST, PORT))
        sock.listen(len(selected))

        received_updates = []

        try:
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

                payload = {
                    "client_id": cid,
                    "weights": updated_weights,
                    "num_samples": len(data["X_train"])
                }

                emit_dashboard_event(
                    event_hook,
                    "client_update",
                    f"Client {cid} sent model update",
                    round=round_num,
                    client_id=int(cid),
                    num_samples=int(len(data["X_train"])),
                    payload_keys=list(payload.keys()),
                    raw_data_sent=False,
                )

                send_payload(payload)
                print(
                    f"[SOCKET] Client {cid} sent model update "
                    f"to {HOST}:{PORT}"
                )

                conn, _ = sock.accept()
                packet = receive_payload(conn)

                if packet is not None:
                    received_updates.append(packet)

                conn.close()

                # Keeps packets visible enough for a live Wireshark demo.
                time.sleep(FL_PACKET_DEMO_DELAY_SEC)

        finally:
            sock.close()

        server.collect_updates(received_updates)

        global_weights = server.fedavg_aggregate()

        emit_dashboard_event(
            event_hook,
            "aggregation",
            f"Server aggregated round {round_num}",
            round=round_num,
            received_updates=len(received_updates),
            total_samples=int(server.total_samples),
            method="FedAvg",
        )

        server.update_global_model(global_weights)

        server.broadcast_global_model(clients)

        acc, f1 = quick_global_test(
            server.get_global_model(),
            partitioner,
            mean,
            std,
            device
        )

        history["accuracy"].append(acc)
        history["f1"].append(f1)

        print(
            f"[ROUND {round_num}] "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f}"
        )

        emit_dashboard_event(
            event_hook,
            "round_completed",
            f"Round {round_num} evaluation complete",
            round=round_num,
            accuracy=round(float(acc), 4),
            f1=round(float(f1), 4),
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
        torch.load("results/federated_best.pth")
    )

    plot_training_curves(history, "Federated")

    total_time = time.time() - start_time

    X_test_raw, y_test_raw = partitioner.get_global_test()

    X_test = reshape_for_cnn((X_test_raw - mean) / std)

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    final_loss, final_acc, final_f1, _, _ = evaluate(
        global_model,
        loader,
        device
    )

    return (
        global_model,
        mean,
        std,
        server,
        total_time,
        final_loss,
        final_acc,
        final_f1
    )

# GLOBAL TEST
def run_global_evaluation(
    model,
    partitioner,
    mean,
    std,
    device,
    model_name
):
    print(f"\n========== {model_name} TEST ==========")

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.metrics import (
        confusion_matrix,
        classification_report
    )

    os.makedirs("results", exist_ok=True)

    # HAR Activity Labels
    class_names = [
        "Walking",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing",
        "Laying"
    ]

    # Load Global Test Set
    X_test_raw, y_test_raw = partitioner.get_global_test()

    # Normalize using training stats
    X_test = reshape_for_cnn(
        (X_test_raw - mean) / std
    )

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    # Evaluate Model
    loss, acc, f1, preds, labels = evaluate(
        model,
        loader,
        device
    )

    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")

    cm = confusion_matrix(labels, preds)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=class_names,
            digits=4
        )
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )

    plt.title(
        f"{model_name} Confusion Matrix (Counts)",
        fontsize=14
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    plt.tight_layout()

    save_path = (
        f"results/"
        f"{model_name.lower()}_confusion_matrix.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)

    # NORMALIZED CONFUSION MATRIX (%)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )

    plt.title(
        f"{model_name} Confusion Matrix (%)",
        fontsize=14
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    plt.tight_layout()

    save_path2 = (
        f"results/"
        f"{model_name.lower()}_confusion_matrix_percent.png"
    )

    plt.savefig(save_path2, dpi=300)
    plt.show()

    print("Saved:", save_path2)

    return loss, acc, f1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="full_demo",
        choices=[
            "centralized",
            "federated",
            "both",
            "privacy_demo",
            "full_demo"
        ]
    )

    args = parser.parse_args()

    if args.mode in ["privacy_demo", "full_demo"]:
        run_privacy_comparison_demo()
        if args.mode == "privacy_demo":
            return

        print(
            "\n========== STARTING MAIN TRAINING PIPELINE =========="
        )

        args.mode = "both"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using Device:", device)

    loader = UCIHARDataLoader("data")

    train_df, test_df = loader.load_full_dataset(
        use_inertial_signals=True
    )

    partitioner = UserPartitioner(
        train_df,
        test_df
    )

    client_data = partitioner.create_clients()

    client_splits = partitioner.split_clients(client_data)

    comparator = ModelComparator()

    # CENTRALIZED
    if args.mode in ["centralized", "both"]:

        (
            central_model,
            c_mean,
            c_std,
            c_time,
            c_loss,
            c_acc,
            c_f1
        ) = run_centralized_training(
            client_splits,
            partitioner,
            device
        )

        run_global_evaluation(
            central_model,
            partitioner,
            c_mean,
            c_std,
            device,
            "Centralized"
        )

        comparator.add_result(
            model_name="Centralized",
            accuracy=c_acc,
            f1_score=c_f1,
            loss=c_loss,
            training_time=c_time,
            communication_rounds=0,
            privacy="Low"
        )

    # FEDERATED
    if args.mode in ["federated", "both"]:

        (
            fed_model,
            f_mean,
            f_std,
            server,
            f_time,
            f_loss,
            f_acc,
            f_f1
        ) = run_federated_training(
            client_splits,
            partitioner,
            device
        )

        run_global_evaluation(
            fed_model,
            partitioner,
            f_mean,
            f_std,
            device,
            "Federated"
        )

        comparator.add_result(
            model_name="Federated",
            accuracy=f_acc,
            f1_score=f_f1,
            loss=f_loss,
            training_time=f_time,
            communication_rounds=15,
            privacy="High"
        )

        print("\n========== PACKET ANALYSIS ==========")

        server.packet_analyzer.summary()
        server.packet_analyzer.save_csv()
        server.packet_analyzer.save_json()

    if len(comparator.results) >= 2:
        comparator.run_all()


if __name__ == "__main__":
    main()
