import argparse
import copy
import os
import random
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from main import (
    evaluate,
    run_centralized_training,
    run_federated_training,
    run_global_evaluation,
)
from privacy_comparison_demo import run_demo as run_privacy_comparison_demo
from src.client import FLClient
from src.comparison import ModelComparator
from src.dashboard_demo import build_dashboard_demo_split, describe_new_client
from src.dashboard_events import DashboardEventLogger
from src.data_loader import UCIHARDataLoader
from src.model_utils import reshape_for_cnn
from src.partition import UserPartitioner
from src.server import FederatedServer


ACTIVITY_LABELS = {
    "1": "Walking",
    "2": "Upstairs",
    "3": "Downstairs",
    "4": "Sitting",
    "5": "Standing",
    "6": "Laying",
}


class DashboardDemoPartitioner:
    def __init__(self, new_client_data):
        self.new_client_data = new_client_data.reset_index(drop=True)

    def get_global_test(self):
        X_test = self.new_client_data.drop(columns=["Activity", "Subject"])
        y_test = self.new_client_data["Activity"]
        return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def split_training_clients(training_clients):
    splitter = UserPartitioner(pd.DataFrame(), pd.DataFrame())
    return splitter.split_clients(training_clients)


def evaluate_model_on_frame(model, frame, mean, std, device):
    X_raw = frame.drop(columns=["Activity", "Subject"])
    y_raw = frame["Activity"]
    X = reshape_for_cnn((X_raw - mean) / std)
    y = torch.tensor(y_raw.values.squeeze() - 1, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=False)
    loss, acc, f1, _, _ = evaluate(model, loader, device)
    return {
        "loss": round(float(loss), 4),
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
    }


def run_new_client_onboarding(model, mean, std, new_client_id, new_client_data, device, logger):
    logger.log(
        "new_client",
        f"New client {new_client_id} connected",
        client_id=int(new_client_id),
        samples=int(len(new_client_data)),
    )
    logger.write_state(status="running", current_stage="new_client_onboarding")

    X = new_client_data.drop(columns=["Activity", "Subject"])
    y = new_client_data["Activity"]

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=42,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )

    eval_frame = X_val.reset_index(drop=True).copy()
    eval_frame["Activity"] = y_val.reset_index(drop=True)
    eval_frame["Subject"] = new_client_id

    before = evaluate_model_on_frame(model, eval_frame, mean, std, device)

    client = FLClient(new_client_id, device)
    client.receive_global_model(copy.deepcopy(model.state_dict()))

    X_train_norm = (X_train.reset_index(drop=True) - mean) / std
    updated_weights = client.local_train(
        X_train_norm,
        y_train.reset_index(drop=True),
        epochs=3,
        batch_size=32,
        lr=0.001,
    )

    payload = {
        "client_id": int(new_client_id),
        "weights": updated_weights,
        "num_samples": int(len(X_train_norm)),
    }

    logger.log(
        "new_client_update",
        "New client sent model update",
        client_id=int(new_client_id),
        payload_keys=list(payload.keys()),
        num_samples=int(len(X_train_norm)),
        raw_data_sent=False,
    )

    server = FederatedServer(model=model)
    server.collect_updates([payload])
    global_weights = server.fedavg_aggregate()
    server.update_global_model(global_weights)

    after = evaluate_model_on_frame(server.get_global_model(), eval_frame, mean, std, device)

    logger.log(
        "new_client_complete",
        "New client update merged into global model",
        client_id=int(new_client_id),
        before_accuracy=before["accuracy"],
        after_accuracy=after["accuracy"],
        validation_samples=int(len(eval_frame)),
    )

    plot_new_client_result(before, after)

    return {
        "client_id": int(new_client_id),
        "train_samples": int(len(X_train_norm)),
        "validation_samples": int(len(eval_frame)),
        "payload_keys": list(payload.keys()),
        "raw_data_sent": False,
        "before": before,
        "after": after,
    }


def plot_new_client_result(before, after):
    os.makedirs("results", exist_ok=True)
    labels = ["Before update", "After update"]
    values = [before["accuracy"], after["accuracy"]]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=["#64748b", "#0f766e"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("New Client Local Validation Accuracy")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("results/new_client_accuracy.png", dpi=220)
    plt.close()


def build_demo_metadata(demo_split):
    new_client = describe_new_client(
        demo_split.new_client_id,
        demo_split.new_client_data,
    )
    return {
        "mode": "dashboard_demo_29_plus_1",
        "total_subjects": demo_split.total_subjects,
        "training_client_count": demo_split.training_client_count,
        "new_client_id": int(demo_split.new_client_id),
        "all_subjects": [int(subject) for subject in demo_split.all_subjects],
        "training_subjects": [int(subject) for subject in sorted(demo_split.training_clients.keys())],
        "client_sample_counts": {
            str(k): int(v)
            for k, v in sorted(demo_split.client_sample_counts.items())
        },
        "new_client": new_client,
        "activity_labels": ACTIVITY_LABELS,
    }


def run_dashboard_demo(rounds, holdout_client, output_dir):
    random.seed(42)
    torch.manual_seed(42)

    logger = DashboardEventLogger(output_dir)
    logger.reset()
    start = time.time()

    try:
        logger.write_state(status="running", current_stage="privacy_demo")
        logger.log("stage", "Running privacy payload comparison")
        run_privacy_comparison_demo(output_dir="results")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log("stage", "Loading UCI HAR dataset", device=str(device))
        logger.write_state(status="running", current_stage="load_dataset")

        loader = UCIHARDataLoader("data")
        train_df, test_df = loader.load_full_dataset(use_inertial_signals=True)

        demo_split = build_dashboard_demo_split(
            train_df,
            test_df,
            holdout_client=holdout_client,
        )
        metadata = build_demo_metadata(demo_split)
        logger.write_json("metadata.json", metadata)

        logger.log(
            "split",
            "Created dashboard 29+1 split",
            training_clients=demo_split.training_client_count,
            new_client_id=int(demo_split.new_client_id),
            total_subjects=demo_split.total_subjects,
        )

        client_splits = split_training_clients(demo_split.training_clients)
        partitioner = DashboardDemoPartitioner(demo_split.new_client_data)
        comparator = ModelComparator()

        logger.write_state(status="running", current_stage="centralized_training")
        logger.log("stage", "Centralized training started", client_count=len(client_splits))
        (
            central_model,
            c_mean,
            c_std,
            c_time,
            c_loss,
            c_acc,
            c_f1,
        ) = run_centralized_training(client_splits, partitioner, device)

        run_global_evaluation(central_model, partitioner, c_mean, c_std, device, "Centralized")

        comparator.add_result(
            model_name="Centralized",
            accuracy=c_acc,
            f1_score=c_f1,
            loss=c_loss,
            training_time=c_time,
            communication_rounds=0,
            privacy="Low",
        )

        logger.log(
            "stage_complete",
            "Centralized training complete",
            accuracy=round(float(c_acc), 4),
            f1=round(float(c_f1), 4),
            loss=round(float(c_loss), 4),
            training_time_sec=round(float(c_time), 2),
        )

        logger.write_state(status="running", current_stage="federated_training")
        logger.log("stage", "Federated training started", rounds=int(rounds))

        def event_hook(event_type, title, **fields):
            logger.log(event_type, title, **fields)

        (
            fed_model,
            f_mean,
            f_std,
            server,
            f_time,
            f_loss,
            f_acc,
            f_f1,
        ) = run_federated_training(
            client_splits,
            partitioner,
            device,
            rounds=rounds,
            event_hook=event_hook,
        )

        run_global_evaluation(fed_model, partitioner, f_mean, f_std, device, "Federated")

        comparator.add_result(
            model_name="Federated",
            accuracy=f_acc,
            f1_score=f_f1,
            loss=f_loss,
            training_time=f_time,
            communication_rounds=rounds,
            privacy="High",
        )

        server.packet_analyzer.summary()
        server.packet_analyzer.save_csv()
        server.packet_analyzer.save_json()

        if len(comparator.results) >= 2:
            comparator.run_all()

        logger.write_state(status="running", current_stage="new_client_onboarding")
        new_client_result = run_new_client_onboarding(
            fed_model,
            f_mean,
            f_std,
            demo_split.new_client_id,
            demo_split.new_client_data,
            device,
            logger,
        )
        logger.write_json("new_client_result.json", new_client_result)

        summary = {
            "status": "completed",
            "rounds": int(rounds),
            "holdout_client": int(demo_split.new_client_id),
            "training_clients": int(demo_split.training_client_count),
            "elapsed_sec": round(time.time() - start, 2),
            "centralized": {
                "accuracy": round(float(c_acc), 4),
                "f1": round(float(c_f1), 4),
                "loss": round(float(c_loss), 4),
                "training_time_sec": round(float(c_time), 2),
            },
            "federated": {
                "accuracy": round(float(f_acc), 4),
                "f1": round(float(f_f1), 4),
                "loss": round(float(f_loss), 4),
                "training_time_sec": round(float(f_time), 2),
                "rounds": int(rounds),
            },
            "new_client": new_client_result,
        }
        logger.write_json("summary.json", summary)
        logger.log("complete", "Dashboard demo run complete", elapsed_sec=summary["elapsed_sec"])
        logger.write_state(status="completed", current_stage="complete", summary=summary)

    except Exception as exc:
        logger.log("error", "Dashboard demo failed", error=str(exc))
        logger.write_state(status="failed", current_stage="error", error=str(exc))
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 29+1 dashboard demo pipeline.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--holdout-client", type=int, default=30)
    parser.add_argument("--output-dir", default=os.path.join("results", "dashboard"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_dashboard_demo(
        rounds=args.rounds,
        holdout_client=args.holdout_client,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
