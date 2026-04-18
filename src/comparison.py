# src/comparison.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ModelComparator:
    """
    Compare Centralized vs Federated Learning models
    Generates:
      - CSV summary
      - Performance graph
      - Time graph
      - Privacy graph
      - Final verdict
    """

    def __init__(self, save_dir="results"):
        self.results = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------------------------------------------------
    # Add Result
    # ---------------------------------------------------
    def add_result(
        self,
        model_name,
        accuracy,
        f1_score,
        loss,
        training_time,
        communication_rounds=0,
        privacy="Low"
    ):
        """
        Add one model result

        Args:
            model_name (str): Centralized / Federated
            accuracy (float)
            f1_score (float)
            loss (float)
            training_time (float)
            communication_rounds (int)
            privacy (str): Low / High
        """

        self.results.append({
            "Model": model_name,
            "Accuracy": round(float(accuracy), 4),
            "F1 Score": round(float(f1_score), 4),
            "Loss": round(float(loss), 4),
            "Training Time (sec)": round(float(training_time), 2),
            "Rounds": communication_rounds,
            "Privacy": privacy
        })

    # ---------------------------------------------------
    # Summary Table
    # ---------------------------------------------------
    def summary(self):
        df = pd.DataFrame(self.results)

        print("\n========== MODEL COMPARISON ==========")
        print(df.to_string(index=False))

        csv_path = os.path.join(
            self.save_dir,
            "comparison_table.csv"
        )

        df.to_csv(csv_path, index=False)

        print(f"\nSaved: {csv_path}")

        return df

    # ---------------------------------------------------
    # Performance Metrics Graph
    # ---------------------------------------------------
    def plot_scores(self):
        df = pd.DataFrame(self.results)

        x = np.arange(len(df))
        width = 0.25

        plt.figure(figsize=(10, 6))

        plt.bar(
            x - width,
            df["Accuracy"],
            width,
            label="Accuracy"
        )

        plt.bar(
            x,
            df["F1 Score"],
            width,
            label="F1 Score"
        )

        plt.bar(
            x + width,
            1 - df["Loss"],
            width,
            label="1 - Loss"
        )

        plt.xticks(
            x,
            df["Model"]
        )

        plt.ylim(0.70, 1.00)

        plt.title(
            "Centralized vs Federated Performance"
        )

        plt.ylabel("Score")
        plt.legend()

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            self.save_dir,
            "comparison_scores.png"
        )

        plt.savefig(path)
        plt.show()

        print("Saved:", path)

    # ---------------------------------------------------
    # Training Time Graph
    # ---------------------------------------------------
    def plot_time(self):
        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8, 5))

        plt.bar(
            df["Model"],
            df["Training Time (sec)"]
        )

        plt.title("Training Time Comparison")
        plt.ylabel("Seconds")

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            self.save_dir,
            "comparison_time.png"
        )

        plt.savefig(path)
        plt.show()

        print("Saved:", path)

    # ---------------------------------------------------
    # Privacy Comparison Graph
    # ---------------------------------------------------
    def plot_privacy(self):
        df = pd.DataFrame(self.results)

        privacy_map = {
            "Low": 3,
            "Medium": 6,
            "High": 10
        }

        scores = [
            privacy_map.get(v, 0)
            for v in df["Privacy"]
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            df["Model"],
            scores
        )

        plt.title("Privacy Comparison")
        plt.ylabel("Privacy Score (0-10)")
        plt.ylim(0, 10)

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            self.save_dir,
            "comparison_privacy.png"
        )

        plt.savefig(path)
        plt.show()

        print("Saved:", path)

    # ---------------------------------------------------
    # Communication Rounds Graph
    # ---------------------------------------------------
    def plot_rounds(self):
        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8, 5))

        plt.bar(
            df["Model"],
            df["Rounds"]
        )

        plt.title("Communication Rounds")
        plt.ylabel("Rounds")

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            self.save_dir,
            "comparison_rounds.png"
        )

        plt.savefig(path)
        plt.show()

        print("Saved:", path)

    # ---------------------------------------------------
    # Final Verdict
    # ---------------------------------------------------
    def verdict(self):
        df = pd.DataFrame(self.results)

        if len(df) < 2:
            print("\nNeed two models for verdict.")
            return

        centralized = df[
            df["Model"] == "Centralized"
        ].iloc[0]

        federated = df[
            df["Model"] == "Federated"
        ].iloc[0]

        acc_gap = (
            centralized["Accuracy"]
            - federated["Accuracy"]
        )

        f1_gap = (
            centralized["F1 Score"]
            - federated["F1 Score"]
        )

        print("\n========== FINAL ANALYSIS ==========")

        print(f"Accuracy Gap : {acc_gap:.4f}")
        print(f"F1 Gap       : {f1_gap:.4f}")

        if acc_gap <= 0.03:
            print(
                "\nFederated Learning achieved "
                "near-centralized performance."
            )
        else:
            print(
                "\nCentralized model performs better."
            )

        print(
            "Privacy Winner : Federated Learning"
        )

        if (
            federated["Training Time (sec)"]
            > centralized["Training Time (sec)"]
        ):
            print(
                "Centralized training is faster."
            )
        else:
            print(
                "Federated training is faster."
            )

    # ---------------------------------------------------
    # Run Everything
    # ---------------------------------------------------
    def run_all(self):
        self.summary()
        self.plot_scores()
        self.plot_time()
        self.plot_privacy()
        self.plot_rounds()
        self.verdict()