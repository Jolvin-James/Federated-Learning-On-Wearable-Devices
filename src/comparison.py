# src/comparison.py

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ModelComparator:
    """
    Compare Centralized vs Federated Learning Models
    """
    def __init__(self):
        self.results = []

    # Add Model Result
    def add_result(
        self,
        model_name,
        accuracy,
        f1_score,
        loss,
        training_time=0,
        communication_rounds=0,
        privacy="Low"
    ):
        self.results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "F1 Score": f1_score,
            "Loss": loss,
            "Training Time (sec)": training_time,
            "Rounds": communication_rounds,
            "Privacy": privacy
        })

    # Table Summary
    def summary(self):
        df = pd.DataFrame(self.results)

        print("\n========== COMPARISON ==========")
        print(df)

        os.makedirs("results", exist_ok=True)

        df.to_csv(
            "results/comparison_table.csv",
            index=False
        )

        return df

    # Bar Chart Metrics
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

        plt.ylim(0.75, 1.0)

        plt.title(
            "Centralized vs Federated Comparison"
        )

        plt.ylabel("Score")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "results/comparison_scores.png"
        )

        plt.show()

    # Training Time Graph
    def plot_time(self):
        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8, 5))

        plt.bar(
            df["Model"],
            df["Training Time (sec)"]
        )

        plt.title("Training Time Comparison")

        plt.ylabel("Seconds")

        plt.tight_layout()

        plt.savefig(
            "results/comparison_time.png"
        )

        plt.show()

    # Text Report
    def verdict(self):

        df = pd.DataFrame(self.results)

        centralized = df[df["Model"] == "Centralized"].iloc[0]
        federated = df[df["Model"] == "Federated"].iloc[0]

        print("\n========== FINAL ANALYSIS ==========")

        acc_gap = centralized["Accuracy"] - federated["Accuracy"]
        f1_gap = centralized["F1 Score"] - federated["F1 Score"]

        print(f"Accuracy Gap : {acc_gap:.4f}")
        print(f"F1 Gap       : {f1_gap:.4f}")

        if acc_gap <= 0.03:
            print(
                "\nFederated Learning achieved "
                "near-centralized performance."
            )
            print(
                "Recommended for real-world deployment "
                "because privacy is preserved."
            )
        else:
            print(
                "\nCentralized performs better,"
                " but privacy risk is higher."
            )

        print("\nPrivacy Winner : Federated Learning")

        if federated["Training Time (sec)"] > centralized["Training Time (sec)"]:
            print("Centralized trains faster.")
        else:
            print("Federated trains faster.")

    # Full Run
    def run_all(self):
        self.summary()
        self.plot_scores()
        self.plot_time()
        self.verdict()