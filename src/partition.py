# src/partition.py
import pandas as pd
from sklearn.model_selection import train_test_split

class UserPartitioner:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initialize partitioner with train and test datasets

        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame): Testing dataframe
        """
        self.train_df = train_df
        self.test_df = test_df

    def create_clients(self):
        """
        Combine train + test data and split user-wise

        Returns:
            dict: {client_id: dataframe}
        """
        print("\n[INFO] Creating client datasets (user-wise)...")

        # Combine datasets
        full_df = pd.concat([self.train_df, self.test_df], axis=0).reset_index(drop=True)

        # Group by user (Subject column)
        user_groups = full_df.groupby("Subject")

        client_datasets = {}

        for user_id, user_data in user_groups:
            client_datasets[int(user_id)] = user_data.reset_index(drop=True)

        print(f"[INFO] Total clients created: {len(client_datasets)}")

        return client_datasets

    def split_clients(self, client_datasets, test_size=0.2):
        """
        Split each client's dataset into local train/test

        Args:
            client_datasets (dict): {client_id: dataframe}
            test_size (float): test split ratio

        Returns:
            dict: structured client splits
        """
        print("\n[INFO] Performing train-test split for each client...")

        client_splits = {}

        for client_id, data in client_datasets.items():

            # Separate features and labels
            X = data.drop(columns=["Activity", "Subject"])
            y = data["Activity"]

            # Skip small datasets
            if len(data) < 10:
                print(f"[WARNING] Skipping Client {client_id} (too few samples)")
                continue

            try:
                # Stratified split to preserve class distribution
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    stratify=y,
                    random_state=42
                )

                client_splits[client_id] = {
                    "X_train": X_train.reset_index(drop=True),
                    "X_test": X_test.reset_index(drop=True),
                    "y_train": y_train.reset_index(drop=True),
                    "y_test": y_test.reset_index(drop=True)
                }

            except ValueError:
                # Happens when stratification fails (rare edge case)
                print(f"[WARNING] Stratified split failed for Client {client_id}, using random split")

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42
                )

                client_splits[client_id] = {
                    "X_train": X_train.reset_index(drop=True),
                    "X_test": X_test.reset_index(drop=True),
                    "y_train": y_train.reset_index(drop=True),
                    "y_test": y_test.reset_index(drop=True)
                }

        print(f"[INFO] Successfully split {len(client_splits)} clients")

        return client_splits

    def validate_splits(self, client_splits, num_clients=3):
        """
        Validate client splits

        Args:
            client_splits (dict)
            num_clients (int): number of clients to print
        """
        print("\n[INFO] Validating client splits...")

        sample_clients = list(client_splits.keys())[:num_clients]

        for cid in sample_clients:
            data = client_splits[cid]

            print(f"\nClient {cid}")
            print("-" * 30)
            print("Train samples:", len(data["X_train"]))
            print("Test samples :", len(data["X_test"]))
            print("Train classes:", data["y_train"].nunique())
            print("Test classes :", data["y_test"].nunique())