# src/centralized_data.py
import numpy as np
import pandas as pd

class CentralizedDataBuilder:
    def __init__(self, client_splits: dict):
        """
        Initialize with client-wise split data
        """
        self.client_splits = client_splits

    def combine_all_clients(self, add_client_id=True, shuffle=True):
        """
        Combine all client datasets into a centralized dataset

        Args:
            add_client_id (bool): keep client identity
            shuffle (bool): shuffle combined dataset

        Returns:
            dict
        """
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        if not self.client_splits:
            raise ValueError(
                "No client splits were provided. For UCI HAR, do not intersect "
                "train/test users. Combine all subjects first, then split each "
                "client locally into train/test data."
            )

        for client_id, data in self.client_splits.items():

            # Safety check
            required_keys = ["X_train", "X_test", "y_train", "y_test"]
            if not all(k in data for k in required_keys):
                raise ValueError(f"Client {client_id} missing required keys")

            X_train = data["X_train"].copy()
            X_test = data["X_test"].copy()

            y_train = data["y_train"].copy()
            y_test = data["y_test"].copy()

            # Preserve client identity
            if add_client_id:
                X_train["client_id"] = client_id
                X_test["client_id"] = client_id

            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

        # Combine — reset_index immediately so each DataFrame has a unique
        # 0..N-1 positional index before any shuffling.
        X_train_global = pd.concat(X_train_list, axis=0).reset_index(drop=True)
        X_test_global  = pd.concat(X_test_list,  axis=0).reset_index(drop=True)
        y_train_global = pd.concat(y_train_list, axis=0).reset_index(drop=True)
        y_test_global  = pd.concat(y_test_list,  axis=0).reset_index(drop=True)

        # Shuffle (prevents client-order bias)
        # Use positional .iloc + np permutation — never .loc on a non-unique index.
        if shuffle:
            rng = np.random.RandomState(42)

            train_perm = rng.permutation(len(X_train_global))
            X_train_global = X_train_global.iloc[train_perm].reset_index(drop=True)
            y_train_global = y_train_global.iloc[train_perm].reset_index(drop=True)

            test_perm = rng.permutation(len(X_test_global))
            X_test_global = X_test_global.iloc[test_perm].reset_index(drop=True)
            y_test_global = y_test_global.iloc[test_perm].reset_index(drop=True)

        return {
            "X_train": X_train_global,
            "X_test": X_test_global,
            "y_train": y_train_global,
            "y_test": y_test_global
        }

    def get_summary(self, global_data: dict):
        """
        Detailed dataset summary
        """
        print("\n--- CENTRALIZED DATA SUMMARY ---")

        print("\nShapes:")
        print("Train:", global_data["X_train"].shape)
        print("Test :", global_data["X_test"].shape)

        print("\nLabel Distribution (Train):")
        print(global_data["y_train"].value_counts())

        print("\nLabel Distribution (Test):")
        print(global_data["y_test"].value_counts())

        print("\nUnique Classes:")
        print("Train:", global_data["y_train"].nunique())
        print("Test :", global_data["y_test"].nunique())
