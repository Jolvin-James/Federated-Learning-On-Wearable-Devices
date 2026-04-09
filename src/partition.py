import pandas as pd
from sklearn.model_selection import train_test_split

class UserPartitioner:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Args:
            train_df: DataFrame containing ONLY the 21 training subjects
                      (as provided by UCI HAR's original train/ split).
            test_df:  DataFrame containing ONLY the 9 held-out test subjects
                      (as provided by UCI HAR's original test/ split).

        IMPORTANT: Do NOT merge train_df and test_df before calling this class.
        UCI HAR guarantees subject-disjoint splits, and merging would introduce
        subject-level data leakage (same person in both train and test).
        """
        self.train_df = train_df
        self.test_df = test_df

    def create_clients(self):
        """
        Create one federated client per training subject.

        Only the 21 TRAINING subjects become FL clients. The 9 test subjects
        in test_df are reserved as the global held-out evaluation set and are
        NEVER used during local training. This preserves the subject-disjoint
        property that UCI HAR was designed to have.
        """
        print("\n[INFO] Creating client datasets (training subjects only)...")

        user_groups = self.train_df.groupby("Subject")

        client_datasets = {}
        for user_id, user_data in user_groups:
            client_datasets[int(user_id)] = user_data.reset_index(drop=True)

        print(f"[INFO] Total FL clients (train subjects): {len(client_datasets)}")

        return client_datasets

    def get_global_test(self):
        """
        Return the original UCI HAR test set (9 held-out subjects) as the
        global evaluation set for the centralized and federated models.

        Returns:
            X_test (DataFrame), y_test (Series)
        """
        X_test = self.test_df.drop(columns=["Activity", "Subject"])
        y_test = self.test_df["Activity"]
        return X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    def split_clients(self, client_datasets, test_size=0.2):
        """
        Split each FL client's (training subject's) local data into a
        local train / local validation split.

        The local test slice is only used for per-client validation.
        The GLOBAL test set (get_global_test) is what matters for final
        accuracy reporting — never the per-client test slices.
        """
        print("\n[INFO] Performing local train/val split for each client...")

        client_splits = {}

        for client_id, data in client_datasets.items():
            X = data.drop(columns=["Activity", "Subject"])
            y = data["Activity"]

            if len(data) < 10:
                print(f"[WARNING] Skipping Client {client_id} (too few samples)")
                continue

            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    stratify=y,
                    random_state=42,
                )
            except ValueError:
                print(f"[WARNING] Stratified split failed for Client {client_id}, using random split")
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                )

            # Keys renamed X_test/y_test kept for backward compat with centralizer
            client_splits[client_id] = {
                "X_train": X_train.reset_index(drop=True),
                "X_test":  X_val.reset_index(drop=True),
                "y_train": y_train.reset_index(drop=True),
                "y_test":  y_val.reset_index(drop=True),
            }

        print(f"[INFO] Successfully split {len(client_splits)} clients")

        return client_splits
