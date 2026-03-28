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
        # Combine datasets
        full_df = pd.concat([self.train_df, self.test_df], axis=0).reset_index(drop=True)

        # Group by user (Subject column)
        user_groups = full_df.groupby("Subject")

        client_datasets = {}

        for user_id, user_data in user_groups:
            client_datasets[int(user_id)] = user_data.reset_index(drop=True)

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
        client_splits = {}

        for client_id, data in client_datasets.items():
            X = data.drop(columns=["Activity", "Subject"])
            y = data["Activity"]

            # Edge case: very small datasets
            if len(data) < 10:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=y,
                random_state=42
            )

            client_splits[client_id] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }

        return client_splits

    def get_client_summary(self, client_datasets):
        """
        Generate summary stats for each client

        Returns:
            dict: summary info
        """
        summary = {}

        for client_id, data in client_datasets.items():
            summary[client_id] = {
                "num_samples": len(data),
                "num_activities": data["Activity"].nunique()
            }

        return summary