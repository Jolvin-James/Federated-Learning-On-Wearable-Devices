# src/data_loader.py
import os
import pandas as pd


class UCIHARDataLoader:
    def __init__(self, base_path: str):
        """
        Initializes dataset path

        Args:
            base_path (str): Path to UCI HAR Dataset folder
                             Example: "data/UCI_HAR"
        """
        self.base_path = base_path

        # Validate dataset path early
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Dataset path not found: {self.base_path}")

    def _load_file(self, file_path, delim_whitespace=True):
        """Generic file loader with validation"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        if delim_whitespace:
            return pd.read_csv(file_path, sep=r'\s+', header=None)
        else:
            return pd.read_csv(file_path, header=None)

    def load_train_data(self):
        """Loads training dataset"""
        train_path = os.path.join(self.base_path, "train")

        X_train = self._load_file(os.path.join(train_path, "X_train.txt"))
        y_train = self._load_file(os.path.join(train_path, "y_train.txt"), delim_whitespace=False)
        subject_train = self._load_file(os.path.join(train_path, "subject_train.txt"), delim_whitespace=False)

        return X_train, y_train, subject_train

    def load_test_data(self):
        """Loads test dataset"""
        test_path = os.path.join(self.base_path, "test")

        X_test = self._load_file(os.path.join(test_path, "X_test.txt"))
        y_test = self._load_file(os.path.join(test_path, "y_test.txt"), delim_whitespace=False)
        subject_test = self._load_file(os.path.join(test_path, "subject_test.txt"), delim_whitespace=False)

        return X_test, y_test, subject_test

    def load_features(self):
        """Loads feature names"""
        feature_path = os.path.join(self.base_path, "features.txt")
        features = self._load_file(feature_path, delim_whitespace=True)

        # Extract feature names (2nd column)
        return features[1].tolist()

    def load_activity_labels(self):
        """Loads activity labels mapping"""
        activity_path = os.path.join(self.base_path, "activity_labels.txt")
        activity_labels = self._load_file(activity_path, delim_whitespace=True)

        return dict(zip(activity_labels[0], activity_labels[1]))

    def load_full_dataset(self):
        """
        Loads and merges complete dataset (train + test)

        Returns:
            train_df, test_df (pandas DataFrames)
        """
        # Load raw data
        X_train, y_train, subject_train = self.load_train_data()
        X_test, y_test, subject_test = self.load_test_data()

        features = self.load_features()

        # Assign feature names
        X_train.columns = features
        X_test.columns = features

        # Merge datasets
        train_df = X_train.copy()
        train_df["Activity"] = y_train
        train_df["Subject"] = subject_train

        test_df = X_test.copy()
        test_df["Activity"] = y_test
        test_df["Subject"] = subject_test

        return train_df, test_df