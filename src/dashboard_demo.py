from dataclasses import dataclass

import pandas as pd


@dataclass
class DashboardDemoSplit:
    training_clients: dict
    new_client_id: int
    new_client_data: pd.DataFrame
    all_subjects: list
    client_sample_counts: dict

    @property
    def total_subjects(self):
        return len(self.all_subjects)

    @property
    def training_client_count(self):
        return len(self.training_clients)


def build_dashboard_demo_split(train_df, test_df, holdout_client=None):
    """
    Build the examiner-facing demo split.

    The research pipeline keeps UCI HAR's official train/test split intact.
    This helper is only for the dashboard story: all 30 subjects are visible,
    29 become federated clients, and one subject is reserved as the new client.
    """
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    subjects = sorted(int(subject) for subject in full_df["Subject"].unique())

    if not subjects:
        raise ValueError("No subjects found for dashboard demo split")

    new_client_id = int(holdout_client) if holdout_client is not None else subjects[-1]
    if new_client_id not in subjects:
        raise ValueError(f"Holdout client {new_client_id} is not present in dataset")

    training_clients = {}
    client_sample_counts = {}
    new_client_data = None

    for subject_id, subject_df in full_df.groupby("Subject"):
        subject_id = int(subject_id)
        clean_df = subject_df.reset_index(drop=True)

        if subject_id == new_client_id:
            new_client_data = clean_df
            continue

        training_clients[subject_id] = clean_df
        client_sample_counts[subject_id] = int(len(clean_df))

    if new_client_data is None:
        raise ValueError(f"Could not reserve new client {new_client_id}")

    return DashboardDemoSplit(
        training_clients=training_clients,
        new_client_id=new_client_id,
        new_client_data=new_client_data,
        all_subjects=subjects,
        client_sample_counts=client_sample_counts,
    )


def describe_new_client(new_client_id, new_client_data):
    activity_counts = (
        new_client_data["Activity"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "client_id": int(new_client_id),
        "samples": int(len(new_client_data)),
        "activity_counts": {str(k): int(v) for k, v in activity_counts.items()},
    }
