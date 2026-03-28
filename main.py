from src.data_loader import UCIHARDataLoader

def main():
    # Initialize loader
    data_path = "data/UCI_HAR"  # Change if needed
    loader = UCIHARDataLoader(data_path)

    # Load dataset
    train_df, test_df = loader.load_full_dataset()

    # 🔍 Verification checks
    print("\n--- DATASET LOADED SUCCESSFULLY ---\n")

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    print("\nUnique Subjects (Train):", train_df["Subject"].nunique())
    print("Unique Activities:", sorted(train_df["Activity"].unique()))

    print("\nSample Data:")
    print(train_df.head())


if __name__ == "__main__":
    main()