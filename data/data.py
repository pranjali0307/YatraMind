import pandas as pd
import os


def load_data():
    """
    Load and clean travel dataset
    """

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(BASE_DIR, "data", "Top Indian Places to Visit.csv")

    # Load CSV
    data = pd.read_csv(file_path)

    # Remove extra spaces from column names
    data.columns = data.columns.str.strip()

    # Drop unwanted column if exists
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    # Convert numeric columns safely
    numeric_columns = [
        "Google review rating",
        "Number of google review in lakhs",
        "Entrance Fee in INR",
        "time needed to visit in hrs"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill missing values instead of dropping everything
    data["Google review rating"] = data["Google review rating"].fillna(
        data["Google review rating"].mean()
    )

    data["Number of google review in lakhs"] = data[
        "Number of google review in lakhs"
    ].fillna(0)

    # Drop rows where Name or Type missing
    data = data.dropna(subset=["Name", "Type", "State"])

    # Remove duplicates
    data = data.drop_duplicates(subset="Name")

    # Reset index
    data = data.reset_index(drop=True)

    return data


if __name__ == "__main__":

    data = load_data()

    print("\nDataset Loaded Successfully\n")

    print("Total rows:", len(data))
    print("Total unique places:", data["Name"].nunique())

    print("\nColumns:\n")
    print(data.columns)

    print("\nSample Data:\n")
    print(data.head())