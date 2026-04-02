import pandas as pd
import os

def recommend(preference, state=None):

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(BASE_DIR, "data", "merged_travel_data.csv")

    data = pd.read_csv(file_path)

    print("Total rows in dataset:", len(data))
    print("Total unique destinations:", data["Name"].nunique())

    # Group by destination and take average popularity
    data = data.groupby(
        ["Name", "State", "Type", "BestTimeToVisit"],
        as_index=False
    ).mean(numeric_only=True)

    print("After grouping unique destinations:", len(data))

    print("\nAll Destinations:")
    print(data[["Name", "Type"]])

    # Filter by preference
    filtered = data[
        data["Type"].str.contains(preference, case=False, na=False)
    ]

    print("Filtered rows:", len(filtered))

    # Filter by state if provided
    if state:
        filtered = filtered[
            filtered["State"].str.contains(state, case=False, na=False)
        ]

    # Sort by popularity
    result = filtered.sort_values(
        by=["Popularity"],
        ascending=False
    )

    return result[
        ["Name", "State", "Type", "Popularity", "BestTimeToVisit"]
    ].head(5)


if __name__ == "__main__":
    result = recommend("Adventure")
    print("\nRecommended Destinations:\n")
    print(result)