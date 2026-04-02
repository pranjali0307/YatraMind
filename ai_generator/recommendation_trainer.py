import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from data.data import load_data


def recommend(
    preference=None,
    state=None,
    city=None,
    best_time=None,
    min_rating=0
):

    data = load_data()
    filtered = data.copy()

    # Smart category mapping
    category_map = {
        "historical": [
            "Monument", "Fort", "Palace", "Tomb",
            "Memorial", "Historical", "Museum"
        ],
        "nature": [
            "Lake", "Waterfall", "Hill", "Valley",
            "National Park", "Scenic Area"
        ],
        "religious": [
            "Temple", "Mosque", "Church",
            "Shrine", "Gurudwara"
        ],
        "adventure": [
            "Trekking", "Ski Resort",
            "Mountain Peak", "Adventure Sport"
        ],
        "beach": ["Beach"],
        "city": ["Market", "Mall", "Commercial Complex"]
    }

    # Apply preference mapping
    if preference:
        preference = preference.lower()

        if preference in category_map:
            filtered = filtered[
                filtered["Type"].isin(category_map[preference])
            ]
        else:
            filtered = filtered[
                filtered["Type"].str.contains(
                    preference, case=False, na=False
                )
            ]

    if state:
        filtered = filtered[
            filtered["State"].str.contains(
                state, case=False, na=False
            )
        ]

    if city:
        filtered = filtered[
            filtered["City"].str.contains(
                city, case=False, na=False
            )
        ]

    if best_time:
        filtered = filtered[
            filtered["Best Time to visit"].str.contains(
                best_time, case=False, na=False
            )
        ]

    filtered = filtered[
        filtered["Google review rating"] >= min_rating
    ]

    # fallback if empty
    if filtered.empty:
        print("\nNo exact match found. Showing best destinations instead.\n")
        filtered = data

    filtered = filtered.sort_values(
        by=[
            "Google review rating",
            "Number of google review in lakhs"
        ],
        ascending=False
    )

    return filtered[
        [
            "Name",
            "City",
            "State",
            "Type",
            "Google review rating",
            "Number of google review in lakhs",
            "Best Time to visit",
            "Entrance Fee in INR"
        ]
    ].head(5)


if __name__ == "__main__":

    data = load_data()

    print("Available Types:\n")
    print(data["Type"].unique())

    print("\nAvailable States:\n")
    print(data["State"].unique())

    result = recommend(
        preference="Historical",
        state="Delhi",
        min_rating=3
    )

    print("\nRecommended Places:\n")
    print(result)