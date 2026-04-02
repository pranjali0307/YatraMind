import pandas as pd

destinations = pd.read_csv("data/Expanded_Destinations.csv")
reviews = pd.read_csv("data/Final_Updated_Expanded_Reviews.csv")
users = pd.read_csv("data/Final_Updated_Expanded_Users.csv")
history = pd.read_csv("data/Final_Updated_Expanded_UserHistory.csv")

# Merge reviews with destinations
merged = pd.merge(
    reviews,
    destinations,
    on="DestinationID"
)

# Merge user preferences
merged = pd.merge(
    merged,
    users,
    on="UserID"
)

# Rename columns
merged.rename(columns={
    "Name_x": "DestinationName",
    "Name_y": "UserName"
}, inplace=True)

print(merged.head())
print(merged.columns)
merged.to_csv("data/merged_travel_data.csv", index=False)

print("Merged dataset saved successfully")