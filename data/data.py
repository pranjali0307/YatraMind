import pandas as pd

destinations = pd.read_csv("data/Expanded_Destinations.csv")
reviews = pd.read_csv("data/Final_Updated_Expanded_Reviews.csv")
users = pd.read_csv("data/Final_Updated_Expanded_Users.csv")
history = pd.read_csv("data/Final_Updated_Expanded_UserHistory.csv")

print("Destinations")
print(destinations.head())

print("\nReviews")
print(reviews.head())

print("\nUsers")
print(users.head())

print("\nHistory")
print(history.head())

print(destinations.columns)
print(reviews.columns)
print(users.columns)
print(history.columns)