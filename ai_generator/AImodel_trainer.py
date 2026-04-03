import sys
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.data import load_data


data = load_data()

model = SentenceTransformer('all-MiniLM-L6-v2')


data["combined"] = (
    data["Name"] + " " +
    data["City"] + " " +
    data["State"] + " " +
    data["Type"] + " " +
    data["Significance"]
)


embeddings = model.encode(data["combined"].tolist())
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def generate_itinerary(state, category, days):

    # Filter dataset first
    filtered = data[
    (data["State"].str.contains(state, case=False)) &
    (data["Type"].str.contains(category, case=False))
]

    if filtered.empty:

        print(f"\nNo '{category}' places found in {state}")

        available = data[
            data["State"].str.contains(state, case=False)
        ]["Type"].unique()

        print("\nAvailable categories in this state:")
        print(available)

        print("\nUsing AI similarity...\n")

        filtered = data[
            data["State"].str.contains(state, case=False)
        ]

    # Create embeddings for filtered data
    filtered_embeddings = model.encode(filtered["combined"].tolist())
    filtered_embeddings = np.array(filtered_embeddings).astype("float32")

    # Build temporary index
    dimension = filtered_embeddings.shape[1]
    temp_index = faiss.IndexFlatL2(dimension)
    temp_index.add(filtered_embeddings)

    query = f"{category} places in {state}"

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = temp_index.search(query_embedding, days * 3)

    places = filtered.iloc[indices[0]]

    print(f"\n📍 {state} | {category} | {days} Days\n")

    day = 1
    count = 0

    for i, row in places.iterrows():

        if count % 2 == 0:
            print(f"\nDay {day}:")

        print(f"- {row['Name']} ({row['City']})")

        count += 1

        if count % 2 == 0:
            day += 1

        if day > days:
            break


if __name__ == "__main__":

    state = input("Enter State: ")
    category = input("Enter Category: ")
    days = int(input("Enter Duration (days): "))

    generate_itinerary(state, category, days)