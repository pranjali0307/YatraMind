import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data import load_places

print("Loading dataset...")
places, df = load_places()

print("Loading AI model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Creating embeddings...")

texts = [place["combined"] for place in places]

embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Saving index...")

faiss.write_index(index, "data/places.index")

print("Saving places...")

with open("data/places.pkl", "wb") as f:
    pickle.dump(places, f)

print("Model Training Completed")