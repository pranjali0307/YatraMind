import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="YatraMind AI",
    page_icon="✈️",
    layout="wide"
)

# 2. ENHANCED CSS STYLING (Professional Overhaul)
st.markdown("""
<style>
/* Main background with a subtle professional gradient */
.stApp {
    background: linear-gradient(135deg, #f8faff 0%, #dbeafe 100%);
}

/* Header Styling */
.main-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(to right, #1e40af, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
}

.sub-title {
    font-size: 20px;
    text-align: center;
    color: #475569;
    font-weight: 500;
    margin-bottom: 40px;
}

/* Button - Using a vibrant "Action" Blue */
.stButton>button {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    color: white !important;
    font-size: 20px;
    font-weight: 600;
    border-radius: 12px;
    padding: 12px 40px;
    border: none;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    transition: all 0.3s ease;
    display: block;
    margin: 0 auto;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    background: #1e40af;
}

/* Card Styling - Modern 'Glassmorphism' feel */
.card {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 16px;
    border-left: 8px solid #2563eb;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}

.card:hover {
    transform: scale(1.01);
}

.card h4 {
    color: #1e293b;
    margin-top: 0;
    font-size: 22px;
    font-weight: 700;
}

/* Tags for visual hierarchy */
.tag-state {
    background-color: #dbeafe;
    color: #1e40af;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
    margin-right: 8px;
}

.tag-category {
    background-color: #fef3c7;
    color: #92400e;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

/* Day Header */
.day-header {
    color: #1e3a8a;
    font-size: 28px;
    font-weight: 700;
    margin-top: 35px;
    border-bottom: 2px solid #bfdbfe;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 3. APP HEADER
st.markdown('<p class="main-title">✈️ YatraMind - AI Travel Planner</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Smart AI Powered Travel Itinerary Generator</p>', unsafe_allow_html=True)

# 4. DATA & MODEL LOADING
@st.cache_resource
def load_model():
    # Using the exact model referenced in the tech stack [cite: 87-88]
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "data", "places.index")
    pkl_path = os.path.join(base_dir, "data", "places.pkl")

    # Load FAISS index and the associated metadata
    index = faiss.read_index(index_path)
    with open(pkl_path, "rb") as f:
        places = pickle.load(f)
    return index, places

model = load_model()
index, places = load_data()

# 5. DYNAMIC UI INPUTS
# Extract unique values for selectors
states = sorted(list(set([p["State"].strip() for p in places if p["State"]])))
categories = sorted(list(set([p["Category"].strip() for p in places if p["Category"]])))

col1, col2, col3 = st.columns(3)

with col1:
    state_input = st.selectbox("Select State", states)

with col2:
    category_input = st.selectbox("Select Category", categories)

with col3:
    # Default duration set to 3 as seen in demo
    days = st.slider("Trip Duration (Days)", 1, 7, 3)

st.write("")

# 6. ITINERARY GENERATION LOGIC
if st.button("Generate Itinerary 🚀"):
    # Semantic Search Query
    query = f"{state_input} {category_input}"
    query_embedding = np.array(model.encode([query])).astype("float32")

    # Fetch top matches to filter later
    distances, indices = index.search(query_embedding, k=30)

    results = []
    seen_names = set()

    for i in indices[0]:
        place = places[i]
        place_name = place['Name'].lower().strip()
        
        # Filter for the correct state and basic category match
        if place["State"] == state_input and category_input.lower() in place["Category"].lower():
            if place_name not in seen_names:
                results.append(place)
                seen_names.add(place_name)

    st.write("")
    st.markdown('<p class="day-header">📍 Your Personalized Plan</p>', unsafe_allow_html=True)

    if not results:
        st.warning("No unique places found for this selection. Try broadening your category!")
    else:
        # Build Day-by-Day View [cite: 45-46, 57-58]
        for day in range(days):
            st.markdown(f'<p class="day-header">🗓️ Day {day+1}</p>', unsafe_allow_html=True)
            
            # Show 2 activities per day if available
            start_idx = day * 2
            end_idx = start_idx + 2
            day_activities = results[start_idx:end_idx]

            if not day_activities:
                st.info("No more unique attractions found for this day.")
                break

            for place in day_activities:
                st.markdown(f"""
                <div class="card">
                    <h4>📍 {place['Name']}</h4>
                    <span class="tag-state">🗺️ {place['State']}</span>
                    <span class="tag-category">🏷️ {place['Category']}</span>
                </div>
                """, unsafe_allow_html=True)