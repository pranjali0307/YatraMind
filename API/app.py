import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="YatraMind AI",
    layout="wide"
)

# ---------------------------
# Background Function (Translucent 0.4)
# ---------------------------

def set_bg(image):

    st.markdown(
        f"""
        <style>

        .stApp {{
        background: linear-gradient(
            rgba(0,0,0,0.4),
            rgba(0,0,0,0.4)
        ),
        url("{image}");
        
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        }}

        .card {{
        background: rgba(0,0,0,0.35);
        backdrop-filter: blur(12px);
        padding:20px;
        border-radius:15px;
        margin-bottom:20px;
        transition:0.3s;
        }}

        .card:hover {{
        transform: translateY(-5px);
        background: rgba(0,0,0,0.5);
        }}

        .big-title {{
        font-size:60px;
        font-weight:800;
        color:white;
        }}

        .subtitle {{
        font-size:20px;
        color:#e5e7eb;
        }}

        .stButton>button {{
        background: linear-gradient(90deg,#ff4b4b,#ff9f43);
        color:white;
        border-radius:12px;
        padding:12px 25px;
        font-weight:600;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------------------
# Navigation
# ---------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


# ---------------------------
# Load Model
# ---------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_data():

    index = faiss.read_index("data/places.index")

    with open("data/places.pkl", "rb") as f:
        places = pickle.load(f)

    return index, places


model = load_model()
index, places = load_data()


# ---------------------------
# Category Cleaner
# ---------------------------

def clean_category(cat):

    cat = str(cat).lower()

    if any(word in cat for word in [
        "fort","palace","monument","museum",
        "heritage","historical","tomb"
    ]):
        return "Historical"

    elif any(word in cat for word in [
        "temple","mosque","church","gurudwara"
    ]):
        return "Religious"

    elif any(word in cat for word in [
        "wildlife","sanctuary","national park","zoo"
    ]):
        return "Wildlife"

    elif any(word in cat for word in [
        "lake","waterfall","garden","park","nature"
    ]):
        return "Nature"

    elif any(word in cat for word in [
        "trek","rafting","paragliding",
        "ski","camp","hiking","adventure"
    ]):
        return "Adventure"

    elif "beach" in cat:
        return "Beach"

    elif any(word in cat for word in [
        "hill","mountain"
    ]):
        return "Hill Station"

    else:
        return "Tourist"


# ---------------------------
# Rating
# ---------------------------

def get_rating(place):

    try:
        return float(place.get("Google review rating",4))
    except:
        return 4


# ---------------------------
# Homepage
# ---------------------------

if st.session_state.page == "home":

    set_bg("https://images.unsplash.com/photo-1501785888041-af3ef285b470")

    col1,col2 = st.columns([2,1])

    with col1:

        st.markdown(
        """
        <div class="big-title">
        🌍 YatraMind AI
        </div>

        <div class="subtitle">
        AI Powered Smart Travel Planner
        </div>
        """,
        unsafe_allow_html=True
        )

        st.write("")

        st.markdown("""
### ✨ Plan Smarter

• AI Itinerary  
• Budget Planning  
• Multi-Day Trips  
• Smart Categories  
""")

        if st.button("🚀 Start Planning"):
            st.session_state.page="planner"
            st.rerun()


# ---------------------------
# Planner Page
# ---------------------------

elif st.session_state.page=="planner":

    set_bg("https://images.unsplash.com/photo-1469854523086-cc02fe5d8800")

    st.title("🧭 Plan Your Trip")

    states = sorted(list(set([p["State"] for p in places])))

    categories = [
        "Historical",
        "Religious",
        "Wildlife",
        "Nature",
        "Adventure",
        "Beach",
        "Hill Station"
    ]

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        state = st.selectbox("State",states)

    with col2:
        category = st.selectbox("Category",categories)

    with col3:
        days = st.slider("Days",1,7,3)

    with col4:
        budget = st.selectbox("Budget",["Low","Medium","High"])

    st.write("")

    if st.button("✨ Generate Smart Trip"):

        st.session_state.state = state
        st.session_state.category = category
        st.session_state.days = days
        st.session_state.budget = budget

        st.session_state.page="results"
        st.rerun()

    if st.button("⬅ Back"):
        st.session_state.page="home"
        st.rerun()


# ---------------------------
# Results Page
# ---------------------------

elif st.session_state.page=="results":

    set_bg("https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1")

    st.title("🗺️ Your Smart Travel Plan")

    state = st.session_state.state
    category = st.session_state.category
    days = st.session_state.days

    query = f"{state} {category}"

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k=200)

    filtered=[]

    for i in indices[0]:

        place = places[i]

        place_state = str(place.get("State","")).lower()
        place_category = clean_category(place.get("Category",""))

        if state.lower() in place_state:

            if place_category == category:
                filtered.append(place)


    # fallback if empty
    if len(filtered) == 0:

        for i in indices[0]:

            place = places[i]

            if state.lower() in place["State"].lower():
                filtered.append(place)


    filtered = sorted(filtered,key=get_rating,reverse=True)

    places_per_day = max(1,len(filtered)//days)

    current=0

    for day in range(days):

        st.subheader(f"📅 Day {day+1}")

        cols = st.columns(3)

        for i in range(3):

            if current >= len(filtered):
                break

            place = filtered[current]

            with cols[i]:

                st.markdown(f"""
<div class="card">

### 📍 {place['Name']}

🏙️ {place['State']}  
🏷️ {clean_category(place['Category'])}  
⭐ {place.get('Google review rating','4')}  
💰 {place.get('Entrance Fee in INR','Free')}

</div>
""",unsafe_allow_html=True)

            current+=1

    st.write("")

    if st.button("🔄 Plan Another Trip"):
        st.session_state.page="planner"
        st.rerun()