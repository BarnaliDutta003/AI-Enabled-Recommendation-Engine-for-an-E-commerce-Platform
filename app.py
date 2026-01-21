import streamlit as st
import pandas as pd
from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from collaborative_based_filtering import collaborative_filtering_recommendations
from hybrid_approach import hybrid_recommendation_filtering

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Recommendation System", layout="wide")
st.title("🛍️ AI Recommendation System")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_and_clean_data():
    raw_data = pd.read_csv("clean_data.csv")
    df = process_data(raw_data)
    if 'ImageURL' in df.columns:
        df['ImageURL'] = df['ImageURL'].astype(str).apply(lambda x: x.split('|')[0].strip())
    return df

data = load_and_clean_data()
valid_user_ids = set(data['ID'].unique())

# ---------------- SESSION STATE MANAGEMENT ----------------
# We track the last action to decide which UI to show
if 'last_action' not in st.session_state:
    st.session_state.last_action = 'user'  # Default view

if 'prev_user' not in st.session_state:
    st.session_state.prev_user = ""

if 'prev_search' not in st.session_state:
    st.session_state.prev_search = ""

# ---------------- SIDEBAR (User ID) ----------------
st.sidebar.header("👤 User Account")
user_id_input = st.sidebar.text_input("Enter User ID", placeholder="e.g. 8, 5, 3", key="user_input_box")

# ---------------- MAIN SEARCH (Product Search) ----------------
search_query = st.text_input("🔍 Search for products...", placeholder="Shampoo, Cream, etc.", key="search_input_box")

# --- Detect Change in User ID ---
if user_id_input != st.session_state.prev_user and user_id_input.strip() != "":
    st.session_state.last_action = 'user'
    st.session_state.prev_user = user_id_input

# --- Detect Change in Search Bar ---
if search_query != st.session_state.prev_search and search_query.strip() != "":
    st.session_state.last_action = 'search'
    st.session_state.prev_search = search_query

# ---------------- LOGIC PREPARATION ----------------
# 1. Resolve User
target_user = None
if user_id_input.strip().isdigit():
    val = int(user_id_input)
    if val in valid_user_ids:
        target_user = val

# 2. Get Trending (Area 2 - Always needed)
trending_products = get_top_rated_items(data, 8)

# 3. Determine Hybrid Seed (Area 3 - Always needed)
seed_item = None
if search_query.strip():
    search_res = data[data['Name'].str.contains(search_query, case=False, na=False)]
    if not search_res.empty:
        seed_item = search_res.iloc[0]['Name']
elif not trending_products.empty:
    seed_item = trending_products.iloc[0]['Name']

# ---------------- UI HELPER ----------------
def display_grid(df, title, max_items=8):
    st.subheader(title)
    if df is None or df.empty:
        st.info("No products found for this section.")
        return
    
    cols = st.columns(4)
    for i, (_, row) in enumerate(df.head(max_items).iterrows()):
        with cols[i % 4]:
            img_url = row.get('ImageURL', "https://via.placeholder.com/300")
            st.image(img_url, use_container_width=True)
            name = row.get('Name', 'Product')
            st.markdown(f"**{name[:50]}...**" if len(name) > 50 else f"**{name}**")
            st.caption(f"⭐ {round(float(row.get('Rating', 0)), 1)} | {row.get('Brand', 'Brand')}")

# ---------------- THREE-AREA INTERFACE ----------------

if st.session_state.last_action == 'search':
    # SCENARIO A: LAST ACTION WAS SEARCH
    # Area 1: Search Results
    search_results = data[data['Name'].str.contains(search_query, case=False, na=False)]
    display_grid(search_results, f"🔎 Search Results for '{search_query}'")
    st.divider()

    # Area 2: Trending Products
    display_grid(trending_products, "🔥 Trending Products")
    st.divider()

    # Area 3: Hybrid Recommendations
    if seed_item:
        hybrid_recs = hybrid_recommendation_filtering(data, item_name=seed_item, target_user_id=target_user, top_n=8)
        display_grid(hybrid_recs, "🔄 Hybrid Recommendations (Based on Search)")

else:
    # SCENARIO B: LAST ACTION WAS USER LOGIN (or default)
    # Area 1: User Recommendations (Collaborative)
    if target_user:
        user_recs = collaborative_filtering_recommendations(data, target_user, top_n=8)
        display_grid(user_recs, f"✨ Recommended for You (User {target_user})")
    else:
        # If no user logged in, Area 1 shows top rated as a welcome
        display_grid(get_top_rated_items(data, 8), "✨ Recommended for You")
    st.divider()

    # Area 2: Trending Products
    display_grid(trending_products, "🔥 Trending Products")
    st.divider()

    # Area 3: Hybrid Recommendations
    if seed_item:
        hybrid_recs = hybrid_recommendation_filtering(data, item_name=seed_item, target_user_id=target_user, top_n=8)
        display_grid(hybrid_recs, "🔄 Hybrid Recommendations (For Your Taste)")