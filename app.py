import streamlit as st
import pandas as pd
from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from collaborative_based_filtering import collaborative_filtering_recommendations
from hybrid_approach import hybrid_recommendation_filtering

# Config
st.set_page_config(page_title="AI Recommendation System", layout="wide")
st.title("🛍️ AI Recommendation System")

# Data Loading
@st.cache_data
def load_and_clean_data():
    raw_data = pd.read_csv("clean_data.csv")
    df = process_data(raw_data)

    if 'ImageURL' in df.columns:
        df['ImageURL'] = df['ImageURL'].astype(str).apply(
            lambda x: x.split('|')[0].strip()
        )

    return df

data = load_and_clean_data()
valid_user_ids = set(data['ID'].unique())

# Session State
if 'last_action' not in st.session_state:
    st.session_state.last_action = 'user'

if 'prev_user' not in st.session_state:
    st.session_state.prev_user = ""

if 'prev_search' not in st.session_state:
    st.session_state.prev_search = ""

# Sidebar - User ID
st.sidebar.header("👤 User Account")
user_id_input = st.sidebar.text_input(
    "Enter User ID",
    placeholder="e.g. 0, 5, 8"
)

# Search Bar
search_query = st.text_input(
    "🔍 Search for products...",
    placeholder="Shampoo, Cream, etc."
)

# Detect user / search change
if user_id_input != st.session_state.prev_user and user_id_input.strip() != "":
    st.session_state.last_action = 'user'
    st.session_state.prev_user = user_id_input

if search_query != st.session_state.prev_search and search_query.strip() != "":
    st.session_state.last_action = 'search'
    st.session_state.prev_search = search_query

# Resolve User
target_user = None
show_only_trending = False

if user_id_input.strip().isdigit():
    val = int(user_id_input)
    if val == 0:
        show_only_trending = True
    elif val in valid_user_ids:
        target_user = val

# Trending Products
trending_products = get_top_rated_items(data, 8)

# Seed Item for Hybrid
seed_item = None

if search_query.strip():
    search_res = data[data['Name'].str.contains(search_query, case=False, na=False)]
    if not search_res.empty:
        seed_item = search_res.iloc[0]['Name']
elif not trending_products.empty:
    seed_item = trending_products.iloc[0]['Name']

# Display Grid
def display_grid(df, title, max_items=8):
    st.subheader(title)

    if df is None or df.empty:
        st.info("No products found for this section.")
        return

    cols = st.columns(4)

    for i, (_, row) in enumerate(df.head(max_items).iterrows()):
        with cols[i % 4]:
            with st.container():
                img_url = row.get('ImageURL', None)

                # Image
                if img_url and img_url != "nan":
                    st.image(img_url, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300", use_container_width=True)

                # Fixed spacing to normalize height
                st.write("")

                # Product name
                name = row.get('Name', 'Product')
                st.markdown(
                    f"**{name[:45]}...**" if len(name) > 45 else f"**{name}**"
                )

                # Rating + Brand
                st.caption(
                    f"⭐ {round(float(row.get('Rating', 0)), 1)} | "
                    f"{row.get('Brand', 'Brand')}"
                )

                # Extra spacing to keep cards equal height
                st.write("")

# Interface Logic
if show_only_trending:
    display_grid(trending_products, "🔥 Trending Products")

elif st.session_state.last_action == 'search':
    search_results = data[data['Name'].str.contains(search_query, case=False, na=False)]
    display_grid(search_results, f"🔎 Search Results for '{search_query}'")
    st.divider()

    display_grid(trending_products, "🔥 Trending Products")
    st.divider()

    if seed_item:
        hybrid_recs = hybrid_recommendation_filtering(
            data,
            item_name=seed_item,
            target_user_id=target_user,
            top_n=8
        )
        display_grid(hybrid_recs, "🔄 Hybrid Recommendations (Based on Search)")

else:
    if target_user:
        user_recs = collaborative_filtering_recommendations(
            data,
            target_user,
            top_n=8
        )
        display_grid(user_recs, f"✨ Recommended for You (User {target_user})")
        st.divider()

    display_grid(trending_products, "🔥 Trending Products")
