import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def get_user_item_matrix(data):
    return data.pivot_table(
        index='ID',
        columns='ProdID',
        values='Rating',
        aggfunc='mean'
    ).fillna(0)

@st.cache_data
def get_similarity_matrix(matrix):
    return cosine_similarity(matrix)

def collaborative_filtering_recommendations(data, target_user_id, top_n=10):
    user_item_matrix = get_user_item_matrix(data)

    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame()

    # Check if user has actually rated anything
    if user_item_matrix.loc[target_user_id].sum() == 0:
        return pd.DataFrame()

    # Get pre-calculated similarity
    similarity_matrix = get_similarity_matrix(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get top 5 similar users (excluding self)
    sim_scores = similarity_df[target_user_id].sort_values(ascending=False)
    similar_users = sim_scores[sim_scores.index != target_user_id].head(5)

    # Filter out users with 0 similarity
    similar_users = similar_users[similar_users > 0]
    if similar_users.empty:
        return pd.DataFrame()

    target_user_rated = user_item_matrix.loc[target_user_id]
    already_rated = target_user_rated[target_user_rated > 0].index.tolist()

    scores = {}
    for sim_user, sim_score in similar_users.items():
        sim_user_ratings = user_item_matrix.loc[sim_user]
        for prod_id, rating in sim_user_ratings.items():
            if prod_id not in already_rated and rating > 0:
                scores[prod_id] = scores.get(prod_id, 0) + sim_score * rating

    if not scores:
        return pd.DataFrame()

    ranked_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_prod_ids = [prod_id for prod_id, _ in ranked_products[:top_n]]

    # Get product details
    recommended_products = data[data['ProdID'].isin(recommended_prod_ids)][
        ['ProdID', 'Name', 'Brand', 'Rating', 'ImageURL']
    ].drop_duplicates(subset=['ProdID'])

    return recommended_products