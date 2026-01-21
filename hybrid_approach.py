import pandas as pd
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations

def hybrid_recommendation_filtering(data, item_name=None, target_user_id=None, top_n=10):
    hybrid_rows = pd.DataFrame()

    # 1. Content-Based
    if item_name:
        cb_rec = content_based_recommendation(data, item_name, top_n)
        if not cb_rec.empty:
            cb_full = data[data['Name'].isin(cb_rec['Name'])]
            hybrid_rows = pd.concat([hybrid_rows, cb_full], ignore_index=True)

    # 2. Collaborative
    if target_user_id is not None:
        collab_rec = collaborative_filtering_recommendations(data, target_user_id, top_n)
        if not collab_rec.empty: # FIX: Use .empty
            hybrid_rows = pd.concat([hybrid_rows, collab_rec], ignore_index=True)

    if hybrid_rows.empty:
        return pd.DataFrame()

    hybrid_rows = hybrid_rows.drop_duplicates(subset=['Name']).head(top_n * 2)
    
    if 'ImageURL' in hybrid_rows.columns:
        hybrid_rows['ImageURL'] = hybrid_rows['ImageURL'].fillna("https://via.placeholder.com/300")
    
    return hybrid_rows