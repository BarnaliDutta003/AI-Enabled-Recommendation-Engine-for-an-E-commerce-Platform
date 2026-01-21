# 🛍️ AI-Enabled Hybrid Recommendation System

This project is a **Hybrid Recommendation System** built using **Content-Based Filtering**, **Collaborative Filtering**, and **Rating-Based Recommendations**, deployed through an interactive **Streamlit web application**.

It provides **personalized product recommendations** based on user behavior, product similarity, and overall popularity trends.

---

## 🔥 Features

- 🔍 Product Search  
- 👤 User-based Personalized Recommendations  
- 🧠 Content-Based Recommendations 
- 👥 Collaborative Filtering  
- 🔀 Hybrid Recommendation Engine  
- ⭐ Rating-Based Trending Products  
- 🖼️ Product Image Display  
- ⚡ Cached Computations for Speed  
- 🎨 Clean and Interactive Streamlit UI  

---

## 🧠 Recommendation Techniques Used

### 1. Content-Based Filtering
Recommends items similar to the selected product using:
- TF-IDF Vectorization
- Cosine Similarity
- Product tags and descriptions


### 2. Collaborative Filtering
Recommends items based on similar users by:
- Creating a user–item rating matrix
- Computing cosine similarity between users
- Weighted rating aggregation



### 3. Rating-Based Recommendation (Trending System)
Recommends globally popular products based on:
- Highest average ratings
- Overall user feedback

Used for:
- Cold-start users
- Trending section
- Homepage recommendations



### 4. Hybrid Recommendation
Combines:
- Content-based recommendations
- Collaborative filtering results

This improves:
- Accuracy
- Cold start handling
- Diversity of recommendations



