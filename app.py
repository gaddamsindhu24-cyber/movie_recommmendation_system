import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# SAFE MODEL IMPORTS (Graceful fallback if run outside WSL)
# ============================================================

model_import_errors = {}

try:
    from src.recommend import get_recommendations
except Exception as e:
    get_recommendations = None
    model_import_errors["KNN"] = str(e)

try:
    from src.svd_recommender import get_svd_recommendations
except Exception as e:
    get_svd_recommendations = None
    model_import_errors["SVD"] = str(e)

try:
    from src.cnn_recommender import get_cnn_recommendations
except Exception as e:
    get_cnn_recommendations = None
    model_import_errors["CNN"] = str(e)

try:
    from src.user_storage import (
        add_to_watchlist,
        remove_from_watchlist,
        is_in_watchlist,
        save_feedback,
        get_watchlist
    )
except ImportError:
    import importlib
    import src.user_storage
    importlib.reload(src.user_storage)
    from src.user_storage import (
        add_to_watchlist,
        remove_from_watchlist,
        is_in_watchlist,
        save_feedback,
        get_watchlist
    )

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #aaaaaa;
        margin-bottom: 24px;
    }

    .movie-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
    }

    .movie-title {
        font-size: 15px;
        font-weight: 600;
        margin-top: 8px;
        min-height: 44px;
        line-height: 1.3;
    }

    .genre-text {
        font-size: 12px;
        color: #999999;
        margin-bottom: 6px;
    }

    .score-badge {
        font-size: 13px;
        font-weight: 600;
        color: #ffbb00;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MOVIE DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_movie_posters():
    return pd.read_csv(BASE_DIR / "data/movie_posters.csv")

movies = load_movie_posters()

# Environment notice if run outside WSL
if model_import_errors:
    st.warning(
        """
        ⚠️ **Running in Windows Host Python instead of Ubuntu WSL**  
        The machine learning libraries (`scikit-learn`, `TensorFlow`, `scikit-surprise`) are installed inside your **Ubuntu WSL virtual environment**.  
        
        **To run with all models active:**  
        - Type `wsl` in your terminal, then run:  
          `cd ~/Projects/Movie_Recommendation_system && source venv/bin/activate && streamlit run app.py`  
        - Or in Windows PowerShell, simply run: `.\\run.bat`
        """
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 AI Movie Recommender")
st.sidebar.info(
    """
    **Personalized Recommendation Engine**  
    Powered by 3 distinct ML architectures:
    - **KNN**: Collaborative filtering via item cosine similarity
    - **SVD**: Latent factor matrix factorization
    - **CNN**: Deep interaction neural network
    """
)

# Synchronize User ID across pages
default_user = st.session_state.get("user_id", 1)

# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 AI-Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover movies personalized to your taste using state-of-the-art ML models</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# USER INPUT SECTION
# ============================================================

st.subheader("👤 Find Your Recommendations")

col_u, col_m, col_c = st.columns([1.5, 2, 2])

with col_u:
    user_id = st.number_input(
        "Enter MovieLens User ID",
        min_value=1,
        max_value=6040,
        value=int(default_user),
        step=1,
        key="main_user_id_input"
    )
    st.session_state["user_id"] = user_id

with col_m:
    method = st.selectbox(
        "🤖 Choose Recommendation Model",
        [
            "KNN / Collaborative Filtering",
            "SVD",
            "CNN"
        ],
        index=0,
        key="main_model_select"
    )

with col_c:
    category_list = [
        "All Genres", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    selected_category = st.selectbox(
        "🎭 Choose Movie Category",
        category_list,
        index=0,
        key="main_category_select"
    )

# ============================================================
# RECOMMENDATION TRIGGER
# ============================================================

if st.button(
    "🎯 Get My Recommendations",
    use_container_width=True,
    type="primary"
):
    with st.spinner("🤖 AI is analyzing your movie preferences..."):
        try:
            if method == "KNN / Collaborative Filtering":
                if get_recommendations is None:
                    st.error(f"KNN dependencies are missing in this environment ({model_import_errors.get('KNN')}). Please run the app in Ubuntu WSL using '.\\run.bat'.")
                    st.stop()
                recommendations = get_recommendations(
                    user_id=user_id,
                    n=10,
                    genre=selected_category
                )
            elif method == "SVD":
                if get_svd_recommendations is None:
                    st.error(f"SVD dependencies are missing in this environment ({model_import_errors.get('SVD')}). Please run the app in Ubuntu WSL using '.\\run.bat'.")
                    st.stop()
                recommendations = get_svd_recommendations(
                    user_id=user_id,
                    n=10,
                    genre=selected_category
                )
            else:
                if get_cnn_recommendations is None:
                    st.error(f"CNN dependencies are missing in this environment ({model_import_errors.get('CNN')}). Please run the app in Ubuntu WSL using '.\\run.bat'.")
                    st.stop()
                recommendations = get_cnn_recommendations(
                    user_id=user_id,
                    n=10,
                    genre=selected_category
                )

            if recommendations.empty:
                st.warning(
                    f"No eligible {selected_category} recommendations found for User ID {user_id} using {method}."
                )
                st.session_state["recommendations"] = None
            else:
                # Merge with TMDB posters
                recommendations = recommendations.merge(
                    movies[["movie_id", "poster_url"]],
                    on="movie_id",
                    how="left"
                )
                st.session_state["recommendations"] = recommendations
                st.session_state["rec_method"] = method
                st.session_state["rec_category"] = selected_category
                st.session_state["rec_user_id"] = user_id
                st.success("Your personalized recommendations are ready!")

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.session_state["recommendations"] = None

# ============================================================
# DISPLAY STORED RECOMMENDATIONS
# ============================================================

saved_recs = st.session_state.get("recommendations")

if saved_recs is not None and not saved_recs.empty:
    rec_user = st.session_state.get("rec_user_id", user_id)
    rec_cat = st.session_state.get("rec_category", selected_category)
    rec_mth = st.session_state.get("rec_method", method)

    # Informational headers
    if rec_cat == "All Genres":
        st.subheader(f"🍿 Top 10 Recommendations for User {rec_user}")
    else:
        st.subheader(f"🍿 Top 10 {rec_cat} Movies for User {rec_user}")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"**Model Active:** {rec_mth}")
    with col_info2:
        st.info(f"**Category Filter:** {rec_cat}")

    # Display Movie Cards in rows of 5
    for row_start in range(0, len(saved_recs), 5):
        cols = st.columns(5)

        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(saved_recs):
                continue

            row = saved_recs.iloc[idx]
            mid = int(row["movie_id"])

            with col:
                poster = row.get("poster_url")
                if pd.notna(poster) and poster != "":
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown("🎬 *Poster unavailable*")

                st.markdown(
                    f'<div class="movie-title">{row["title"]}</div>',
                    unsafe_allow_html=True
                )

                genres = str(row.get("genres", "")).replace("|", " • ")
                st.markdown(
                    f'<div class="genre-text">{genres}</div>',
                    unsafe_allow_html=True
                )

                # Score / Rating Badge
                if "PredictedRating" in row and pd.notna(row["PredictedRating"]):
                    st.markdown(
                        f'<div class="score-badge">⭐ Predicted: {row["PredictedRating"]:.2f} / 5</div>',
                        unsafe_allow_html=True
                    )
                elif "recommendation_score" in row and pd.notna(row["recommendation_score"]):
                    st.markdown(
                        f'<div class="score-badge">📊 Similarity: {row["recommendation_score"]:.2f}</div>',
                        unsafe_allow_html=True
                    )

                # --------------------------------------------
                # Watchlist Action Button
                # --------------------------------------------
                already_in_watchlist = is_in_watchlist(rec_user, mid)

                if already_in_watchlist:
                    if st.button(
                        "🗑️ In Watchlist (Remove)",
                        key=f"card_rm_wl_{rec_user}_{mid}",
                        use_container_width=True
                    ):
                        try:
                            remove_from_watchlist(rec_user, mid)
                            st.toast(f"Removed '{row['title']}' from your watchlist.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not remove: {e}")
                else:
                    if st.button(
                        "➕ Add to Watchlist",
                        key=f"card_add_wl_{rec_user}_{mid}",
                        use_container_width=True
                    ):
                        try:
                            movie_dict = {
                                "movie_id": mid,
                                "title": str(row["title"]),
                                "genres": str(row.get("genres", "")),
                                "poster_url": str(row.get("poster_url", "") or "")
                            }
                            add_to_watchlist(rec_user, movie_dict)
                            st.toast(f"✅ Added '{row['title']}' to your watchlist!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not save: {e}")

                # --------------------------------------------
                # Feedback Expandable Form
                # --------------------------------------------
                with st.expander("⭐ Rate / Review"):
                    fb_star = st.selectbox(
                        "Stars",
                        ["No Rating", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
                        key=f"fb_star_{rec_user}_{mid}"
                    )
                    fb_sentiment = st.radio(
                        "Choice",
                        ["None", "Like", "Dislike"],
                        horizontal=True,
                        key=f"fb_senti_{rec_user}_{mid}"
                    )
                    fb_comment = st.text_input(
                        "Comment",
                        placeholder="Your thoughts...",
                        key=f"fb_comm_{rec_user}_{mid}"
                    )
                    if st.button(
                        "Submit Feedback",
                        key=f"fb_sub_{rec_user}_{mid}",
                        use_container_width=True
                    ):
                        rating_num = int(fb_star[0]) if fb_star != "No Rating" else None
                        senti_val = fb_sentiment if fb_sentiment in ["Like", "Dislike"] else None
                        comm_val = fb_comment.strip() if fb_comment else None

                        if rating_num is None and senti_val is None and not comm_val:
                            st.warning("Please provide a rating, sentiment, or comment.")
                        else:
                            try:
                                save_feedback(
                                    user_id=rec_user,
                                    movie={"movie_id": mid, "title": row["title"]},
                                    feedback=senti_val,
                                    rating=rating_num,
                                    comments=comm_val
                                )
                                st.toast(f"Feedback submitted for '{row['title']}'!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to save feedback: {e}")

# ============================================================
# HOW DOES IT WORK?
# ============================================================

st.divider()

st.subheader("🤖 How Does It Work?")

st.markdown(
    """
    **1. User Identification**  
    The user enters their MovieLens User ID (1 to 6040). Watched movies are dynamically identified and excluded from recommendations.

    **2. Multi-Model Architecture**  
    - **KNN (Item-Based Collaborative Filtering)**: Discovers items similar to movies the user rated highly (ratings ≥ 4) via item-item Cosine Similarity.
    - **SVD (Singular Value Decomposition)**: Matrix factorization technique that projects users and items into a 50-dimensional latent factor space to predict unobserved ratings.
    - **CNN (Convolutional Neural Network)**: Deep learning model that constructs outer interaction matrices between user and movie embeddings, applying 2D convolutions and dense layers.

    **3. Candidate Pool & Genre Pre-Filtering**  
    Candidate movies are filtered to match the requested genre before ranking, guaranteeing up to 10 top-rated eligible recommendations even for niche genres.

    **4. Real-Time Watchlist & Feedback System**  
    Users can bookmark movies to their persistent SQLite watchlist and submit multi-signal feedback (star ratings, sentiment, and written commentary) with automatic conflict resolution.

    **5. TMDB Movie Posters**  
    Poster metadata is fetched via The Movie Database (TMDB) API and stored locally for ultra-fast, responsive card presentation.
    """
)

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📊 Dataset & Project Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Movies", "3,883")

with col2:
    st.metric("Total Users", "6,040")

with col3:
    st.metric("Total Ratings", "1,000,209")

# ============================================================
# FOOTER & ATTRIBUTION
# ============================================================

st.divider()

st.caption(
    "AI-Powered Movie Recommendation System • College Project • "
    "Dataset: MovieLens 1M • Powered by Python, Streamlit, Scikit-Learn, Scikit-Surprise, and TensorFlow/Keras."
)

st.caption(
    "Movie poster information is provided through the TMDB API. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
