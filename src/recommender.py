import streamlit as st
import pandas as pd

from src.recommend import get_recommendations
from src.svd_recommender import get_svd_recommendations


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .movie-title {
        font-size: 17px;
        font-weight: 600;
        margin-top: 8px;
        min-height: 50px;
    }

    .genre-text {
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load Movie Database
# --------------------------------------------------

movies = pd.read_csv(
    "data/movie_posters.csv"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 AI-Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover movies personalized to your interests</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# User Input Section
# --------------------------------------------------

st.subheader("👤 Find Your Recommendations")

user_id = st.number_input(
    "Enter MovieLens User ID",
    min_value=1,
    max_value=6040,
    value=1,
    step=1
)

method = st.selectbox(
    "Choose Recommendation Method",
    [
        "KNN / Collaborative Filtering",
        "SVD"
    ]
)


# --------------------------------------------------
# Recommendation Button
# --------------------------------------------------

if st.button(
    "🎯 Get My Recommendations",
    use_container_width=True
):

    with st.spinner(
        "Analyzing your movie preferences..."
    ):

        # KNN Recommendation
        if method == "KNN / Collaborative Filtering":

            recommendations = get_recommendations(
                user_id=user_id,
                n=10
            )

        # SVD Recommendation
        else:

            recommendations = get_svd_recommendations(
                user_id=user_id,
                n=10
            )


    # --------------------------------------------------
    # Check Recommendations
    # --------------------------------------------------

    if recommendations.empty:

        st.error(
            f"User ID {user_id} was not found in the MovieLens dataset."
        )

    else:

        # --------------------------------------------------
        # Add Poster URLs
        # --------------------------------------------------

        recommendations = recommendations.merge(
            movies[["movie_id", "poster_url"]],
            on="movie_id",
            how="left"
        )


        # --------------------------------------------------
        # Success Message
        # --------------------------------------------------

        st.success(
            "Your personalized recommendations are ready!"
        )


        # --------------------------------------------------
        # Method Information
        # --------------------------------------------------

        if method == "KNN / Collaborative Filtering":

            st.info(
                "Recommendation method: KNN / Item-Based "
                "Collaborative Filtering using Cosine Similarity."
            )

        else:

            st.info(
                "Recommendation method: SVD "
                "(Singular Value Decomposition)."
            )


        # --------------------------------------------------
        # Recommendation Heading
        # --------------------------------------------------

        st.subheader(
            f"🍿 Top 10 Movies for User {user_id}"
        )


        # --------------------------------------------------
        # Movie Cards
        # --------------------------------------------------

        for row_start in range(0, 10, 5):

            cols = st.columns(5)

            for i, col in enumerate(cols):

                index = row_start + i

                if index >= len(recommendations):
                    continue

                row = recommendations.iloc[index]

                with col:

                    # Poster
                    poster = row["poster_url"]

                    if pd.notna(poster) and poster != "":

                        st.image(
                            poster,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "Poster unavailable"
                        )


                    # Movie Title
                    st.markdown(
                        f'<div class="movie-title">{row["title"]}</div>',
                        unsafe_allow_html=True
                    )


                    # Genres
                    genres = row["genres"].replace(
                        "|",
                        " • "
                    )

                    st.markdown(
                        f'<div class="genre-text">{genres}</div>',
                        unsafe_allow_html=True
                    )


                    # SVD Predicted Rating
                    if (
                        method == "SVD"
                        and "PredictedRating" in recommendations.columns
                    ):

                        st.caption(
                            f"Predicted Rating: "
                            f"{row['PredictedRating']:.2f} / 5"
                        )


# --------------------------------------------------
# About Project
# --------------------------------------------------

st.divider()

st.subheader("🤖 How Does It Work?")

st.markdown(
    """
    **1. User Ratings**  
    The system uses movie ratings from the MovieLens 1M dataset.

    **2. KNN / Collaborative Filtering**  
    The system identifies similar movies based on user rating patterns
    using cosine similarity.

    **3. SVD**  
    Singular Value Decomposition learns hidden patterns in the
    user-movie rating matrix and predicts ratings for movies.

    **4. Personalized Recommendations**  
    Movies already rated by the user are excluded and the system
    selects the Top 10 recommended movies.

    **5. Movie Posters**  
    Poster information is obtained through the TMDB API.
    """
)


# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.divider()

st.subheader("📊 Project Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Movies",
        "3,883"
    )

with col2:

    st.metric(
        "Users",
        "6,040"
    )

with col3:

    st.metric(
        "Ratings",
        "1,000,209"
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Recommendation Methods: KNN / Item-Based Collaborative Filtering "
    "using Cosine Similarity and SVD."
)

st.caption(
    "Dataset: MovieLens 1M"
)

st.caption(
    "Movie poster information provided through the TMDB API."
)

st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)