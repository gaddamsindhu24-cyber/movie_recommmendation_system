import streamlit as st
import pandas as pd

from src.recommend import get_recommendations
from src.svd_recommender import get_svd_recommendations
from src.cnn_recommender import get_cnn_recommendations
from src.hybrid_svd_recommender import get_hybrid_recommendations
from mood_recommendations import (
    get_mood_recommendations,
    add_recommendation_explanations
)

from src.user_storage import (
    add_to_watchlist,
    remove_from_watchlist,
    is_in_watchlist,
    save_feedback,
    get_watchlist,
    create_new_user_profile,
    get_new_user_profiles,
    get_new_user_profile,
    update_new_user_genres,
    save_new_user_feedback,
    get_new_user_feedback,
    DB_PATH,
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
        margin-bottom: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MOVIE DATA
# ============================================================

movies = pd.read_csv(
    "data/movie_posters.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 AI Movie Recommender")

st.sidebar.info(
    """
    Personalized movie recommendations
    using KNN, SVD and CNN.
    """
)


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 AI-Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover movies personalized to your interests</div>',
    unsafe_allow_html=True
)

st.divider()



# ============================================================
# USER INPUT
# ============================================================

st.subheader("👤 Find Your Recommendations")

user_type = st.radio(
    "Choose User Type",
    ["Existing MovieLens User", "New User"],
    horizontal=True
)

category_list = [
    "All Genres", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

if user_type == "Existing MovieLens User":

    default_user_id = st.session_state.get("user_id", 1)

    user_id = st.number_input(
        "Enter MovieLens User ID",
        min_value=1,
        max_value=6040,
        value=int(default_user_id),
        step=1
    )
    st.session_state["user_id"] = user_id

    method = st.selectbox(
        "🤖 Choose Recommendation Model",
        [
            "KNN / Collaborative Filtering",
            "SVD",
            "Hybrid SVD",
            "CNN"
        ]
    )

    selected_category = st.selectbox(
        "🎭 Choose Movie Category",
        category_list
    )

    selected_mood = st.selectbox(
        "😊 Select Your Current Mood",
        ["Happy 😊", "Sad 😢", "Romantic ❤️", "Excited 🤩", "Bored 😴"]
    )

    use_mood = st.checkbox("Enable Mood-Based Recommendations")

else:

    st.info(
        "New users do not need a MovieLens ID. "
        "Create a profile or select an existing profile."
    )

    profiles_df = get_new_user_profiles()

    profile_options = {"Create a new profile": None}

    for _, profile_row in profiles_df.iterrows():
        profile_id_value = int(profile_row["profile_id"])
        profile_label = (
            f"{profile_row['display_name']} "
            f"(Profile {profile_id_value})"
        )
        profile_options[profile_label] = profile_id_value

    selected_profile_label = st.selectbox(
        "👤 Choose Your Profile",
        list(profile_options.keys())
    )

    selected_profile_id = profile_options[selected_profile_label]

    if selected_profile_id is None:

        display_name = st.text_input(
            "Enter Your Name",
            placeholder="Example: Sindhu"
        )

        selected_genres = st.multiselect(
            "🎭 Select Your Favourite Genres",
            category_list[1:],
            default=["Comedy", "Drama"]
        )

    else:

        saved_profile = get_new_user_profile(selected_profile_id)
        display_name = saved_profile["display_name"]

        st.success(
            f"Welcome back, {display_name}! "
            f"Your profile has been loaded."
        )

        selected_genres = st.multiselect(
            "🎭 Update Your Favourite Genres",
            category_list[1:],
            default=saved_profile["favorite_genres"]
        )


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button("🎯 Get My Recommendations", width="stretch"):

    with st.spinner("🤖 Finding movies for you..."):

        if user_type == "New User":

            from src.cold_start_recommender import (
                get_cold_start_recommendations
            )

            rec_user_id = None
            rec_profile_id = None

            if not selected_genres:
                recommendations = None
                st.warning("Please select at least one favourite genre.")

            elif selected_profile_id is None and not display_name.strip():
                recommendations = None
                st.warning("Please enter your name to create a profile.")

            else:

                if selected_profile_id is None:
                    rec_profile_id = create_new_user_profile(
                        display_name,
                        selected_genres
                    )
                    st.success(
                        f"Profile created successfully! "
                        f"Your profile ID is {rec_profile_id}."
                    )

                else:
                    rec_profile_id = selected_profile_id
                    update_new_user_genres(
                        rec_profile_id,
                        selected_genres
                    )

                profile_feedback = get_new_user_feedback(rec_profile_id)

                recommendations = get_cold_start_recommendations(
                    selected_genres=selected_genres,
                    n=10,
                    profile_feedback=profile_feedback
                )

            rec_method_value = "Popular Movies (Cold Start)"
            rec_category_value = ", ".join(selected_genres)
            rec_mood_value = None

        else:

            if method == "KNN / Collaborative Filtering":
                recommendations = get_recommendations(
                    user_id=user_id,
                    n=10,
                    genre=selected_category
                )

            elif method == "SVD":
                if use_mood:
                    recommendations = get_mood_recommendations(
                        user_id=user_id,
                        mood=selected_mood,
                        n=10,
                        genre=selected_category
                    )
                else:
                    recommendations = get_svd_recommendations(
                        user_id=user_id,
                        n=10,
                        genre=selected_category
                    )

            elif method == "Hybrid SVD":
                recommendations = get_hybrid_recommendations(
                    user_id=user_id,
                    n=10,
                    genre=selected_category
                )

            else:
                recommendations = get_cnn_recommendations(
                    user_id=user_id,
                    n=10,
                    genre=selected_category
                )

            if recommendations is not None and not recommendations.empty:
                explanation_mood = (
                    selected_mood if method == "SVD" and use_mood else None
                )

                recommendations = add_recommendation_explanations(
                    recommendations,
                    user_id,
                    mood=explanation_mood
                )

            rec_user_id = int(user_id)
            rec_method_value = method
            rec_category_value = selected_category
            rec_mood_value = (
                selected_mood if method == "SVD" and use_mood else None
            )

    if recommendations is None or recommendations.empty:

        st.error("No recommendations found. Please check your selections.")
        st.session_state["recommendations"] = None

    else:

        recommendations = recommendations.merge(
            movies[["movie_id", "poster_url"]],
            on="movie_id",
            how="left"
        )

        st.session_state["recommendations"] = recommendations
        st.session_state["rec_user_id"] = rec_user_id
        st.session_state["rec_profile_id"] = (
            rec_profile_id if user_type == "New User" else None
        )
        st.session_state["rec_method"] = rec_method_value
        st.session_state["rec_category"] = rec_category_value
        st.session_state["rec_mood"] = rec_mood_value

# ============================================================
# DISPLAY RECOMMENDATIONS (PERSISTENT ACROSS RERUNS)
# ============================================================

recs = st.session_state.get("recommendations")

if recs is not None and not recs.empty:
    rec_user = st.session_state.get("rec_user_id")
    rec_cat = st.session_state.get("rec_category", "All Genres")
    rec_method = st.session_state.get("rec_method", "SVD")
    rec_mood = st.session_state.get("rec_mood")

    # Success message
    if rec_user is None:
        st.success("Popular movies matching your favourite genres are ready!")
    elif rec_cat == "All Genres":
        st.success("Your personalized recommendations are ready!")
    else:
        st.success(f"Your personalized {rec_cat} recommendations are ready!")

    # Model description
    if rec_method == "Popular Movies (Cold Start)":
        st.info(
            "Recommendation method: Genre-based cold-start filtering "
            "ranked using Bayesian popularity scores."
        )
    elif rec_method == "KNN / Collaborative Filtering":
        st.info(
            "Recommendation method: KNN / Item-Based "
            "Collaborative Filtering using Cosine Similarity."
        )
    elif rec_method == "SVD":
        st.info(
            "Recommendation method: SVD "
            "(Singular Value Decomposition)."
        )
    elif rec_method == "Hybrid SVD":
        st.info(
            "Recommendation method: Hybrid SVD combining SVD predictions "
            "with genre preference scores."
        )
    else:
        st.info(
            "Recommendation method: CNN "
            "(Convolutional Neural Network) for rating prediction."
        )

    # Category information

    if rec_mood:
        st.info(f"😊 Selected Mood: {rec_mood}")
        st.success("Mood-based movie recommendations are ready!")
    if rec_cat == "All Genres":
        st.info("Category: All Genres")
    else:
        st.info(f"Category: {rec_cat}")

    # Header
    if rec_user is None:
        st.subheader("🍿 Popular Movies for New Users")
    elif rec_cat == "All Genres":
        st.subheader(f"🍿 Top 10 Movies for User {rec_user}")
    else:
        st.subheader(f"🍿 Top 10 {rec_cat} Movies for User {rec_user}")

    # Movie cards grid
    for row_start in range(0, len(recs), 5):
        cols = st.columns(5)

        for i, col in enumerate(cols):
            index = row_start + i
            if index >= len(recs):
                continue

            row = recs.iloc[index]
            mid = int(row["movie_id"])

            with col:
                # Poster
                poster = row["poster_url"]
                if pd.notna(poster) and poster != "":
                    st.image(poster, use_container_width=True)
                else:
                    st.info("Poster unavailable")

                # Movie Title
                st.markdown(
                    f'<div class="movie-title">{row["title"]}</div>',
                    unsafe_allow_html=True
                )

                # Genres
                genres = str(row["genres"]).replace("|", " • ")
                st.markdown(
                    f'<div class="genre-text">{genres}</div>',
                    unsafe_allow_html=True
                )

                # Predicted Rating / Score
                if rec_method in ["SVD", "CNN"] and "PredictedRating" in recs.columns:
                    st.caption(
                        f"⭐ Predicted Rating: {row['PredictedRating']:.2f} / 5"
                    )
                elif "recommendation_score" in recs.columns:
                    st.caption(
                        f"📊 Score: {row['recommendation_score']:.2f}"
                    )

                # Why this movie was recommended
                if "Why This Movie?" in recs.columns:
                    with st.expander("💡 Why This Movie?"):
                        st.write(row["Why This Movie?"])

                if rec_user is not None:
                    # Watchlist button
                    if is_in_watchlist(rec_user, mid):
                        if st.button("🗑️ Remove Watchlist", key=f"wl_rm_{rec_user}_{mid}"):
                            remove_from_watchlist(rec_user, mid)
                            st.success("Removed from watchlist!")
                            st.rerun()
                    else:
                        if st.button("➕ Add to Watchlist", key=f"wl_add_{rec_user}_{mid}"):
                            movie_data = {
                                "movie_id": mid,
                                "title": str(row["title"]),
                                "genres": str(row.get("genres", "")),
                                "poster_url": str(row.get("poster_url", "") or "")
                            }
                            add_to_watchlist(rec_user, movie_data)
                            st.success(f"✅ Added to your watchlist!")
                            st.rerun()

                    # Feedback expander
                    with st.expander("⭐ Rate / Review"):
                        rating_sel = st.selectbox(
                            "Rating",
                            ["No Rating", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
                            key=f"fb_rate_{rec_user}_{mid}"
                        )
                        sentiment_sel = st.radio(
                            "Choice",
                            ["None", "Like", "Dislike"],
                            horizontal=True,
                            key=f"fb_sent_{rec_user}_{mid}"
                        )
                        comment_in = st.text_input(
                            "Comment",
                            key=f"fb_comm_{rec_user}_{mid}"
                        )
                        if st.button("Submit Feedback", key=f"fb_btn_{rec_user}_{mid}"):
                            r_val = int(rating_sel[0]) if rating_sel != "No Rating" else None
                            s_val = sentiment_sel if sentiment_sel in ["Like", "Dislike"] else None
                            c_val = comment_in.strip() if comment_in else None
                            if r_val is not None or s_val is not None or c_val:
                                save_feedback(
                                    user_id=rec_user,
                                    movie={"movie_id": mid, "title": row["title"]},
                                    feedback=s_val,
                                    rating=r_val,
                                    comments=c_val
                                )
                                st.success("Feedback saved!")
                                st.rerun()
                            else:
                                st.warning("Please provide a rating or choice.")

                else:
                    # New User Feedback
                    profile_id = st.session_state.get("rec_profile_id")

                    if profile_id is not None:
                        with st.expander("⭐ Like / Dislike / Rate"):

                            new_feedback = st.radio(
                                "Your opinion",
                                ["No Choice", "Like", "Dislike"],
                                horizontal=True,
                                key=f"new_fb_{profile_id}_{mid}"
                            )

                            new_rating = st.selectbox(
                                "Your rating",
                                [
                                    "No Rating", "1 Star", "2 Stars",
                                    "3 Stars", "4 Stars", "5 Stars"
                                ],
                                key=f"new_rate_{profile_id}_{mid}"
                            )

                            new_comment = st.text_input(
                                "Comment (optional)",
                                key=f"new_comment_{profile_id}_{mid}"
                            )

                            if st.button(
                                "Submit Feedback",
                                key=f"new_submit_{profile_id}_{mid}"
                            ):
                                rating_value = (
                                    int(new_rating[0])
                                    if new_rating != "No Rating"
                                    else None
                                )

                                feedback_value = (
                                    new_feedback
                                    if new_feedback in ["Like", "Dislike"]
                                    else None
                                )

                                if (
                                    rating_value is not None
                                    or feedback_value is not None
                                    or new_comment.strip()
                                ):
                                    save_new_user_feedback(
                                        profile_id=profile_id,
                                        movie={
                                            "movie_id": mid,
                                            "title": str(row["title"])
                                        },
                                        feedback=feedback_value,
                                        rating=rating_value,
                                        comments=new_comment
                                    )

                                    st.success("Feedback saved!")
                                    st.rerun()
                                else:
                                    st.warning(
                                        "Please select Like/Dislike or give a rating."
                                    )


# ============================================================
# HOW DOES IT WORK?
# ============================================================

st.divider()

st.subheader("🤖 How Does It Work?")

st.markdown(
    """
    **1. User ID**  
    The user enters their MovieLens User ID.

    **2. Recommendation Model**  
    The user selects KNN, SVD, or CNN.

    **3. Movie Category**  
    The user can select a preferred movie category such as
    Action, Comedy, Drama, Sci-Fi, Thriller, or Romance.

    **4. AI Recommendation**  
    The selected machine learning model analyzes user-movie
    interaction patterns and generates candidate recommendations.

    **5. Category Filtering**  
    The recommendations are filtered according to the selected
    movie category.

    **6. Personalized Top 10**  
    The system selects the Top 10 movies from the filtered
    recommendations.

    **7. Movie Posters**  
    Poster information is obtained through the TMDB API and
    displayed with the recommended movies.
    """
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

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


# ============================================================
# TECHNOLOGY / DATASET INFORMATION
# ============================================================

st.divider()

st.caption(
    "Recommendation Methods: KNN / Item-Based Collaborative Filtering "
    "using Cosine Similarity, SVD, and CNN."
)

st.caption(
    "Dataset: MovieLens 1M"
)

st.caption(
    "Movie poster information provided through the TMDB API."
)

