import streamlit as st
import pandas as pd
from pathlib import Path

from src.user_storage import (
    get_feedback,
    save_feedback,
    delete_feedback
)

st.set_page_config(
    page_title="My Feedback",
    page_icon="⭐",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.feedback-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}
.feedback-card-title {
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 4px;
}
.feedback-comment {
    background-color: rgba(255, 255, 255, 0.05);
    border-left: 3px solid #ffaa00;
    padding: 8px 12px;
    border-radius: 4px;
    margin-top: 6px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="feedback-title">⭐ Movie Feedback & Reviews</div>', unsafe_allow_html=True)
st.write("Submit ratings, sentiment, and written reviews for movies, and review your feedback history.")

# Synchronize User ID across pages
default_user = st.session_state.get("user_id", 1)

user_id = st.number_input(
    "Enter MovieLens User ID",
    min_value=1,
    max_value=6040,
    value=int(default_user),
    step=1,
    key="feedback_user_id"
)
st.session_state["user_id"] = user_id

# Load movie dataset for search/selection
BASE_DIR = Path(__file__).resolve().parent.parent
movies_file = BASE_DIR / "data/movie_posters.csv"
movies_df = pd.read_csv(movies_file)

# -------------------------------------------------------------
# SUBMIT FEEDBACK SECTION
# -------------------------------------------------------------
with st.expander("📝 Submit Feedback for a Movie", expanded=True):
    col_movie, col_rating = st.columns([3, 2])

    with col_movie:
        selected_movie_title = st.selectbox(
            "Select Movie",
            options=movies_df["title"].tolist(),
            key="fb_movie_select"
        )
        selected_movie_row = movies_df[movies_df["title"] == selected_movie_title].iloc[0]

    with col_rating:
        star_rating = st.selectbox(
            "Rating (1 to 5 Stars)",
            options=["No Rating", "1 Star ⭐", "2 Stars ⭐⭐", "3 Stars ⭐⭐⭐", "4 Stars ⭐⭐⭐⭐", "5 Stars ⭐⭐⭐⭐⭐"],
            index=5,
            key="fb_star_select"
        )

    col_sentiment, col_comment = st.columns([1.5, 3.5])

    with col_sentiment:
        sentiment = st.radio(
            "Overall Sentiment",
            options=["None", "Like", "Dislike"],
            horizontal=True,
            key="fb_sentiment_radio"
        )

    with col_comment:
        written_comment = st.text_area(
            "Written Feedback / Comments (Optional)",
            placeholder="Share your thoughts about this movie...",
            key="fb_comment_text",
            height=70
        )

    if st.button("🚀 Submit Feedback", use_container_width=True, type="primary"):
        # Parse inputs
        rating_val = None
        if star_rating != "No Rating":
            rating_val = int(star_rating[0])

        sentiment_val = sentiment if sentiment in ["Like", "Dislike"] else None
        comment_val = written_comment.strip() if written_comment else None

        # Validation: At least one feedback signal must be provided
        if rating_val is None and sentiment_val is None and not comment_val:
            st.error("Please provide at least a star rating, sentiment (Like/Dislike), or written comments.")
        else:
            try:
                movie_dict = {
                    "movie_id": int(selected_movie_row["movie_id"]),
                    "title": str(selected_movie_row["title"])
                }
                save_feedback(
                    user_id=user_id,
                    movie=movie_dict,
                    feedback=sentiment_val,
                    rating=rating_val,
                    comments=comment_val
                )
                st.toast(f"✅ Feedback saved for {selected_movie_title}!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not save feedback: {e}")

st.divider()

# -------------------------------------------------------------
# PREVIOUS FEEDBACK HISTORY
# -------------------------------------------------------------
st.subheader(f"📋 Feedback History for User {user_id}")

feedback_df = get_feedback(user_id)

if feedback_df.empty:
    st.info(f"No feedback records found for User {user_id}. Submit feedback using the form above or on the Recommendations page.")
else:
    st.success(f"You have submitted feedback on {len(feedback_df)} movie(s).")

    for _, row in feedback_df.iterrows():
        col_info, col_del = st.columns([4, 1])

        with col_info:
            st.markdown(f'<div class="feedback-card-title">{row["title"]}</div>', unsafe_allow_html=True)

            details = []
            if row.get("feedback"):
                sentiment_icon = "👍 Like" if row["feedback"] == "Like" else "👎 Dislike"
                details.append(f"**Choice:** {sentiment_icon}")

            if pd.notna(row.get("rating")) and row["rating"] is not None:
                details.append(f"**Rating:** {int(row['rating'])}/5 ⭐")

            if details:
                st.write(" • ".join(details))

            if "comments" in row and pd.notna(row["comments"]) and row["comments"]:
                st.markdown(f'<div class="feedback-comment">"{row["comments"]}"</div>', unsafe_allow_html=True)

            st.caption(f"Movie ID: {row['movie_id']} • Last updated: {row['updated_at']}")

        with col_del:
            if st.button(
                "🗑️ Delete",
                key=f"del_fb_{user_id}_{row['movie_id']}",
                use_container_width=True
            ):
                try:
                    delete_feedback(user_id, row["movie_id"])
                    st.toast(f"Deleted feedback for '{row['title']}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete feedback: {e}")

        st.divider()
