
import streamlit as st
import sqlite3
import pandas as pd
from src.user_storage import DB_PATH

st.set_page_config(
    page_title="New User Feedback History",
    page_icon="💬",
    layout="wide"
)

st.title("💬 New User Feedback History")
st.write("View feedback submitted by new users.")

# Connect to the existing database
conn = sqlite3.connect(DB_PATH)

# Get all new user profiles
profiles = pd.read_sql_query(
    """
    SELECT profile_id, display_name
    FROM new_user_profiles
    ORDER BY profile_id
    """,
    conn
)

if profiles.empty:
    st.info("No new user profiles found.")
else:
    # Select a user profile
    profile_options = profiles["profile_id"].tolist()

    profile_names = dict(
        zip(profiles["profile_id"], profiles["display_name"])
    )

    selected_profile = st.selectbox(
        "Select New User Profile",
        profile_options,
        format_func=lambda pid: (
            f"{profile_names[pid]} (Profile ID: {pid})"
        )
    )

    # Get feedback for the selected profile
    feedback = pd.read_sql_query(
        """
        SELECT
            movie_id,
            title,
            feedback,
            rating,
            comments,
            updated_at
        FROM new_user_feedback
        WHERE profile_id = ?
        ORDER BY updated_at DESC
        """,
        conn,
        params=(selected_profile,)
    )

    st.subheader(
        f"Feedback from {profile_names[selected_profile]}"
    )

    if feedback.empty:
        st.info("This user has not submitted any feedback yet.")
    else:
        st.metric("Total Feedback Records", len(feedback))

        feedback.columns = [
            "Movie ID",
            "Movie Title",
            "Like / Dislike",
            "Star Rating",
            "Comments",
            "Date and Time"
        ]

        st.dataframe(
            feedback,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Feedback Summary")

        col1, col2, col3 = st.columns(3)

        likes = (
            feedback["Like / Dislike"] == "Like"
        ).sum()

        dislikes = (
            feedback["Like / Dislike"] == "Dislike"
        ).sum()

        ratings = feedback["Star Rating"].notna().sum()

        col1.metric("Likes", int(likes))
        col2.metric("Dislikes", int(dislikes))
        col3.metric("Star Ratings", int(ratings))

conn.close()
