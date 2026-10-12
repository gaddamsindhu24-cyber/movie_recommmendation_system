import streamlit as st
import pandas as pd

from src.user_storage import (
    get_watchlist,
    remove_from_watchlist
)

st.set_page_config(
    page_title="My Watchlist",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.watchlist-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}
.watchlist-item-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 4px;
}
.watchlist-genres {
    font-size: 14px;
    color: #888888;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="watchlist-title">🎬 My Watchlist</div>', unsafe_allow_html=True)
st.write("View and manage your saved movies. Watchlist records are stored persistently in SQLite.")

# Synchronize User ID across pages
default_user = st.session_state.get("user_id", 1)

user_id = st.number_input(
    "Enter MovieLens User ID",
    min_value=1,
    max_value=6040,
    value=int(default_user),
    step=1,
    key="watchlist_user_id"
)
st.session_state["user_id"] = user_id

watchlist = get_watchlist(user_id)

st.divider()

if watchlist.empty:
    st.info(f"Your watchlist is currently empty for User {user_id}. Browse recommendations or genres to add movies.")
else:
    st.success(f"You have {len(watchlist)} saved movie(s) in your watchlist.")

    for _, movie in watchlist.iterrows():
        col1, col2, col3 = st.columns([1, 4, 1.2])

        with col1:
            poster = movie.get("poster_url")
            if pd.notna(poster) and poster != "":
                st.image(poster, use_container_width=True)
            else:
                st.caption("🎬 *Poster unavailable*")

        with col2:
            st.markdown(f'<div class="watchlist-item-title">{movie["title"]}</div>', unsafe_allow_html=True)
            genres_display = str(movie.get("genres", "")).replace("|", " • ")
            st.markdown(f'<div class="watchlist-genres">{genres_display}</div>', unsafe_allow_html=True)
            st.caption(f"Movie ID: {movie['movie_id']}")
            if "added_at" in movie and pd.notna(movie["added_at"]):
                st.caption(f"Saved on: {movie['added_at']}")

        with col3:
            if st.button(
                "🗑️ Remove",
                key=f"remove_wl_{user_id}_{movie['movie_id']}",
                use_container_width=True
            ):
                try:
                    remove_from_watchlist(user_id, movie["movie_id"])
                    st.toast(f"Removed '{movie['title']}' from your watchlist.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not remove movie: {e}")

        st.divider()
