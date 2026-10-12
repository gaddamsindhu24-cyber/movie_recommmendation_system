import streamlit as st
import pandas as pd
from pathlib import Path

try:
    from src.user_storage import (
        add_to_watchlist,
        remove_from_watchlist,
        is_in_watchlist
    )
except ImportError:
    import importlib
    import src.user_storage
    importlib.reload(src.user_storage)
    from src.user_storage import (
        add_to_watchlist,
        remove_from_watchlist,
        is_in_watchlist
    )

st.set_page_config(
    page_title="Browse Movies by Genre",
    page_icon="🎭",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MOVIES_PATH = BASE_DIR / "data/movie_posters.csv"

@st.cache_data
def load_movies():
    return pd.read_csv(MOVIES_PATH)

movies = load_movies()

st.title("🎭 Explore Movies by Genre")
st.write("Choose your favourite category and discover movies. You can also save titles directly to your watchlist.")

# Synchronize User ID across pages
default_user = st.session_state.get("user_id", 1)
user_id = st.sidebar.number_input(
    "Your MovieLens User ID",
    min_value=1,
    max_value=6040,
    value=int(default_user),
    step=1,
    key="browse_user_id"
)
st.session_state["user_id"] = user_id

genres = [
    "All Genres", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

emojis = {
    "All Genres": "🍿",
    "Action": "💥",
    "Adventure": "🗺️",
    "Animation": "🎨",
    "Children's": "🧸",
    "Comedy": "😂",
    "Crime": "🕵️",
    "Documentary": "🎥",
    "Drama": "🎭",
    "Fantasy": "🪄",
    "Film-Noir": "🌑",
    "Horror": "👻",
    "Musical": "🎵",
    "Mystery": "🔍",
    "Romance": "❤️",
    "Sci-Fi": "🚀",
    "Thriller": "⚡",
    "War": "⚔️",
    "Western": "🤠"
}

st.markdown("""
<style>
.genre-heading {
    font-size: 27px;
    font-weight: 700;
    margin-bottom: 6px;
}
.movie-title {
    font-size: 15px;
    font-weight: 600;
    min-height: 44px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

for start in range(0, len(genres), 4):
    row_genres = genres[start:start + 4]
    columns = st.columns(4)

    for col, genre in zip(columns, row_genres):
        with col:
            if st.button(
                f"{emojis[genre]}  {genre}",
                key=f"genre_{genre}",
                use_container_width=True,
                type="primary" if st.session_state.get(
                    "selected_genre", "All Genres"
                ) == genre else "secondary"
            ):
                st.session_state["selected_genre"] = genre

selected = st.session_state.get("selected_genre", "All Genres")

st.divider()
st.markdown(
    f'<div class="genre-heading">{emojis[selected]} {selected} Movies</div>',
    unsafe_allow_html=True
)

if selected != "All Genres":
    filtered = movies[
        movies["genres"].str.contains(
            selected, case=False, na=False, regex=False
        )
    ].copy()
else:
    filtered = movies.copy()

st.caption(f"{len(filtered)} movies found in this category")

if filtered.empty:
    st.info("No movies found in this category.")
else:
    poster_column = "poster_url" if "poster_url" in filtered.columns else None
    columns = st.columns(5)

    for i, (_, movie) in enumerate(filtered.head(100).iterrows()):
        mid = int(movie["movie_id"])
        with columns[i % 5]:
            if poster_column and pd.notna(movie[poster_column]) and movie[poster_column]:
                st.image(movie[poster_column], use_container_width=True)
            else:
                st.markdown("🎬 *Poster unavailable*")

            st.markdown(
                f'<div class="movie-title">{movie["title"]}</div>',
                unsafe_allow_html=True
            )
            st.caption(str(movie["genres"]).replace("|", " • "))

            in_wl = is_in_watchlist(user_id, mid)
            if in_wl:
                if st.button("🗑️ Remove", key=f"browse_wl_rm_{user_id}_{mid}", use_container_width=True):
                    remove_from_watchlist(user_id, mid)
                    st.toast(f"Removed '{movie['title']}' from watchlist.")
                    st.rerun()
            else:
                if st.button("➕ Watchlist", key=f"browse_wl_add_{user_id}_{mid}", use_container_width=True):
                    add_to_watchlist(user_id, {
                        "movie_id": mid,
                        "title": movie["title"],
                        "genres": movie.get("genres", ""),
                        "poster_url": movie.get("poster_url", "")
                    })
                    st.toast(f"Saved '{movie['title']}' to watchlist!")
                    st.rerun()
