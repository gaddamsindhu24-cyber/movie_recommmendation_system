import sqlite3
import os
from pathlib import Path
import pandas as pd
import json

# Store the database inside the project's data folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "user_data.db"


def get_connection():
    """Open a connection to the SQLite database with network UNC fallback."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.execute("PRAGMA busy_timeout = 10000;")
        # Test query to detect network UNC file lock issues early
        conn.execute("SELECT 1;")
        return conn
    except sqlite3.OperationalError:
        # Fallback if accessed via network share (e.g. Windows host accessing \\wsl.localhost)
        if os.name == "nt":
            import tempfile
            local_tmp = Path(tempfile.gettempdir()) / "movie_recs_user_data.db"
            return sqlite3.connect(local_tmp)
        raise


def init_db():
    """Create tables if they do not already exist, and execute safe migrations."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    genres TEXT,
                    poster_url TEXT,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, movie_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    feedback TEXT,
                    rating INTEGER,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, movie_id),
                    CHECK (feedback IN ('Like', 'Dislike') OR feedback IS NULL),
                    CHECK (rating IS NULL OR rating BETWEEN 1 AND 5)
                )
            """)

            # Migration: Ensure comments column exists in feedback table
            cursor.execute("PRAGMA table_info(feedback)")
            columns = [col[1] for col in cursor.fetchall()]
            if "comments" not in columns:
                cursor.execute("ALTER TABLE feedback ADD COLUMN comments TEXT")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS new_user_profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name TEXT NOT NULL,
                    favorite_genres TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS new_user_feedback (
                    profile_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    feedback TEXT,
                    rating INTEGER,
                    comments TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (profile_id, movie_id),
                    CHECK (feedback IN ('Like', 'Dislike') OR feedback IS NULL),
                    CHECK (rating IS NULL OR rating BETWEEN 1 AND 5)
                )
            """)

            conn.commit()
    except Exception as e:
        print(f"Notice: SQLite initialization deferred ({e})")


def add_to_watchlist(user_id, movie):
    """Add or update a movie in user's watchlist."""
    try:
        user_id_int = int(user_id)
        movie_id_int = int(movie["movie_id"])
        title_str = str(movie["title"])
        genres_str = str(movie.get("genres", ""))
        poster_str = str(movie.get("poster_url", "") or "")

        with get_connection() as conn:
            conn.execute("""
                INSERT INTO watchlist
                    (user_id, movie_id, title, genres, poster_url, added_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET
                    title = excluded.title,
                    genres = excluded.genres,
                    poster_url = excluded.poster_url
            """, (user_id_int, movie_id_int, title_str, genres_str, poster_str))
            conn.commit()
        return True
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        raise e


def get_watchlist(user_id):
    """Return a user's saved movies as a DataFrame."""
    try:
        with get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT movie_id, title, genres, poster_url, added_at
                FROM watchlist
                WHERE user_id = ?
                ORDER BY added_at DESC
            """, conn, params=(int(user_id),))
            return df
    except Exception as e:
        print(f"Error reading watchlist: {e}")
        return pd.DataFrame(columns=["movie_id", "title", "genres", "poster_url", "added_at"])


def is_in_watchlist(user_id, movie_id):
    """Check if a specific movie is already in user's watchlist."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM watchlist
                WHERE user_id = ? AND movie_id = ?
                LIMIT 1
            """, (int(user_id), int(movie_id)))
            return cursor.fetchone() is not None
    except Exception:
        return False


def get_watchlist_movie_ids(user_id):
    """Return set of movie_ids saved by the user for fast O(1) checks."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT movie_id FROM watchlist
                WHERE user_id = ?
            """, (int(user_id),))
            return {row[0] for row in cursor.fetchall()}
    except Exception:
        return set()


def remove_from_watchlist(user_id, movie_id):
    """Remove one movie from a user's watchlist."""
    try:
        with get_connection() as conn:
            conn.execute("""
                DELETE FROM watchlist
                WHERE user_id = ? AND movie_id = ?
            """, (int(user_id), int(movie_id)))
            conn.commit()
        return True
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        raise e


def save_feedback(user_id, movie, feedback=None, rating=None, comments=None):
    """Save or update a user's feedback for a movie."""
    if feedback not in (None, "Like", "Dislike"):
        raise ValueError("Feedback must be Like, Dislike, or None.")

    if rating is not None:
        try:
            rating = int(rating)
        except (ValueError, TypeError):
            raise ValueError("Rating must be an integer between 1 and 5.")
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5.")

    comments_clean = str(comments).strip() if (comments is not None and str(comments).strip() != "") else None

    try:
        user_id_int = int(user_id)
        movie_id_int = int(movie["movie_id"])
        title_str = str(movie["title"])

        with get_connection() as conn:
            conn.execute("""
                INSERT INTO feedback
                    (user_id, movie_id, title, feedback, rating, comments, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET
                    title = excluded.title,
                    feedback = excluded.feedback,
                    rating = excluded.rating,
                    comments = excluded.comments,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_id_int,
                movie_id_int,
                title_str,
                feedback,
                rating,
                comments_clean
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        raise e


def get_feedback(user_id):
    """Return saved feedback for a user."""
    try:
        with get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT movie_id, title, feedback, rating, comments, updated_at
                FROM feedback
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """, conn, params=(int(user_id),))
            return df
    except Exception as e:
        print(f"Error getting feedback: {e}")
        return pd.DataFrame(columns=["movie_id", "title", "feedback", "rating", "comments", "updated_at"])


def delete_feedback(user_id, movie_id):
    """Delete a user's feedback for a movie."""
    try:
        with get_connection() as conn:
            conn.execute("""
                DELETE FROM feedback
                WHERE user_id = ? AND movie_id = ?
            """, (int(user_id), int(movie_id)))
            conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting feedback: {e}")
        raise e




# ============================================================
# NEW USER PROFILE FUNCTIONS
# ============================================================

def create_new_user_profile(display_name, favorite_genres):
    """Create a separate profile for a new user."""
    name = str(display_name).strip()
    genres = list(dict.fromkeys(favorite_genres))

    if not name:
        raise ValueError("Display name cannot be empty.")

    if not genres:
        raise ValueError("Select at least one favourite genre.")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO new_user_profiles
                (display_name, favorite_genres)
            VALUES (?, ?)
            """,
            (name, json.dumps(genres))
        )
        conn.commit()
        return cursor.lastrowid


def get_new_user_profiles():
    """Return all saved new-user profiles."""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT profile_id, display_name, favorite_genres, created_at
            FROM new_user_profiles
            ORDER BY profile_id DESC
            """,
            conn
        )


def get_new_user_profile(profile_id):
    """Get one new-user profile by its profile ID."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT profile_id, display_name, favorite_genres, created_at
            FROM new_user_profiles
            WHERE profile_id = ?
            """,
            (int(profile_id),)
        ).fetchone()

    if row is None:
        return None

    return {
        "profile_id": row[0],
        "display_name": row[1],
        "favorite_genres": json.loads(row[2]),
        "created_at": row[3]
    }


def update_new_user_genres(profile_id, favorite_genres):
    """Update a new user's favourite genres."""
    genres = list(dict.fromkeys(favorite_genres))

    if not genres:
        raise ValueError("Select at least one favourite genre.")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE new_user_profiles
            SET favorite_genres = ?
            WHERE profile_id = ?
            """,
            (json.dumps(genres), int(profile_id))
        )
        conn.commit()
        return cursor.rowcount > 0



# ============================================================
# NEW USER FEEDBACK FUNCTIONS
# ============================================================

def save_new_user_feedback(
    profile_id, movie, feedback=None, rating=None, comments=None
):
    """Save feedback separately for a new-user profile."""

    if feedback not in (None, "Like", "Dislike"):
        raise ValueError("Feedback must be Like, Dislike, or None.")

    if rating is not None:
        rating = int(rating)
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5.")

    comments_clean = (
        str(comments).strip()
        if comments is not None and str(comments).strip()
        else None
    )

    if feedback is None and rating is None and comments_clean is None:
        raise ValueError("Please provide a rating, choice, or comment.")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO new_user_feedback
                (profile_id, movie_id, title, feedback, rating, comments)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, movie_id) DO UPDATE SET
                title = excluded.title,
                feedback = excluded.feedback,
                rating = excluded.rating,
                comments = excluded.comments,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                int(profile_id),
                int(movie["movie_id"]),
                str(movie["title"]),
                feedback,
                rating,
                comments_clean
            )
        )
        conn.commit()

    return True


def get_new_user_feedback(profile_id):
    """Get all feedback submitted by a new-user profile."""

    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT movie_id, title, feedback, rating, comments, updated_at
            FROM new_user_feedback
            WHERE profile_id = ?
            ORDER BY updated_at DESC
            """,
            conn,
            params=(int(profile_id),)
        )


# Initialize tables and migrations upon import
init_db()
