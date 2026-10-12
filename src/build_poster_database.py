import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load MovieLens movies
movies = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

poster_urls = []

print(f"Fetching posters for {len(movies)} movies...")

for index, row in movies.iterrows():

    title = row["title"]

    # Remove year from title
    search_title = title.rsplit(" (", 1)[0]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": search_title
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code == 200:

            results = response.json().get("results", [])

            if results and results[0].get("poster_path"):

                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + results[0]["poster_path"]
                )

            else:
                poster_url = ""

        else:
            poster_url = ""

    except requests.RequestException:
        poster_url = ""

    poster_urls.append(poster_url)

    # Progress
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(movies)} movies")

    # Small delay to avoid sending requests too quickly
    time.sleep(0.05)


movies["poster_url"] = poster_urls

output_file = "data/movie_posters.csv"

movies.to_csv(
    output_file,
    index=False
)

print()
print("Poster database created successfully!")
print("Saved to:", output_file)
