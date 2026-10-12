import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

input_file = "data/recommendations.csv"
output_file = "data/recommendations_with_posters.csv"

recommendations = pd.read_csv(input_file)

poster_urls = []

for title in recommendations["title"]:
    # Remove year, for example:
    # "Matrix, The (1999)" -> "Matrix, The"
    search_title = title.rsplit(" (", 1)[0]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": search_title
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json()["results"]

        if results and results[0]["poster_path"]:
            poster_url = (
                "https://image.tmdb.org/t/p/w500"
                + results[0]["poster_path"]
            )
        else:
            poster_url = ""

    else:
        poster_url = ""

    poster_urls.append(poster_url)

recommendations["poster_url"] = poster_urls

recommendations.to_csv(output_file, index=False)

print("Poster URLs added successfully!")
print("Saved to:", output_file)
