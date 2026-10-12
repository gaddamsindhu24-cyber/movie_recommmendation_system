import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TMDB_API_KEY")

url = "https://api.themoviedb.org/3/search/movie"

params = {
    "api_key": api_key,
    "query": "The Matrix"
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print("TMDB connection successful!")
    print("Movie:", data["results"][0]["title"])
else:
    print("TMDB connection failed.")
    print(response.text)
