# AI-Powered Movie Recommendation System

## 1. Project Overview

The **AI-Powered Movie Recommendation System** is a machine learning-based web application that recommends movies according to users' preferences and past ratings.

The system uses the MovieLens 1M dataset and implements four recommendation algorithms: K-Nearest Neighbors (KNN), Singular Value Decomposition (SVD), Non-negative Matrix Factorization (NMF), and CoClustering.

The four algorithms are compared using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). Based on the model comparison, SVD is selected as the final recommendation model.

The application is developed using Python and Streamlit.

## 2. Project Objectives

- To develop a personalized movie recommendation system.
- To analyze user ratings and movie preferences.
- To implement and compare four machine learning algorithms.
- To select the best-performing model based on evaluation metrics.
- To recommend movies that users have not rated previously.
- To provide an interactive and user-friendly interface.

## 3. Dataset

This project uses the **MovieLens 1M Dataset** provided by GroupLens.

Dataset link: https://grouplens.org/datasets/movielens/1m/

Dataset details:

- 1,000,209 movie ratings
- 6,040 users
- 3,883 movie records
- Ratings range from 1 to 5

Main dataset files:

- `ratings.dat` – Contains user IDs, movie IDs, ratings and timestamps.
- `movies.dat` – Contains movie IDs, titles and genres.
- `users.dat` – Contains user demographic information.

## 4. Machine Learning Algorithms

Four algorithms are implemented and compared:

### K-Nearest Neighbors (KNN)
Uses similarity between users or movies to identify similar preferences and generate recommendations.

### Singular Value Decomposition (SVD)
Uses matrix factorization to learn latent user and movie factors and predict ratings for movies.

### Non-negative Matrix Factorization (NMF)
Decomposes the rating matrix into non-negative user and movie factor matrices.

### CoClustering
Groups users and movies into clusters to identify rating patterns and generate predictions.

**Final Selected Model:** SVD

SVD is selected as the final recommendation model based on the experimental comparison.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Scikit-Surprise | Recommendation algorithms |
| Scikit-learn | Data processing and evaluation utilities |
| Streamlit | Web application interface |
| SQLite | Watchlist and feedback storage |
| TMDB API | Movie poster information |
| Matplotlib | Graphs and visualization |
| Git and GitHub | Version control |

## 6. Application Features

- Personalized movie recommendations.
- User ID-based recommendations.
- Top 10 movie recommendations.
- Movie browsing by genre.
- Movie posters and movie information.
- Mood-based movie recommendations.
- Explanation of why a movie is recommended.
- Personal watchlist.
- User feedback and reviews.

## 7. System Workflow

1. Collect the MovieLens 1M dataset.
2. Load ratings and movie information.
3. Preprocess the dataset.
4. Prepare the user–movie rating data.
5. Train and evaluate the recommendation algorithms.
6. Select SVD as the final model.
7. Enter the user ID.
8. Predict ratings for unrated movies.
9. Remove movies already rated by the user.
10. Rank movies based on predicted ratings.
11. Display the Top 10 recommendations through Streamlit.

## 8. Model Evaluation

The recommendation algorithms are evaluated using:

- **MSE (Mean Squared Error):** Measures the average squared difference between actual and predicted ratings.
- **RMSE (Root Mean Squared Error):** Measures the square root of the average squared prediction error.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted ratings.

Lower values indicate lower rating-prediction errors.

## 9. Project Folder Structure

```text
Movie_Recommendation_system/
│
├── app.py
├── mood_recommendations.py
├── compare_models.py
│
├── data/
│   ├── ratings.dat
│   ├── movies.dat
│   ├── users.dat
│   ├── movie_posters.csv
│   └── user_data.db
│
├── src/
│   ├── load_data.py
│   ├── recommend.py
│   ├── svd_recommender.py
│   └── fetch_posters.py
│
├── models/
│   └── svd_model.pkl
│
├── venv/
│
├── model_results.csv
├── recommendation_comparison.csv
├── model_comparison.png
├── mae_comparison.png
├── mse_comparison.png
└── README.md
```

*Note: This is an illustrative structure. Keep only the files and folders that actually exist in your project.*

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project folder

```bash
cd Movie_Recommendation_system
```

### Step 3: Create a virtual environment

```bash
python3 -m venv venv
```

### Step 4: Activate the virtual environment

```bash
source venv/bin/activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 11. Future Enhancements

- Improve recommendation accuracy.
- Add more advanced recommendation algorithms.
- Enhance mood-based recommendations.
- Improve recommendation explanations.
- Support additional movie datasets.
- Improve application performance and scalability.

## 12. References

1. GroupLens Research, MovieLens 1M Dataset. https://grouplens.org/datasets/movielens/1m/
2. Surprise: A Python Scikit for Building and Analyzing Recommender Systems. https://surprise.readthedocs.io/
3. Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," Computer, 2009.
