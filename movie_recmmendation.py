

"""
Movie Recommendation Engine Module
-----------------------------------
Generates content-based movie recommendations using TF-IDF and Cosine Similarity.
"""
pip install: pandas sckit-learn
from typing import List
#import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Content-based Movie Recommendation System."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes recommender with dataset.

        Data required columns: ['title', 'genres', 'description']
        """
        required_cols = {"title", "genres", "description"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Dataset must contain columns: {required_cols}")

        self.df = data.copy()
        self._prepare_data()
        self._build_similarity_matrix()

    def _prepare_data(self) -> None:
        """Combines and cleans metadata features into a unified text feature."""
        self.df["features"] = (
            self.df["genres"].fillna("") + " " + self.df["description"].fillna("")
        ).str.lower()

    def _build_similarity_matrix(self) -> None:
        """Computes TF-IDF vectors and the cosine similarity matrix."""
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(self.df["features"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Mapping lower-cased title to index for case-insensitive lookup
        self.indices = pd.Series(
            self.df.index, index=self.df["title"].str.lower()
        ).drop_duplicates()

    def recommend(
        self, movie_title: str, top_n: int = 3
    ) -> pd.DataFrame:
        """Returns top_n movie recommendations based on content similarity.

        Args:
            movie_title (str): Name of the reference movie.
            top_n (int): Number of recommendations to return.

        Returns:
            pd.DataFrame: Recommended movies with title, genres, and similarity score.
        """
        lookup_title = movie_title.strip().lower()

        if lookup_title not in self.indices:
            raise KeyError(
                f"Movie '{movie_title}' was not found in the dataset database."
            )

        idx = self.indices[lookup_title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Calculate similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip the movie itself (index 0) and fetch top_n
        sim_scores = sim_scores[1 : top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [round(i[1], 4) for i in sim_scores]

        result = self.df.iloc[movie_indices][["title", "genres"]].copy()
        result["similarity_score"] = scores

        return result.reset_index(drop=True)


# ==========================================
# Example Execution Workflow
# ==========================================
if __name__ == "__main__":
    # Sample Dataset
    movies_db = pd.DataFrame(
        {
            "title": [
                "The Dark Knight",
                "Inception",
                "Interstellar",
                "The Avengers",
                "The Notebook",
                "La La Land",
            ],
            "genres": [
                "Action Crime Drama",
                "Action Sci-Fi Thriller",
                "Adventure Drama Sci-Fi",
                "Action Sci-Fi Superhero",
                "Drama Romance",
                "Comedy Drama Romance Musical",
            ],
            "description": [
                "Batman fights crime and chaos in Gotham City orchestrated by the Joker.",
                "A thief enters dreams to steal secrets and plant ideas in subconsciousness.",
                "A team of explorers travels through a wormhole in space to ensure humanity's survival.",
                "Earth's mightiest heroes assemble to stop an alien invasion.",
                "A tragic love story between two young lovers in 1940s South Carolina.",
                "An aspiring actress and a dedicated jazz musician fall in love in Los Angeles.",
            ],
        }
    )

    # Initialize Engine
    recommender = MovieRecommender(movies_db)

    # Get Recommendations
    query = "Inception"
    print(f"Top recommendations for '{query}':\n")
    recs = recommender.recommend(query, top_n=3)
    print(recs.to_string(index=False))