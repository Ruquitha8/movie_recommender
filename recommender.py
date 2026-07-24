import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Maps everyday mood words to one or more genres in the dataset.
# Feel free to extend this — it's just a dictionary, no ML needed here.
MOOD_TO_GENRES = {
    "happy": ["Comedy"],
    "fun": ["Comedy", "Adventure"],
    "excited": ["Action", "Adventure"],
    "adrenaline": ["Action", "Thriller"],
    "sad": ["Drama"],
    "romantic": ["Romance"],
    "date night": ["Romance", "Comedy"],
    "scared": ["Horror"],
    "spooky": ["Horror", "Mystery"],
    "thoughtful": ["Sci-Fi", "Mystery"],
    "curious": ["Mystery", "Sci-Fi"],
    "relaxed": ["Family", "Animation"],
    "cozy": ["Family", "Animation"],
    "nostalgic": ["Drama", "Family"],
    "adventurous": ["Adventure"],
    "inspired": ["Biography", "Sport"],
}


class MovieRecommender:
    def __init__(self, csv_path: str):
        # Step 1: Load the dataset into a pandas DataFrame
        self.df = pd.read_csv(csv_path)

        # Step 2: Combine genres + description into one "content soup" per movie.
        # We repeat the genres a few times so they carry more weight than the
        # description in the similarity calculation — genres are a stronger
        # signal of "this is the same kind of movie" than a short blurb.
        self.df["content"] = (
            (self.df["genres"].str.replace("|", " ", regex=False) + " ") * 3
            + self.df["description"]
        )

        # Step 3: Convert text into TF-IDF vectors.
        # TF-IDF = Term Frequency-Inverse Document Frequency. It scores words
        # by how important they are to a document relative to the whole
        # dataset (common words like "the" get downweighted automatically).
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])

        # Step 4: Precompute similarity between every pair of movies.
        # cosine_similarity gives a score from 0 (nothing alike) to 1 (identical)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Helper lookup: movie title (lowercase) -> row index
        self.title_to_index = {
            title.lower(): idx for idx, title in enumerate(self.df["title"])
        }

    def list_titles(self):
        """Return all movie titles in the dataset."""
        return self.df["title"].tolist()

    def list_genres(self):
        """Return every unique genre present in the dataset."""
        genres = set()
        for genre_str in self.df["genres"]:
            genres.update(genre_str.split("|"))
        return sorted(genres)

    def list_languages(self):
        """Return every unique language present in the dataset."""
        return sorted(self.df["language"].unique().tolist())

    def recommend_by_language(self, language: str, genre: str = None, top_n: int = 10):
        """
        Return movies in a given language (e.g. "Telugu", "Hindi", "English"),
        optionally narrowed further by genre (e.g. "Telugu comedy movies").
        Returns None if the language isn't recognized.
        """
        language = language.lower().strip()
        lang_lookup = {l.lower(): l for l in self.list_languages()}
        if language not in lang_lookup:
            return None

        subset = self.df[self.df["language"] == lang_lookup[language]]

        if genre:
            genre_lookup = {g.lower(): g for g in self.list_genres()}
            genre = genre.lower().strip()
            if genre in genre_lookup:
                target_genre = genre_lookup[genre]
                subset = subset[subset["genres"].str.contains(target_genre)]

        subset = subset.sample(frac=1)  # shuffle so results vary between runs
        results = []
        for _, row in subset.head(top_n).iterrows():
            results.append({
                "title": row["title"],
                "language": row["language"],
                "genres": row["genres"],
                "hero": row["hero"],
                "heroine": row["heroine"],
            })
        return results

    def recommend_by_mood(self, query: str, top_n: int = 5):
        """
        Recommend movies based on a mood word (e.g. "happy", "scared") or a
        genre name typed directly (e.g. "comedy", "action").

        Returns None if the query isn't a recognized mood or genre, so the
        caller can decide how to handle that (e.g. show an error message).
        """
        query = query.lower().strip()
        all_genres = self.list_genres()
        genre_lookup = {g.lower(): g for g in all_genres}

        # 1. Direct genre match (e.g. user typed "comedy")
        if query in genre_lookup:
            target_genres = [genre_lookup[query]]
        # 2. Mood word match (e.g. user typed "happy" -> Comedy)
        elif query in MOOD_TO_GENRES:
            target_genres = MOOD_TO_GENRES[query]
        else:
            # 3. Fall back to scanning individual words in a phrase, so
            #    things like "I want comedy movies now" or "feeling scared"
            #    still work, not just single bare words.
            target_genres = []
            for word in query.split():
                if word in genre_lookup:
                    target_genres.append(genre_lookup[word])
                elif word in MOOD_TO_GENRES:
                    target_genres.extend(MOOD_TO_GENRES[word])
            if not target_genres:
                return None
            target_genres = list(dict.fromkeys(target_genres))  # dedupe, keep order

        # Score each movie by how many of its genres match the target genres
        def match_count(genre_str):
            movie_genres = set(genre_str.split("|"))
            return len(movie_genres.intersection(target_genres))

        scored = []
        for _, row in self.df.iterrows():
            count = match_count(row["genres"])
            if count > 0:
                scored.append((row, count))

        # Sort by match strength; shuffle within ties so results aren't static
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for row, count in scored[:top_n]:
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "hero": row["hero"],
                "heroine": row["heroine"],
            })
        return results

    def recommend_by_actor(self, name: str, top_n: int = 10):
        """
        Return movies starring a given hero or heroine (partial, case-insensitive
        match — e.g. "shah rukh" matches "Shah Rukh Khan"). Returns None if
        no movie in the dataset features that actor.
        """
        name = name.lower().strip()
        subset = self.df[
            self.df["hero"].str.lower().str.contains(name, na=False)
            | self.df["heroine"].str.lower().str.contains(name, na=False)
        ]
        if subset.empty:
            return None

        results = []
        for _, row in subset.head(top_n).iterrows():
            results.append({
                "title": row["title"],
                "language": row["language"],
                "genres": row["genres"],
                "hero": row["hero"],
                "heroine": row["heroine"],
            })
        return results

    def recommend(self, title: str, top_n: int = 5, allow_partial: bool = True):
        """
        Given a movie title, return the top_n most similar movies.
        Returns a list of dicts: [{"title": ..., "genres": ..., "score": ...}, ...]

        If allow_partial=False, only exact (case-insensitive) title matches
        are accepted — no substring guessing. Useful when you want to check
        "is this definitely a title?" before falling back to something else.
        """
        key = title.lower().strip()
        if key not in self.title_to_index:
            if not allow_partial:
                raise ValueError(f"'{title}' not found in dataset. Try list_titles().")
            # Try a loose partial match so users don't need exact titles
            matches = [t for t in self.df["title"] if key in t.lower()]
            if not matches:
                raise ValueError(f"'{title}' not found in dataset. Try list_titles().")
            key = matches[0].lower()

        idx = self.title_to_index[key]

        # Get similarity scores of this movie against every other movie
        scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by score, descending, and drop the movie itself (always score 1.0)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        results = []
        for movie_idx, score in scores:
            row = self.df.iloc[movie_idx]
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "score": round(float(score), 3),
                "hero": row["hero"],
                "heroine": row["heroine"],
            })
        return results


if __name__ == "__main__":
    # Quick smoke test when running this file directly
    rec = MovieRecommender("movies.csv")
    print("Recommendations for 'The Matrix':")
    for movie in rec.recommend("The Matrix"):
        print(f"  {movie['title']}  (score={movie['score']}, genres={movie['genres']})")
