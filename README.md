# Content-Based Movie Recommendation System

A beginner-friendly recommender that suggests movies similar to one you like,
based on genres and description — no user ratings required.

## How it works

1. **TF-IDF Vectorization** — Each movie's genres + description are turned
   into a numeric vector. Words that are distinctive to a movie (like "wizard"
   or "hitman") get more weight than common words.
2. **Cosine Similarity** — We compare every movie's vector to every other
   movie's vector. A score close to 1 means "very similar," close to 0 means
   "very different."
3. **Recommendation** — Given a movie title, we look up its similarity scores
   against all other movies and return the highest-scoring ones.

## Files

- `movies.csv` — the dataset (45 sample movies with genres + descriptions)
- `recommender.py` — the core engine (`MovieRecommender` class)
- `main.py` — an interactive command-line interface to try it out

## Running it

```bash
pip install pandas scikit-learn
python main.py
```

Then type a movie title (e.g. `Inception`) and press Enter.

## Extending this into a bigger project (next steps)

1. **Use the real MovieLens dataset.** Download it from
   https://grouplens.org/datasets/movielens/ (the "latest small" 100k version
   is a good starting size). Replace `movies.csv` and adjust column names in
   `recommender.py` if needed — this instantly takes you from 45 movies to
   thousands.
2. **Add a web interface.** Wrap `MovieRecommender` in a small Flask or
   Streamlit app so people can search in a browser instead of a terminal.
3. **Add collaborative filtering.** Once you're comfortable, combine this
   content-based approach with a ratings-based approach (using the MovieLens
   `ratings.csv`) to build a *hybrid* recommender — this is what production
   systems like Netflix actually use.
4. **Improve the text features.** Right now we only use genres + a one-line
   description. Try pulling in actual plot summaries, cast/crew, or keywords
   from a source like TMDB's API for richer recommendations.
5. **Evaluate it.** Pick a few movies you know well, run recommendations, and
   sanity-check whether the results make sense. This is basically how
   real recommender systems get evaluated in early stages.

## Why this is a good portfolio project

It touches core ML/data science skills — text vectorization, similarity
metrics, and turning unstructured data into something a model can reason
about — without needing a GPU or a huge dataset to get started.
