from recommender import MovieRecommender


def main():
    rec = MovieRecommender("movies.csv")

    print("=" * 55)
    print(" CONTENT-BASED MOVIE RECOMMENDER")
    print("=" * 55)
    print(f"Loaded {len(rec.list_titles())} movies.\n")
    print("Type a MOVIE TITLE (e.g. 'Inception') for similar movies,")
    print("a MOOD/GENRE (e.g. 'comedy', 'happy', 'scared') for a themed pick,")
    print("a LANGUAGE, optionally with a genre (e.g. 'telugu', 'hindi comedy'),")
    print("or an ACTOR/ACTRESS name (e.g. 'Mahesh Babu', 'Deepika Padukone').")
    print("Languages:", ", ".join(rec.list_languages()))
    print("Genres:", ", ".join(rec.list_genres()))
    print("Type 'list' to see all movies, or 'quit' to exit.\n")

    while True:
        user_input = input("You > ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "list":
            for t in rec.list_titles():
                print(f"  - {t}")
            print()
            continue

        def print_title_results(results, label):
            print(f"\nBecause you liked '{label}', you might enjoy:")
            for movie in results:
                cast = f"{movie['hero']} / {movie['heroine']}"
                print(f"  {movie['title']:35s} | {movie['genres']:30s} | {cast:40s} | similarity={movie['score']}")
            print()

        def print_mood_results(results, label):
            print(f"\nFeeling '{label}'? Here are some picks:")
            for movie in results:
                cast = f"{movie['hero']} / {movie['heroine']}"
                print(f"  {movie['title']:35s} | {movie['genres']:30s} | {cast:40s}")
            print()

        def print_language_results(results, label):
            print(f"\nHere are some '{label}' picks:")
            for movie in results:
                cast = f"{movie['hero']} / {movie['heroine']}"
                print(f"  {movie['title']:35s} | {movie['language']:8s} | {movie['genres']:30s} | {cast:40s}")
            print()

        def print_actor_results(results, label):
            print(f"\nMovies featuring '{label}':")
            for movie in results:
                cast = f"{movie['hero']} / {movie['heroine']}"
                print(f"  {movie['title']:35s} | {movie['language']:8s} | {movie['genres']:30s} | {cast:40s}")
            print()

        # 1. Exact movie title match (highest confidence)
        try:
            results = rec.recommend(user_input, top_n=5, allow_partial=False)
            print_title_results(results, user_input)
            continue
        except ValueError:
            pass

        # 2. Language match, e.g. "telugu", "hindi comedy", "english movies"
        #    Scan words: if any word is a known language, treat the rest of
        #    the phrase as an optional genre filter.
        words = user_input.lower().split()
        lang_names = [l.lower() for l in rec.list_languages()]
        genre_names = [g.lower() for g in rec.list_genres()]
        found_language = next((w for w in words if w in lang_names), None)
        if found_language:
            found_genre = next((w for w in words if w in genre_names), None)
            lang_results = rec.recommend_by_language(found_language, genre=found_genre, top_n=8)
            if lang_results:
                print_language_results(lang_results, user_input)
                continue

        # 3. Actor/actress name match (e.g. "Mahesh Babu", "Deepika Padukone")
        actor_results = rec.recommend_by_actor(user_input, top_n=8)
        if actor_results:
            print_actor_results(actor_results, user_input)
            continue

        # 4. Recognized mood or genre word (also high confidence, and should
        #    win over a loose/accidental title substring match — e.g. "happy"
        #    shouldn't get swallowed by "The Pursuit of Happyness")
        mood_results = rec.recommend_by_mood(user_input, top_n=5)
        if mood_results:
            print_mood_results(mood_results, user_input)
            continue

        # 5. Last resort: fuzzy/partial title match
        try:
            results = rec.recommend(user_input, top_n=5, allow_partial=True)
            print_title_results(results, user_input)
            continue
        except ValueError:
            pass

        # 6. Nothing matched
        print(f"  '{user_input}' isn't a movie title, mood, genre, language, or actor I recognize.")
        print(f"  Try a title, a language ({', '.join(rec.list_languages())}), a genre, or an actor's name.\n")


if __name__ == "__main__":
    main()
