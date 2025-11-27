import sys
import os
from tabulate import tabulate
import lucene_search

# Add src to path to import search.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from search import RecipeSearchEngine
except ImportError:
    print("Error: Could not import RecipeSearchEngine from search.py")
    sys.exit(1)

QUERIES = [
    "pasta",
    "chicken dinner",
    "gluten free",
    "chocolate cake",
    "vegan breakfast",
    "spicy curry",
    "italian cuisine",
    "quick snack",
    "fish",
    "soup",
    "pasta 10-20min",
    "chicken 1-2hr",
    "soup 30-45 min",
]


def run_old_search(engine, query):
    try:
        results = engine.search(query, top_k=3, idf_method="robertson")
        return [
            {
                "rank": i + 1,
                "title": r.title,
                "score": r.score,
                "prep_time": r.prep_time,
            }
            for i, r in enumerate(results)
        ]
    except Exception as e:
        return [{"error": str(e)}]


def run_new_search(query):
    try:
        results_dict = lucene_search.search_recipes(query, max_results=3)
        if "error" in results_dict:
            return [{"error": results_dict["error"]}]

        results = results_dict.get("results", [])
        return [
            {
                "rank": r["rank"],
                "title": r["recipe_title"],
                "score": r["score"],
                "prep_time": r.get("prep_time", ""),
            }
            for r in results[:3]
        ]
    except Exception as e:
        return [{"error": str(e)}]


def main():
    print("Initializing Old Search Engine...")
    old_engine = RecipeSearchEngine()

    print("Initializing New Search Engine (PyLucene)...")
    index_dir = os.getenv("INDEX_DIR", "/app/index")
    lucene_search.init_searcher(index_dir)

    print("\n" + "=" * 100)
    print(f"{'COMPARISON RESULTS':^100}")
    print("=" * 100 + "\n")

    for query in QUERIES:
        print(f"Query: '{query}'")
        print("-" * 100)

        old_results = run_old_search(old_engine, query)
        new_results = run_new_search(query)

        # Prepare table data
        table_data = []
        max_len = max(len(old_results), len(new_results))

        for i in range(max_len):
            old_row = old_results[i] if i < len(old_results) else {}
            new_row = new_results[i] if i < len(new_results) else {}

            if "error" not in old_row:
                old_pt = old_row.get("prep_time", "")
                old_pt_str = f" ({old_pt})" if old_pt else ""
                old_str = f"{old_row.get('rank', '-')}. {old_row.get('title', '---')}{old_pt_str} ({old_row.get('score', 0):.4f})"
            else:
                old_str = f"Error: {old_row['error']}"

            if "error" not in new_row:
                new_pt = new_row.get("prep_time", "")
                new_pt_str = f" ({new_pt})" if new_pt else ""
                new_str = f"{new_row.get('rank', '-')}. {new_row.get('title', '---')}{new_pt_str} ({new_row.get('score', 0):.4f})"
            else:
                new_str = f"Error: {new_row['error']}"

            table_data.append([old_str, new_str])

        print(
            tabulate(
                table_data,
                headers=["Old Search (Spark/TF-IDF)", "New Search (PyLucene)"],
                tablefmt="grid",
            )
        )
        print("\n")


if __name__ == "__main__":
    main()
