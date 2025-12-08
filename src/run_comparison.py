import sys
import os
import time
import re
import urllib.parse
from tabulate import tabulate
import lucene_search
import requests
from bs4 import BeautifulSoup
import config
from Levenshtein import ratio

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
        start_time = time.time()
        results, total_hits = engine.search(query, top_k=5, idf_method="robertson")
        latency = time.time() - start_time

        return {
            "results": [
                {
                    "rank": i + 1,
                    "title": r.title,
                    "score": r.score,
                    "prep_time": r.prep_time,
                }
                for i, r in enumerate(results)
            ],
            "latency": latency,
            "total_hits": total_hits,
        }
    except Exception as e:
        return {"results": [{"error": str(e)}], "latency": 0, "total_hits": 0}


def run_new_search(query):
    try:
        start_time = time.time()
        results_dict = lucene_search.search_recipes(query, max_results=5)
        latency = time.time() - start_time

        if "error" in results_dict:
            return {
                "results": [{"error": results_dict["error"]}],
                "latency": latency,
                "total_hits": 0,
            }

        results = results_dict.get("results", [])
        total_hits = results_dict.get("total_hits", 0)
        return {
            "results": [
                {
                    "rank": r["rank"],
                    "title": r["recipe_title"],
                    "score": r["score"],
                    "prep_time": r.get("prep_time", ""),
                }
                for r in results[:5]
            ],
            "latency": latency,
            "total_hits": total_hits,
        }
    except Exception as e:
        return {"results": [{"error": str(e)}], "latency": 0, "total_hits": 0}


def normalize_text(text):
    """Normalize text for fuzzy matching - lowercase, remove punctuation, extra spaces"""
    import re

    if not text:
        return ""
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    """Tokenize text into words for token-based matching"""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def calculate_token_fuzzy_score(query, title):
    """
    Calculate fuzzy match score based on tokens and normalized text.
    Uses combination of:
    1. Full text Levenshtein similarity
    2. Token-level matching (how many query tokens appear in title)
    3. Best token-to-token Levenshtein for partial matches
    """
    query_normalized = normalize_text(query)
    title_normalized = normalize_text(title)

    if not query_normalized or not title_normalized:
        return 0.0

    # 1. Full text similarity (weight: 0.3)
    full_text_similarity = ratio(query_normalized, title_normalized)

    # 2. Token-level matching
    query_tokens = tokenize_text(query)
    title_tokens = tokenize_text(title)

    if not query_tokens or not title_tokens:
        return full_text_similarity

    # Count exact token matches (weight: 0.4)
    exact_matches = sum(1 for qt in query_tokens if qt in title_tokens)
    exact_match_ratio = exact_matches / len(query_tokens) if query_tokens else 0.0

    # 3. Best fuzzy token matches for non-exact matches (weight: 0.3)
    fuzzy_token_scores = []
    for qt in query_tokens:
        if qt not in title_tokens:
            # Find best fuzzy match for this query token
            best_match = max((ratio(qt, tt) for tt in title_tokens), default=0.0)
            fuzzy_token_scores.append(best_match)
        else:
            fuzzy_token_scores.append(1.0)  # Exact match

    avg_fuzzy_token_score = (
        sum(fuzzy_token_scores) / len(fuzzy_token_scores) if fuzzy_token_scores else 0.0
    )

    # Combine scores with weights
    combined_score = (
        0.3 * full_text_similarity
        + 0.4 * exact_match_ratio
        + 0.3 * avg_fuzzy_token_score
    )

    return combined_score


def run_foodnetwork_search(query):
    """Scrape FoodNetwork website search results"""
    try:
        start_time = time.time()
        encoded_query = urllib.parse.quote(query)
        url = f"{config.START_URL}search?q={encoded_query}&type=recipe"

        headers = {"User-Agent": config.CHROME_USER_AGENT}

        response = requests.get(
            url, headers=headers, timeout=config.SELENIUM_PAGE_LOAD_TIMEOUT
        )
        latency = time.time() - start_time

        if response.status_code != 200:
            return {
                "results": [{"error": f"HTTP {response.status_code}"}],
                "latency": latency,
                "total_hits": 0,
            }

        soup = BeautifulSoup(response.content, "html.parser")

        hit_count = 0
        hit_count_tag = soup.find("h4", attrs={"data-v-45435b46": True})
        if hit_count_tag:
            hit_text = hit_count_tag.get_text(strip=True)
            # Remove parentheses and extract number
            match = re.search(r'\((\d+)\)', hit_text)
            if match:
                hit_count = int(match.group(1))

        # Find recipe titles from search results
        recipe_titles = []

        recipe_cards = soup.find_all("a", class_="block group")

        for card in recipe_cards:
            # Check if it's a recipe link (usually starts with /recipes/)
            href = card.get("href", "")
            if "/recipes/" in href and "/collections/" not in href:
                title_tag = card.find("h3")
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    if title:
                        recipe_titles.append(title)
                        if len(recipe_titles) >= 5:
                            break

        # Calculate fuzzy match scores based on tokens and normalized text
        results = []
        for i, title in enumerate(recipe_titles[:5]):
            fuzzy_score = calculate_token_fuzzy_score(query, title)
            results.append(
                {
                    "rank": i + 1,
                    "title": title,
                    "score": fuzzy_score,
                    "prep_time": "",
                }
            )

        if not results:
            results = [{"error": "No results found"}]

        return {"results": results, "latency": latency, "total_hits": hit_count}

    except Exception as e:
        return {"results": [{"error": str(e)}], "latency": 0, "total_hits": 0}


def calculate_relevancy_score(all_results, query):
    """
    Calculate relevancy metrics by comparing each title to the query using token-based fuzzy matching:
    - Average relevancy score per search engine (how well titles match the query)
    - Overall relevancy comparison

    Uses normalized text and token-level Levenshtein matching for more accurate relevancy.
    """
    relevancy_scores = {}

    for search_name, search_data in all_results.items():
        results = search_data.get("results", [])
        scores = []

        for result in results:
            if "error" not in result and "title" in result:
                # Use token-based fuzzy matching for relevancy
                similarity = calculate_token_fuzzy_score(query, result["title"])
                scores.append(similarity)

        # Calculate average relevancy for this search engine
        if scores:
            avg_relevancy = sum(scores) / len(scores)
        else:
            avg_relevancy = 0.0

        relevancy_scores[search_name] = {
            "avg_relevancy": avg_relevancy,
            "scores": scores,
        }

    # Calculate overall average relevancy across all engines
    all_scores = []
    for engine_data in relevancy_scores.values():
        all_scores.extend(engine_data["scores"])

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {"relevancy_scores": relevancy_scores, "overall_avg_relevancy": overall_avg}


def main():
    print("Initializing Old Search Engine...")
    old_engine = RecipeSearchEngine()

    print("Initializing New Search Engine (PyLucene)...")
    index_dir = os.getenv("INDEX_DIR", "/app/index")
    lucene_search.init_searcher(index_dir)

    print("\n" + "=" * 150)
    print(f"{'SEARCH ENGINE COMPARISON RESULTS':^150}")
    print("=" * 150 + "\n")

    total_latencies = {
        "Spark/TF-IDF": [],
        "PyLucene": [],
        "FoodNetwork": [],
    }

    total_hits_data = []
    relevancy_data = []

    for query in QUERIES:
        print(f"\nQuery: '{query}'")
        print("-" * 150)

        # Run all searches
        old_results = run_old_search(old_engine, query)
        new_results = run_new_search(query)
        foodnetwork_results = run_foodnetwork_search(query)

        # Store latencies
        total_latencies["Spark/TF-IDF"].append(old_results.get("latency", 0))
        total_latencies["PyLucene"].append(new_results.get("latency", 0))
        total_latencies["FoodNetwork"].append(foodnetwork_results.get("latency", 0))

        # Store total hits
        total_hits_data.append(
            [
                query,
                old_results.get("total_hits", 0),
                new_results.get("total_hits", 0),
                foodnetwork_results.get("total_hits", "N/A"),
            ]
        )

        # Calculate relevancy
        all_results = {
            "Spark/TF-IDF": old_results,
            "PyLucene": new_results,
            "FoodNetwork": foodnetwork_results,
        }
        relevancy = calculate_relevancy_score(all_results, query)

        # Track relevancy data
        fn_valid = True
        if (
            "results" in foodnetwork_results
            and len(foodnetwork_results["results"]) == 1
            and "error" in foodnetwork_results["results"][0]
        ):
            fn_valid = False

        relevancy_data.append(
            {
                "query": query,
                "Spark/TF-IDF": relevancy["relevancy_scores"]["Spark/TF-IDF"][
                    "avg_relevancy"
                ],
                "PyLucene": relevancy["relevancy_scores"]["PyLucene"]["avg_relevancy"],
                "FoodNetwork": relevancy["relevancy_scores"]["FoodNetwork"][
                    "avg_relevancy"
                ],
                "fn_valid": fn_valid,
            }
        )

        # Prepare table data
        table_data = []
        max_len = max(
            len(old_results.get("results", [])),
            len(new_results.get("results", [])),
            len(foodnetwork_results.get("results", [])),
        )

        for i in range(max_len):
            old_row = (
                old_results.get("results", [])[i]
                if i < len(old_results.get("results", []))
                else {}
            )
            new_row = (
                new_results.get("results", [])[i]
                if i < len(new_results.get("results", []))
                else {}
            )
            fn_row = (
                foodnetwork_results.get("results", [])[i]
                if i < len(foodnetwork_results.get("results", []))
                else {}
            )

            # Format each column
            def format_result(row):
                if not row:
                    return "---"
                if "error" in row:
                    return f"Error: {row['error']}"
                pt = row.get("prep_time", "")
                pt_str = f" ({pt})" if pt else ""
                title = row.get("title", "---")
                # Truncate long titles
                if len(title) > 25:
                    title = title[:22] + "..."
                return f"{row.get('rank', '-')}. {title}{pt_str}\n({row.get('score', 0):.3f})"

            table_data.append(
                [
                    format_result(old_row),
                    format_result(new_row),
                    format_result(fn_row),
                ]
            )

        print(
            tabulate(
                table_data,
                headers=[
                    "Spark/TF-IDF",
                    "PyLucene",
                    "FoodNetwork Web",
                ],
                tablefmt="grid",
            )
        )

        # Print metrics
        print(f"\n📊 Metrics for '{query}':")
        print(
            f"  Latency: Spark={old_results.get('latency', 0):.3f}s | "
            f"PyLucene={new_results.get('latency', 0):.3f}s | "
            f"FoodNetwork={foodnetwork_results.get('latency', 0):.3f}s"
        )

        # Print relevancy scores
        rel_scores = relevancy["relevancy_scores"]
        print("  Query Relevancy (fuzzy match to query):")
        print(f"    Spark/TF-IDF: {rel_scores['Spark/TF-IDF']['avg_relevancy']:.2%}")
        print(f"    PyLucene: {rel_scores['PyLucene']['avg_relevancy']:.2%}")
        print(f"    FoodNetwork: {rel_scores['FoodNetwork']['avg_relevancy']:.2%}")
        print(f"    Overall Average: {relevancy['overall_avg_relevancy']:.2%}")
        print()

    # Print summary statistics
    print("\n" + "=" * 150)
    print(f"{'SUMMARY STATISTICS':^150}")
    print("=" * 150 + "\n")

    summary_data = []
    for engine, latencies in total_latencies.items():
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            summary_data.append(
                [
                    engine,
                    f"{avg_latency:.3f}s",
                    f"{min_latency:.3f}s",
                    f"{max_latency:.3f}s",
                ]
            )

    print(
        tabulate(
            summary_data,
            headers=["Search Engine", "Avg Latency", "Min Latency", "Max Latency"],
            tablefmt="grid",
        )
    )
    print()

    # Print total hits summary
    print("\n" + "=" * 150)
    print(f"{'TOTAL HITS SUMMARY':^150}")
    print("=" * 150 + "\n")

    print(
        tabulate(
            total_hits_data,
            headers=["Query", "Spark/TF-IDF Hits", "PyLucene Hits", "FoodNetwork Hits"],
            tablefmt="grid",
        )
    )
    print()

    # Calculate and print Relevancy Summaries

    # 1. Overall Relevancy
    overall_avgs = {
        "Spark/TF-IDF": sum(d["Spark/TF-IDF"] for d in relevancy_data)
        / len(relevancy_data)
        if relevancy_data
        else 0,
        "PyLucene": sum(d["PyLucene"] for d in relevancy_data) / len(relevancy_data)
        if relevancy_data
        else 0,
        "FoodNetwork": sum(d["FoodNetwork"] for d in relevancy_data)
        / len(relevancy_data)
        if relevancy_data
        else 0,
    }

    print("\n" + "=" * 150)
    print(f"{'OVERALL RELEVANCY SUMMARY':^150}")
    print("=" * 150 + "\n")

    print(
        tabulate(
            [
                [
                    f"{overall_avgs['Spark/TF-IDF']:.2%}",
                    f"{overall_avgs['PyLucene']:.2%}",
                    f"{overall_avgs['FoodNetwork']:.2%}",
                ]
            ],
            headers=["Spark/TF-IDF Avg", "PyLucene Avg", "FoodNetwork Avg"],
            tablefmt="grid",
        )
    )
    print()

    # 2. Valid FoodNetwork Relevancy (only queries where FN found results)
    valid_fn_data = [d for d in relevancy_data if d["fn_valid"]]

    if valid_fn_data:
        valid_avgs = {
            "Spark/TF-IDF": sum(d["Spark/TF-IDF"] for d in valid_fn_data)
            / len(valid_fn_data),
            "PyLucene": sum(d["PyLucene"] for d in valid_fn_data) / len(valid_fn_data),
            "FoodNetwork": sum(d["FoodNetwork"] for d in valid_fn_data)
            / len(valid_fn_data),
        }

        print("\n" + "=" * 150)
        print(f"{'VALID FOODNETWORK RELEVANCY SUMMARY (Queries with FN results)':^150}")
        print("=" * 150 + "\n")

        print(
            tabulate(
                [
                    [
                        f"{valid_avgs['Spark/TF-IDF']:.2%}",
                        f"{valid_avgs['PyLucene']:.2%}",
                        f"{valid_avgs['FoodNetwork']:.2%}",
                    ]
                ],
                headers=["Spark/TF-IDF Avg", "PyLucene Avg", "FoodNetwork Avg"],
                tablefmt="grid",
            )
        )
        print()
    else:
        print("\nNo valid FoodNetwork results found to calculate specific average.\n")


if __name__ == "__main__":
    main()
