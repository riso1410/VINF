import os
import sys

import lucene
from java.nio.file import Paths as JavaPaths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import IntPoint
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import BoostQuery, IndexSearcher
from org.apache.lucene.store import MMapDirectory

SEARCH_FIELDS = [
    "recipe_title",
    "wiki_title",
    "ingredients",
    "description",
    "wiki_description",
    "method",
    "chef",
    "wiki_course",
    "wiki_origin",
    "wiki_serving_temp",
    "wiki_categories",
]

FIELD_BOOSTS = {
    "recipe_title": 3.0,
    "wiki_title": 2.0,
    "ingredients": 2.5,
    "description": 1.5,
    "wiki_description": 1.0,
    "method": 1.0,
    "chef": 0.5,
    "wiki_course": 1.0,
    "wiki_origin": 1.5,
    "wiki_serving_temp": 0.5,
    "wiki_categories": 1.0,
}

searcher = None
analyzer = None


class CustomQueryParser(QueryParser):
    def getRangeQuery(self, field, part1, part2, startInclusive, endInclusive):
        if field == "prep_time":
            try:
                lower = int(part1) if part1 else 0
                upper = int(part2) if part2 else 2147483647
                return IntPoint.newRangeQuery(field, lower, upper)
            except ValueError:
                pass
        return super().getRangeQuery(field, part1, part2, startInclusive, endInclusive)


# Initialize Lucene searcher
def init_searcher(index_dir):
    global searcher, analyzer

    lucene.initVM(vmargs=["-Djava.awt.headless=true"])

    sys.stderr.write(f"Opening index from: {index_dir}\n")

    store = MMapDirectory(JavaPaths.get(index_dir))
    reader = DirectoryReader.open(store)
    searcher = IndexSearcher(reader)
    analyzer = StandardAnalyzer()

    sys.stderr.write(f"Index loaded: {reader.numDocs():,} documents\n")
    return searcher


# Execute multi-field Lucene search
def search_recipes(query_text, max_results=10, fields=None):
    if not searcher:
        return {"error": "Index not initialized"}

    if fields is None:
        fields = SEARCH_FIELDS

    try:
        from org.apache.lucene.search import BooleanClause, BooleanQuery
        import re

        time_query = None

        # Check for minute ranges: 10-20min, 10-20 min
        min_match = re.search(r"(\d+)-(\d+)\s*min", query_text, re.IGNORECASE)
        if min_match:
            lower = int(min_match.group(1))
            upper = int(min_match.group(2))
            time_query = IntPoint.newRangeQuery("prep_time", lower, upper)
            # Remove the time part from the text query
            query_text = re.sub(
                r"(\d+)-(\d+)\s*min", "", query_text, flags=re.IGNORECASE
            ).strip()

        # Check for hour ranges: 1-2hr, 1-2 hours
        if not time_query:
            hr_match = re.search(
                r"(\d+)-(\d+)\s*(?:hr|hour|hours)", query_text, re.IGNORECASE
            )
            if hr_match:
                lower = int(hr_match.group(1)) * 60
                upper = int(hr_match.group(2)) * 60
                time_query = IntPoint.newRangeQuery("prep_time", lower, upper)
                # Remove the time part from the text query
                query_text = re.sub(
                    r"(\d+)-(\d+)\s*(?:hr|hour|hours)",
                    "",
                    query_text,
                    flags=re.IGNORECASE,
                ).strip()

        builder = BooleanQuery.Builder()

        # Add the time range query if found
        if time_query:
            builder.add(time_query, BooleanClause.Occur.MUST)

        # Use AND as default operator for all searches
        if query_text:
            for field in fields:
                field_parser = CustomQueryParser(field, analyzer)
                field_parser.setDefaultOperator(QueryParser.Operator.AND)
                try:
                    field_query = field_parser.parse(query_text)
                    boost = FIELD_BOOSTS.get(field, 1.0)

                    if boost != 1.0:
                        boosted_query = BoostQuery(field_query, boost)
                        builder.add(boosted_query, BooleanClause.Occur.SHOULD)
                    else:
                        builder.add(field_query, BooleanClause.Occur.SHOULD)
                except:
                    pass

        query = builder.build()

        hits = searcher.search(query, max_results)

        results = []
        for i, hit in enumerate(hits.scoreDocs):
            doc = searcher.storedFields().document(hit.doc)

            result = {
                "rank": i + 1,
                "score": float(hit.score),
                "recipe_url": doc.get("recipe_url"),
                "recipe_title": doc.get("recipe_title"),
                "wiki_title": doc.get("wiki_title"),
                "description": doc.get("description"),
                "wiki_description": doc.get("wiki_description"),
                "ingredients": doc.get("ingredients"),
                "chef": doc.get("chef"),
                "difficulty": doc.get("difficulty"),
                "prep_time": doc.get("prep_time"),
                "servings": doc.get("servings"),
                "wiki_url": doc.get("wiki_url"),
                "wiki_course": doc.get("wiki_course"),
                "wiki_origin": doc.get("wiki_origin"),
                "wiki_serving_temp": doc.get("wiki_serving_temp"),
                "wiki_categories": doc.get("wiki_categories"),
            }

            results.append(result)

        return {
            "query": query_text,
            "total_hits": hits.totalHits.value(),
            "results": results,
        }

    except Exception as e:
        return {"error": str(e)}


def batch_search():
    import json
    import sys

    # Read queries from stdin
    for line in sys.stdin:
        query = line.strip()
        if not query:
            continue

        results = search_recipes(query, max_results=10)
        print(json.dumps(results))
        sys.stdout.flush()


def interactive_search():
    print("\n" + "=" * 70)
    print("RECIPE SEARCH - Interactive Mode")
    print("=" * 70)
    print("\nEnter search queries (or 'quit' to exit)\n")

    while True:
        try:
            query = input("Search> ").strip()

            if not query or query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            results = search_recipes(query, max_results=10)

            if "error" in results:
                print(f"Error: {results['error']}\n")
                continue

            print(f"\nFound {results['total_hits']} results:\n")

            for result in results["results"]:
                desc = result.get("description", "") or ""
                if len(desc) > 150:
                    desc = desc[:150] + "..."

                wiki_desc = result.get("wiki_description", "") or ""
                if len(wiki_desc) > 150:
                    wiki_desc = wiki_desc[:150] + "..."

                print(f"{result['rank']}. {result['score']:.2f}")
                print(f"{result['recipe_title']}")
                print(f"{result['recipe_url']}")
                print(f"{desc}")
                print("---")
                print(f"{result['wiki_title']}")
                print(f"{result['wiki_url']}")
                print(f"{wiki_desc}")
                print(f"{result['wiki_course']}")
                print(f"{result['wiki_origin']}")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    index_dir = os.getenv("INDEX_DIR", "/app/index")

    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print(f"Error: Index directory not found or empty: {index_dir}")
        print("Please run the indexer first.")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        init_searcher(index_dir)
        batch_search()
    else:
        init_searcher(index_dir)
        interactive_search()


if __name__ == "__main__":
    main()
