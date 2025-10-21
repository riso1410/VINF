import json
import math
import os
from dataclasses import dataclass

import config


@dataclass
class SearchResult:
    doc_id: str
    url: str
    title: str
    score: float
    chef: str = ""
    difficulty: str = ""
    prep_time: str = ""
    servings: str = ""


class RecipeSearchEngine:
    def __init__(self):
        self.index_dir = config.INDEX_DIR

        self.inverted_index: dict[str, dict] = {}
        self.document_stats: dict[str, dict] = {}
        self.total_documents: int = 0
        self.vocabulary_size: int = 0

        self.load_index()

    def load_index(self):
        metadata_path = os.path.join(self.index_dir, "stats.jsonl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.loads(f.readline())
                self.total_documents = metadata.get("total_documents", 0)
                self.vocabulary_size = metadata.get("vocabulary_size", 0)

        mapping_path = os.path.join(self.index_dir, "mapping.jsonl")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    self.document_stats[doc["doc_id"]] = doc

        index_path = os.path.join(self.index_dir, "index.jsonl")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.inverted_index[entry["term"]] = {
                        "df": entry["document_frequency"],
                        "postings": entry["postings"],
                    }

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        text = text.lower()
        tokens = text.split()
        return [
            token
            for token in tokens
            if config.MIN_WORD_LENGTH <= len(token) <= config.MAX_WORD_LENGTH
            and token not in config.STOP_WORDS
        ]

    def idf_robertson(self, df: int) -> float:
        return math.log((self.total_documents - df + 0.5) / (df + 0.5))

    def idf_classic(self, df: int) -> float:
        return math.log(self.total_documents / (df))

    def search(
        self, query: str, top_k: int, idf_method: str = "robertson"
    ) -> list[SearchResult]:
        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        candidate_docs: set[str] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term]["postings"].keys())

        if not candidate_docs:
            return []

        doc_scores: dict[str, float] = {}
        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                if term in self.inverted_index:
                    postings = self.inverted_index[term]["postings"]
                    if doc_id in postings:
                        tf = postings[doc_id]
                        df = self.inverted_index[term]["df"]
                        if idf_method == "classic":
                            idf = self.idf_classic(df)
                        else:
                            idf = self.idf_robertson(df)
                        score += tf * idf
            if score > 0:
                doc_scores[doc_id] = score

        if not doc_scores:
            return []

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results = []
        for doc_id, score in sorted_docs:
            doc_info = self.document_stats.get(doc_id)
            if doc_info:
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        url=doc_info["url"],
                        title=doc_info["title"],
                        score=score,
                        chef=doc_info.get("chef", ""),
                        difficulty=doc_info.get("difficulty", ""),
                        prep_time=doc_info.get("prep_time", ""),
                        servings=doc_info.get("servings", ""),
                    )
                )

        return results

    def display_results(self, results: list[SearchResult], show_details: bool = True):
        if not results:
            print("\nNo results found.")
            return

        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   Score: {result.score:.4f}")
            print(f"   URL: {result.url}")

            if show_details:
                details = []
                if result.chef:
                    details.append(f"Chef: {result.chef}")
                if result.difficulty:
                    details.append(f"Difficulty: {result.difficulty}")
                if result.prep_time:
                    details.append(f"Prep Time: {result.prep_time}")
                if result.servings:
                    details.append(f"Servings: {result.servings}")

                if details:
                    print(f"   {' | '.join(details)}")

            print()


def main():
    search_engine = RecipeSearchEngine()

    print("\nSelect IDF calculation method:")
    print("  1. Robertson (BM25-style)")
    print("  2. Classic (TF-IDF-style)")
    print()

    method_choice = input("Choose IDF method (1-2): ").strip()

    if method_choice == "1":
        idf_method = "robertson"
        print("\nUsing: Robertson (BM25-style) IDF")
    elif method_choice == "2":
        idf_method = "classic"
        print("\nUsing: Classic (TF-IDF-style) IDF")
    else:
        idf_method = "robertson"
        print("\nInvalid choice. Using: Robertson (BM25-style) IDF")

    print("\nEnter search queries (or 'quit' to exit, 'change' to change IDF method)")
    print()

    while True:
        try:
            query = input("Search query: ").strip()

            if not query or query.lower() == "quit":
                break

            if query.lower() == "change":
                print("\nSelect IDF calculation method:")
                print("  1. Robertson (BM25-style)")
                print("  2. Classic (TF-IDF-style)")
                print()

                method_choice = input("Choose IDF method (1-2): ").strip()

                if method_choice == "1":
                    idf_method = "robertson"
                    print("\nUsing: Robertson (BM25-style) IDF")
                elif method_choice == "2":
                    idf_method = "classic"
                    print("\nUsing: Classic (TF-IDF-style) IDF")
                else:
                    idf_method = "robertson"
                    print("\nInvalid choice. Using: Robertson (BM25-style) IDF")
                print()
                continue

            results = search_engine.search(
                query, top_k=config.DEFAULT_TOP_K, idf_method=idf_method
            )
            search_engine.display_results(results)

        except KeyboardInterrupt:
            print("\n\nSearch interrupted.")
            break
        except Exception as e:
            print(f"\nError during search: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
