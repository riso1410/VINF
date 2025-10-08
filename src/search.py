import os
import json
import math
import re
from typing import Dict, List, Set
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

        self.inverted_index: Dict[str, Dict] = {}
        self.document_stats: Dict[str, Dict] = {}
        self.total_documents: int = 0
        self.vocabulary_size: int = 0
        
        self.load_index()
            
    def load_index(self):
        metadata_path = os.path.join(self.index_dir, "metadata.jsonl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(f.readline())
                self.total_documents = metadata.get("total_documents", 0)
                self.vocabulary_size = metadata.get("vocabulary_size", 0)
        
        mapping_path = os.path.join(self.index_dir, "mapping.jsonl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    self.document_stats[doc["doc_id"]] = doc
        
        index_path = os.path.join(self.index_dir, "index.jsonl")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.inverted_index[entry["term"]] = {
                        "df": entry["document_frequency"],
                        "postings": entry["postings"]
                    }
    
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        return [
            token for token in tokens
            if config.MIN_WORD_LENGTH <= len(token) <= config.MAX_WORD_LENGTH
            and token not in config.STOP_WORDS
            and not token.isdigit()
        ]
    
    def calculate_bm25_score(self, term: str, doc_id: str) -> float:
        if term not in self.inverted_index:
            return 0.0
        
        postings = self.inverted_index[term]["postings"]
        if doc_id not in postings:
            return 0.0
        
        tf = postings[doc_id]
        df = self.inverted_index[term]["df"]
        
        score = math.log((self.total_documents - df + 0.5) / (df + 0.5 + tf))
        return score
    
    def search_bm25(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[SearchResult]:
        query_terms = self.tokenize(query)
        if not query_terms:
            return []
        
        candidate_docs: Set[str] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term]["postings"].keys())
        
        if not candidate_docs:
            return []
        
        doc_scores: Dict[str, float] = {}
        for doc_id in candidate_docs:
            score = sum(self.calculate_bm25_score(term, doc_id) for term in query_terms)
            doc_scores[doc_id] = score
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            doc_info = self.document_stats[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                url=doc_info["url"],
                title=doc_info["title"],
                score=score,
                chef=doc_info.get("chef", ""),
                difficulty=doc_info.get("difficulty", ""),
                prep_time=doc_info.get("prep_time", ""),
                servings=doc_info.get("servings", ""),
                word_count=doc_info.get("word_count", 0)
            ))
        
        return results
    
    def search_tfidf(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[SearchResult]:
        query_terms = self.tokenize(query)
        if not query_terms:
            return []
        
        candidate_docs: Set[str] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term]["postings"].keys())
        
        if not candidate_docs:
            return []
        
        doc_scores: Dict[str, float] = {}
        for doc_id in candidate_docs:
            score = 0.0
            
            for term in query_terms:
                if term not in self.inverted_index:
                    continue
                
                postings = self.inverted_index[term]["postings"]
                if doc_id not in postings:
                    continue
                
                tf = postings[doc_id]
                df = self.inverted_index[term]["df"]
                
                score += math.log(self.total_documents / (df + tf))
            
            if score > 0:
                doc_scores[doc_id] = score
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            doc_info = self.document_stats[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                url=doc_info["url"],
                title=doc_info["title"],
                score=score,
                chef=doc_info.get("chef", ""),
                difficulty=doc_info.get("difficulty", ""),
                prep_time=doc_info.get("prep_time", ""),
                servings=doc_info.get("servings", "")
            ))
        
        return results
    
    def search(self, query: str, method: str = 'bm25', top_k: int = None) -> List[SearchResult]:
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
        if method.lower() == 'tfidf':
            return self.search_tfidf(query, top_k=top_k)
        else:
            return self.search_bm25(query, top_k=top_k)
    
    def display_results(self, results: List[SearchResult], show_details: bool = True):
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
    
    print("\nSelect search method:")
    print("  1. BM25 (Robertson IDF)")
    print("  2. TF-IDF (Classic IDF)")
    print()
    
    method_choice = input("Choose search method (1-2): ").strip()
    
    if method_choice == '1':
        search_method = 'bm25'
        print("\nUsing: BM25 (Robertson IDF)")
    elif method_choice == '2':
        search_method = 'tfidf'
        print("\nUsing: TF-IDF (Classic IDF)")
    else:
        search_method = 'bm25'
        print("\nInvalid choice. Using: BM25 (Robertson IDF)")
    
    print("\nEnter search queries (or 'quit' to exit, 'change' to change search method)")
    print()
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if not query or query.lower() == 'quit':
                break
            
            if query.lower() == 'change':
                print("\nSelect search method:")
                print("  1. BM25 (Robertson IDF)")
                print("  2. TF-IDF (Classic IDF)")
                print()
                
                method_choice = input("Choose search method (1-2): ").strip()
                
                if method_choice == '1':
                    search_method = 'bm25'
                    print("\nUsing: BM25 (Robertson IDF)")
                elif method_choice == '2':
                    search_method = 'tfidf'
                    print("\nUsing: TF-IDF (Classic IDF)")
                else:
                    search_method = 'bm25'
                    print("\nInvalid choice. Using: BM25 (Robertson IDF)")
                print()
                continue
            
            results = search_engine.search(query, method=search_method, top_k=config.DEFAULT_TOP_K)
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
