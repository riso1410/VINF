"""
Recipe Search Engine with BM25 and TF-IDF Search Methods

Features:
- BM25 fulltext search algorithm
- TF-IDF cosine similarity search algorithm
"""

import os
import json
import math
import re
from typing import Dict, List, Set
from dataclasses import dataclass

import config


@dataclass
class SearchResult:
    """Represents a single search result"""
    doc_id: str
    url: str
    title: str
    score: float
    chef: str = ""
    difficulty: str = ""
    prep_time: str = ""
    servings: str = ""
    word_count: int = 0
    
    def __repr__(self):
        return f"SearchResult(title='{self.title}', score={self.score:.4f}, url='{self.url}')"


class RecipeSearchEngine:
    """
    Search engine with BM25 and TF-IDF search algorithms.
    """
    
    def __init__(self, index_dir: str = None, log_level: int = None):
        """
        Initialize the search engine.
        
        Args:
            index_dir: Directory containing index files
            log_level: Logging level
        """
        self.index_dir = index_dir if index_dir is not None else config.INDEX_DIR
        log_level = log_level if log_level is not None else config.LOG_LEVEL
        
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        self.logger = config.setup_logging(config.SEARCH_LOG, log_level)
        
        # Index data structures
        self.inverted_index: Dict[str, Dict] = {}  # term -> {df, postings}
        self.document_stats: Dict[str, Dict] = {}  # doc_id -> stats
        self.total_documents: int = 0
        self.vocabulary_size: int = 0
        self.avg_doc_length: float = 0.0
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
        
        # Load index
        self._load_index()
        
        self.logger.info(f"Search engine initialized with {self.total_documents} documents and {self.vocabulary_size} terms")
    
    def _load_index(self):
        """Load the inverted index and document statistics from disk."""
        self.logger.info("Loading search index...")
        
        # Load metadata
        metadata_path = os.path.join(self.index_dir, "metadata.jsonl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(f.readline())
                self.total_documents = metadata.get("total_documents", 0)
                self.vocabulary_size = metadata.get("vocabulary_size", 0)
        
        # Load document stats
        mapping_path = os.path.join(self.index_dir, "mapping.jsonl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    self.document_stats[doc["doc_id"]] = doc
        
        # Calculate average document length
        if self.document_stats:
            total_length = sum(doc["word_count"] for doc in self.document_stats.values())
            self.avg_doc_length = total_length / len(self.document_stats)
        
        # Load inverted index
        index_path = os.path.join(self.index_dir, "index.jsonl")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.inverted_index[entry["term"]] = {
                        "df": entry["document_frequency"],
                        "postings": entry["postings"]
                    }
        
        self.logger.info(f"Loaded {len(self.inverted_index)} terms and {len(self.document_stats)} documents")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for searching."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and filter query text."""
        if not text:
            return []
        normalized = self._normalize_text(text)
        tokens = normalized.split()
        return [
            token for token in tokens
            if config.MIN_WORD_LENGTH <= len(token) <= config.MAX_WORD_LENGTH
            and token not in config.STOP_WORDS
            and not token.isdigit()
        ]
    
    def _calculate_bm25_score(self, term: str, doc_id: str, query_terms: List[str]) -> float:
        """
        Calculate BM25 score for a term in a document.
        
        BM25 Formula:
        score = IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        
        Args:
            term: Search term
            doc_id: Document ID
            query_terms: All query terms for IDF calculation
            
        Returns:
            BM25 score
        """
        if term not in self.inverted_index:
            return 0.0
        
        # Get term frequency in document
        postings = self.inverted_index[term]["postings"]
        if doc_id not in postings:
            return 0.0
        
        tf = postings[doc_id]
        
        # Get document frequency and calculate IDF
        df = self.inverted_index[term]["df"]
        idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1.0)
        
        # Get document length
        doc_length = self.document_stats[doc_id]["word_count"]
        
        # Calculate BM25 score
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
        
        score = idf * (numerator / denominator)
        return score
    
    def search_bm25(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[SearchResult]:
        """
        Perform BM25 fulltext search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects ranked by BM25 score
        """        
        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            self.logger.warning("Query resulted in no valid terms after tokenization")
            return []
        
        self.logger.info(f"Searching for query: '{query}' (tokens: {query_terms})")
        
        # Find candidate documents (documents containing at least one query term)
        candidate_docs: Set[str] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term]["postings"].keys())
        
        if not candidate_docs:
            self.logger.info("No documents found matching query terms")
            return []
        
        # Calculate BM25 scores for all candidate documents
        doc_scores: Dict[str, float] = {}
        for doc_id in candidate_docs:
            score = sum(self._calculate_bm25_score(term, doc_id, query_terms) for term in query_terms)
            doc_scores[doc_id] = score
        
        # Sort by score and get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create SearchResult objects
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
        
        self.logger.info(f"Found {len(results)} results for query '{query}'")
        return results
    
    def search_tfidf(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[SearchResult]:
        """
        Perform TF-IDF search using raw TF-IDF scoring.
        
        This method:
        1. Builds TF-IDF vectors for query and documents
        2. Calculates dot product (sum of TF-IDF scores) for ranking
        3. Ranks results based on cumulative TF-IDF scores (unbounded)
        
        Unlike cosine similarity (which is capped at 1.0), this method returns
        raw TF-IDF scores that can be higher, similar to BM25 scoring behavior.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects ranked by TF-IDF score
        """
        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            self.logger.warning("Query resulted in no valid terms after tokenization")
            return []
        
        self.logger.info(f"Searching for query: '{query}' (tokens: {query_terms})")
        
        # Find candidate documents (documents containing at least one query term)
        candidate_docs: Set[str] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term]["postings"].keys())
        
        if not candidate_docs:
            self.logger.info("No documents found matching query terms")
            return []
        
        # Calculate TF-IDF scores for each candidate document
        doc_scores: Dict[str, float] = {}
        for doc_id in candidate_docs:
            score = 0.0
            
            # For each query term, calculate its contribution to the score
            for term in query_terms:
                if term not in self.inverted_index:
                    continue
                
                postings = self.inverted_index[term]["postings"]
                if doc_id not in postings:
                    continue
                
                # Get term frequency in document
                tf = postings[doc_id]
                
                # Calculate IDF
                df = self.inverted_index[term]["df"]
                idf = math.log((self.total_documents + 1) / (df + 1)) + 1.0
                
                # TF-IDF score contribution: tf * idf
                # Using logarithmic TF normalization for better results
                tf_normalized = 1.0 + math.log(tf) if tf > 0 else 0.0
                score += tf_normalized * idf
            
            if score > 0:
                doc_scores[doc_id] = score
        
        # Sort by score and get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create SearchResult objects
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
        
        self.logger.info(f"Found {len(results)} results for query '{query}'")
        return results
    
    def search(self, query: str, method: str = 'bm25', top_k: int = None) -> List[SearchResult]:
        """
        Perform search using specified method.
        
        Args:
            query: Search query string
            method: Search method ('bm25' or 'tfidf')
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
        if method.lower() == 'tfidf':
            return self.search_tfidf(query, top_k=top_k)
        else:
            return self.search_bm25(query, top_k=top_k)
    
    def display_results(self, results: List[SearchResult], show_details: bool = True):
        """
        Display search results in a formatted way.
        
        Args:
            results: List of search results
            show_details: Whether to show detailed information
        """
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
    """Main function demonstrating search capabilities."""
    
    # Initialize search engine
    print("Initializing Recipe Search Engine...")
    search_engine = RecipeSearchEngine()
    
    # Example 1: BM25 search
    print("\n" + "="*80)
    print("Example 1: BM25 Fulltext Search")
    print("="*80)
    query = "chicken pasta garlic"
    results = search_engine.search_bm25(query, top_k=5)
    search_engine.display_results(results)
    
    # Example 2: TF-IDF search
    print("\n" + "="*80)
    print("Example 2: TF-IDF Cosine Similarity Search")
    print("="*80)
    query = "chocolate cake dessert"
    results = search_engine.search_tfidf(query, top_k=5)
    search_engine.display_results(results)
    
    # Interactive search
    print("\n" + "="*80)
    print("Interactive Search Mode")
    print("="*80)
    print("\nSelect search method:")
    print("  1. BM25 (Best Match 25)")
    print("  2. TF-IDF (Cosine Similarity)")
    print()
    
    # Get search method selection
    method_choice = input("Choose search method (1-2): ").strip()
    
    search_method = 'bm25'
    if method_choice == '1':
        search_method = 'bm25'
        print("\nUsing: BM25")
    elif method_choice == '2':
        search_method = 'tfidf'
        print("\nUsing: TF-IDF Cosine Similarity")
    else:
        search_method = 'bm25'
        print("\nInvalid choice. Using: BM25")
    
    print("\nEnter search queries (or 'quit' to exit, 'change' to change search method)")
    print()
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if not query or query.lower() == 'quit':
                break
            
            if query.lower() == 'change':
                print("\nSelect search method:")
                print("  1. BM25 (Best Match 25)")
                print("  2. TF-IDF (Cosine Similarity)")
                print()
                
                method_choice = input("Choose search method (1-2): ").strip()
                
                if method_choice == '1':
                    search_method = 'bm25'
                    print("\nUsing: BM25")
                elif method_choice == '2':
                    search_method = 'tfidf'
                    print("\nUsing: TF-IDF Cosine Similarity")
                else:
                    search_method = 'bm25'
                    print("\nInvalid choice. Using: BM25")
                print()
                continue
            
            # Perform search with selected method
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
