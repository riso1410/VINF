"""
Recipe Search Engine with Fulltext Search and Reranking Algorithms

Features:
- BM25 fulltext search algorithm
- Two reranking methods:
  1. TF-IDF Cosine Similarity Reranking
  2. Metadata-based Personalized Reranking
"""

import os
import json
import math
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import Counter

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
    bm25_score: float = 0.0
    tfidf_score: float = 0.0
    metadata_score: float = 0.0
    
    def __repr__(self):
        return f"SearchResult(title='{self.title}', score={self.score:.4f}, url='{self.url}')"


class RecipeSearchEngine:
    """
    Advanced search engine with BM25 fulltext search and multiple reranking algorithms.
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
    
    def search_bm25(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Perform BM25 fulltext search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects ranked by BM25 score
        """
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
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
                bm25_score=score,
                chef=doc_info.get("chef", ""),
                difficulty=doc_info.get("difficulty", ""),
                prep_time=doc_info.get("prep_time", ""),
                servings=doc_info.get("servings", ""),
                word_count=doc_info.get("word_count", 0)
            ))
        
        self.logger.info(f"Found {len(results)} results for query '{query}'")
        return results
    
    def rerank_cosine_similarity(self, query: str, initial_results: List[SearchResult], 
                                 top_k: int = None) -> List[SearchResult]:
        """
        Rerank results using TF-IDF cosine similarity.
        
        This method:
        1. Builds TF-IDF vectors for query and documents
        2. Calculates cosine similarity between query and each document
        3. Re-scores results based on cosine similarity
        
        Args:
            query: Original search query
            initial_results: Initial search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked list of SearchResult objects
        """
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
        if not initial_results:
            return []
        
        self.logger.info(f"Reranking {len(initial_results)} results using TF-IDF cosine similarity")
        
        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return initial_results
        
        # Build query TF-IDF vector
        query_tf = Counter(query_terms)
        query_vector = {}
        
        for term in query_tf:
            if term not in self.inverted_index:
                continue
            
            tf = query_tf[term]
            df = self.inverted_index[term]["df"]
            idf = math.log(self.total_documents / (df + 1))
            query_vector[term] = tf * idf
        
        # Calculate cosine similarity for each document
        reranked_results = []
        for result in initial_results:
            doc_id = result.doc_id
            
            # Build document TF-IDF vector
            doc_vector = {}
            for term in query_vector.keys():
                if term in self.inverted_index:
                    postings = self.inverted_index[term]["postings"]
                    if doc_id in postings:
                        tf = postings[doc_id]
                        df = self.inverted_index[term]["df"]
                        idf = math.log(self.total_documents / (df + 1))
                        doc_vector[term] = tf * idf
            
            # Calculate cosine similarity
            if not doc_vector:
                cosine_score = 0.0
            else:
                dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
                query_norm = math.sqrt(sum(v**2 for v in query_vector.values()))
                doc_norm = math.sqrt(sum(v**2 for v in doc_vector.values()))
                
                if query_norm == 0 or doc_norm == 0:
                    cosine_score = 0.0
                else:
                    cosine_score = dot_product / (query_norm * doc_norm)
            
            # Create new result with cosine similarity score
            reranked_results.append(SearchResult(
                doc_id=result.doc_id,
                url=result.url,
                title=result.title,
                score=cosine_score,
                bm25_score=result.bm25_score,
                tfidf_score=cosine_score,
                metadata_score=result.metadata_score,
                chef=result.chef,
                difficulty=result.difficulty,
                prep_time=result.prep_time,
                servings=result.servings,
                word_count=result.word_count
            ))
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Reranking complete, returning top {top_k} results")
        return reranked_results[:top_k]
    
    def rerank_personalized(self, query: str, initial_results: List[SearchResult],
                           preferred_difficulty: Optional[str] = None,
                           preferred_chef: Optional[str] = None,
                           max_prep_time: Optional[int] = None,
                           top_k: int = None) -> List[SearchResult]:
        """
        Rerank results using personalized metadata-based scoring.
        
        This method adjusts scores based on user preferences:
        - Difficulty level preference
        - Chef preference
        - Maximum preparation time
        - Document length (shorter recipes ranked higher)
        
        Args:
            query: Original search query
            initial_results: Initial search results to rerank
            preferred_difficulty: Preferred difficulty level (e.g., "Easy", "Medium", "Hard")
            preferred_chef: Preferred chef name
            max_prep_time: Maximum preparation time in minutes
            top_k: Number of top results to return
            
        Returns:
            Reranked list of SearchResult objects
        """
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
        if not initial_results:
            return []
        
        self.logger.info(f"Reranking {len(initial_results)} results using personalized scoring")
        
        # Normalize initial scores to [0, 1]
        max_score = max(r.score for r in initial_results) if initial_results else 1.0
        min_score = min(r.score for r in initial_results) if initial_results else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        reranked_results = []
        for result in initial_results:
            # Start with normalized BM25 score
            normalized_score = (result.score - min_score) / score_range if score_range > 0 else 0.5
            
            # Initialize boost factors
            difficulty_boost = 1.0
            chef_boost = 1.0
            time_boost = 1.0
            length_boost = 1.0
            
            # Difficulty preference boost
            if preferred_difficulty and result.difficulty:
                if preferred_difficulty.lower() in result.difficulty.lower():
                    difficulty_boost = 1.3
                elif self._is_similar_difficulty(preferred_difficulty, result.difficulty):
                    difficulty_boost = 1.1
            
            # Chef preference boost
            if preferred_chef and result.chef:
                if preferred_chef.lower() in result.chef.lower():
                    chef_boost = 1.5
            
            # Preparation time boost
            if max_prep_time and result.prep_time:
                prep_minutes = self._extract_minutes(result.prep_time)
                if prep_minutes and prep_minutes <= max_prep_time:
                    time_boost = 1.2
                elif prep_minutes and prep_minutes > max_prep_time * 1.5:
                    time_boost = 0.7
            
            # Length boost (prefer concise recipes)
            if result.word_count > 0:
                if result.word_count < 200:
                    length_boost = 1.1
                elif result.word_count > 500:
                    length_boost = 0.9
            
            # Calculate final personalized score
            personalized_score = normalized_score * difficulty_boost * chef_boost * time_boost * length_boost
            
            reranked_results.append(SearchResult(
                doc_id=result.doc_id,
                url=result.url,
                title=result.title,
                score=personalized_score,
                bm25_score=result.bm25_score,
                tfidf_score=result.tfidf_score,
                metadata_score=personalized_score,
                chef=result.chef,
                difficulty=result.difficulty,
                prep_time=result.prep_time,
                servings=result.servings,
                word_count=result.word_count
            ))
        
        # Sort by personalized scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Personalized reranking complete, returning top {top_k} results")
        return reranked_results[:top_k]
    
    def _is_similar_difficulty(self, pref: str, actual: str) -> bool:
        """Check if difficulty levels are similar."""
        difficulty_mapping = {
            "easy": ["very easy", "simple"],
            "medium": ["moderate", "intermediate"],
            "hard": ["difficult", "challenging", "advanced"]
        }
        
        pref_lower = pref.lower()
        actual_lower = actual.lower()
        
        for key, synonyms in difficulty_mapping.items():
            if key in pref_lower or any(s in pref_lower for s in synonyms):
                return key in actual_lower or any(s in actual_lower for s in synonyms)
        
        return False
    
    def _extract_minutes(self, time_str: str) -> Optional[int]:
        """Extract minutes from time string (e.g., '30 min', '1 hr 20 min')."""
        if not time_str:
            return None
        
        total_minutes = 0
        
        # Extract hours
        hr_match = re.search(r'(\d+)\s*h(?:r|our)?', time_str, re.IGNORECASE)
        if hr_match:
            total_minutes += int(hr_match.group(1)) * 60
        
        # Extract minutes
        min_match = re.search(r'(\d+)\s*min', time_str, re.IGNORECASE)
        if min_match:
            total_minutes += int(min_match.group(1))
        
        return total_minutes if total_minutes > 0 else None
    
    def search(self, query: str, 
               rerank_method: Optional[str] = None,
               top_k: int = None,
               **rerank_params) -> List[SearchResult]:
        """
        Perform search with optional reranking.
        
        Args:
            query: Search query string
            rerank_method: Reranking method ('cosine', 'personalized', or None)
            top_k: Number of results to return
            **rerank_params: Additional parameters for reranking (difficulty, chef, max_prep_time, etc.)
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
        # Initial BM25 search (get more results for reranking)
        initial_top_k = top_k * 3 if rerank_method else top_k
        results = self.search_bm25(query, top_k=initial_top_k)
        
        # Apply reranking if requested
        if rerank_method == 'cosine':
            results = self.rerank_cosine_similarity(query, results, top_k=top_k)
        elif rerank_method == 'personalized':
            results = self.rerank_personalized(query, results, top_k=top_k, **rerank_params)
        
        return results
    
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
        
        print(f"\n{'='*80}")
        print(f"Found {len(results)} results")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            
            # Display all available scores
            score_parts = []
            if result.bm25_score > 0:
                score_parts.append(f"BM25: {result.bm25_score:.4f}")
            if result.tfidf_score > 0:
                score_parts.append(f"TF-IDF: {result.tfidf_score:.4f}")
            if result.metadata_score > 0:
                score_parts.append(f"Metadata: {result.metadata_score:.4f}")
            if score_parts:
                print(f"   Scores: {' | '.join(score_parts)}")
            
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
    
    # Example 1: Basic BM25 search
    print("\n" + "="*80)
    print("Example 1: Basic BM25 Fulltext Search")
    print("="*80)
    query = "chicken pasta garlic"
    results = search_engine.search_bm25(query, top_k=5)
    search_engine.display_results(results)
    
    # Example 2: Cosine similarity reranking
    print("\n" + "="*80)
    print("Example 2: Search with TF-IDF Cosine Similarity Reranking")
    print("="*80)
    query = "chocolate cake dessert"
    results = search_engine.search(query, rerank_method='cosine', top_k=5)
    search_engine.display_results(results)
    
    # Example 3: Personalized reranking
    print("\n" + "="*80)
    print("Example 3: Search with Personalized Metadata Reranking")
    print("="*80)
    query = "quick easy salad"
    results = search_engine.search(
        query, 
        rerank_method='personalized',
        top_k=5,
        preferred_difficulty="Easy",
        max_prep_time=30
    )
    search_engine.display_results(results)
    
    # Interactive search
    print("\n" + "="*80)
    print("Interactive Search Mode")
    print("="*80)
    print("\nSelect reranking method:")
    print("  1. None (BM25 only)")
    print("  2. TF-IDF Cosine Similarity")
    print("  3. Metadata-based Personalization")
    print("  4. Both (TF-IDF + Metadata)")
    print()
    
    # Get reranking method selection
    rerank_choice = input("Choose reranking method (1-4): ").strip()
    
    rerank_method = None
    if rerank_choice == '1':
        rerank_method = None
        print("\nUsing: BM25 only")
    elif rerank_choice == '2':
        rerank_method = 'tfidf'
        print("\nUsing: BM25 + TF-IDF Cosine Similarity Reranking")
    elif rerank_choice == '3':
        rerank_method = 'metadata'
        print("\nUsing: BM25 + Metadata-based Personalization")
    elif rerank_choice == '4':
        rerank_method = 'both'
        print("\nUsing: BM25 + TF-IDF + Metadata Reranking")
    else:
        rerank_method = None
        print("\nInvalid choice. Using: BM25 only")
    
    print("\nEnter search queries (or 'quit' to exit, 'change' to change reranking method)")
    print()
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if not query or query.lower() == 'quit':
                break
            
            if query.lower() == 'change':
                print("\nSelect reranking method:")
                print("  1. None (BM25 only)")
                print("  2. TF-IDF Cosine Similarity")
                print("  3. Metadata-based Personalization")
                print("  4. Both (TF-IDF + Metadata)")
                print()
                
                rerank_choice = input("Choose reranking method (1-4): ").strip()
                
                if rerank_choice == '1':
                    rerank_method = None
                    print("\nUsing: BM25 only")
                elif rerank_choice == '2':
                    rerank_method = 'tfidf'
                    print("\nUsing: BM25 + TF-IDF Cosine Similarity Reranking")
                elif rerank_choice == '3':
                    rerank_method = 'metadata'
                    print("\nUsing: BM25 + Metadata-based Personalization")
                elif rerank_choice == '4':
                    rerank_method = 'both'
                    print("\nUsing: BM25 + TF-IDF + Metadata Reranking")
                else:
                    rerank_method = None
                    print("\nInvalid choice. Using: BM25 only")
                print()
                continue
            
            # Always start with BM25 search
            if rerank_method is None:
                # BM25 only
                results = search_engine.search_bm25(query, top_k=10)
            elif rerank_method == 'tfidf':
                # BM25 + TF-IDF reranking
                initial_results = search_engine.search_bm25(query, top_k=30)
                results = search_engine.rerank_cosine_similarity(query, initial_results, top_k=10)
            elif rerank_method == 'metadata':
                # BM25 + Metadata reranking
                initial_results = search_engine.search_bm25(query, top_k=30)
                results = search_engine.rerank_personalized(
                    query,
                    initial_results,
                    top_k=10,
                    preferred_difficulty="Easy",
                    max_prep_time=45
                )
            elif rerank_method == 'both':
                # BM25 + TF-IDF + Metadata reranking
                initial_results = search_engine.search_bm25(query, top_k=30)
                tfidf_results = search_engine.rerank_cosine_similarity(query, initial_results, top_k=30)
                results = search_engine.rerank_personalized(
                    query,
                    tfidf_results,
                    top_k=10,
                    preferred_difficulty="Easy",
                    max_prep_time=45
                )
            
            search_engine.display_results(results)
            
        except KeyboardInterrupt:
            print("\n\nSearch interrupted.")
            break
        except Exception as e:
            print(f"\nError during search: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nThank you for using Recipe Search Engine!")


if __name__ == "__main__":
    main()
