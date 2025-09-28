#!/usr/bin/env python3
"""
Recipe Search Engine

This module provides a fulltext search interface for recipe documents
using TF-based scoring with multi-term query optimization.
"""

import json
import re
import sys
from typing import Dict, List
import logging
from pathlib import Path

class RecipeSearchEngine:
    """Search engine for recipe documents using fulltext search with TF scoring."""
    
    def __init__(self, index_file: str = 'data/index/fulltext_index.json', 
                 documents_file: str = 'data/index/recipe_documents.json'):
        """
        Initialize the search engine.
        
        Args:
            index_file: Path to the fulltext index JSON file
            documents_file: Path to the recipe documents JSON file
        """
        self.documents = {}
        self.inverted_index = {}
        self.stats = {}
        self.logger = self._setup_logging()
        self.load_fulltext_index(index_file)
        self.load_documents(documents_file)
    
    def _setup_logging(self):
        """
        Setup logging configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/search.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_fulltext_index(self, filename: str) -> bool:
        """
        Load fulltext index from JSON file.
        
        Args:
            filename: Path to the fulltext index JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(filename).exists():
                self.logger.error(f"Fulltext index file {filename} not found. Please run indexer.py first.")
                return False
                
            with open(filename, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.inverted_index = index_data.get('inverted_index', {})
            self.stats = index_data.get('stats', {})
            
            self.logger.info(f"Fulltext index loaded from {filename}")
            self.logger.info(f"Vocabulary size: {len(self.inverted_index)}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading fulltext index: {e}")
            return False
    
    def load_documents(self, filename: str) -> bool:
        """
        Load recipe documents from JSON file.
        
        Args:
            filename: Path to the recipe documents JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(filename).exists():
                self.logger.error(f"Documents file {filename} not found. Please run indexer.py first.")
                return False
                
            with open(filename, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            self.logger.info(f"Documents loaded from {filename}")
            self.logger.info(f"Total documents: {len(self.documents)}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            return False
    
    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize and normalize query terms with simple stemming.
        
        Args:
            query: Search query string
            
        Returns:
            list: List of processed query terms
        """
        query = query.lower()
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-\']*\b', query)
        
        # Basic stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how'
        }
        
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in stop_words:
                # Apply simple stemming
                stemmed = self._simple_stem(token)
                if len(stemmed) > 2:
                    processed_tokens.append(stemmed)
        
        return processed_tokens
    
    def _simple_stem(self, word: str) -> str:
        """
        Apply simple stemming rules to a word.
        
        Args:
            word: Word to stem
            
        Returns:
            str: Stemmed word
        """
        if word.endswith('ing') and len(word) > 5:
            return word[:-3]
        elif word.endswith('ed') and len(word) > 4:
            return word[:-2]
        elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            return word[:-1]
        return word
    
    def calculate_tf_score(self, term: str, doc_text: str) -> float:
        """
        Calculate Term Frequency score for a term in document text.
        
        Args:
            term: Term to calculate TF for
            doc_text: Document text
            
        Returns:
            float: TF score (term count / total words)
        """
        words = doc_text.lower().split()
        if not words:
            return 0.0
        
        term_count = words.count(term.lower())
        return term_count / len(words)
    
    def search(self, query: str, top_k: int = 1) -> Dict:
        """
        Search using TF scoring with multi-term query optimization.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            dict: Search results dictionary
        """
        if not self.documents or not self.inverted_index:
            self.logger.error("No index loaded. Cannot perform search.")
            return {}
        
        query_terms = self.tokenize_query(query)
        if not query_terms:
            return {'fulltext': []}
        
        # Find candidate documents containing any query term
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        if not candidate_docs:
            return {'fulltext': []}
        
        # Score each candidate document
        results = []
        for recipe_title in candidate_docs:
            doc = self.documents[recipe_title]
            
            # Combine searchable text fields
            searchable_text = " ".join([
                recipe_title,
                " ".join(doc['ingredients']),
                doc['method'],
                doc['chef_name']
            ])
            
            # Calculate scores for each query term
            individual_tf_scores = {}
            matched_terms = []
            total_tf_score = 0.0
            
            for term in query_terms:
                if term in self.inverted_index and recipe_title in self.inverted_index[term]:
                    # Base TF score
                    tf_score = self.calculate_tf_score(term, searchable_text)
                    
                    # Apply field-specific bonuses
                    title_bonus = 0.5 if term.lower() in recipe_title.lower() else 0
                    ingredient_bonus = 0.3 if any(term.lower() in ing.lower() for ing in doc['ingredients']) else 0
                    
                    adjusted_tf_score = tf_score + title_bonus + ingredient_bonus
                    individual_tf_scores[term] = adjusted_tf_score
                    matched_terms.append(term)
                    total_tf_score += adjusted_tf_score
                else:
                    individual_tf_scores[term] = 0.0
            
            if not matched_terms:
                continue
            
            # Calculate final score with multi-term bonus
            avg_tf_score = total_tf_score / len(query_terms)
            term_match_ratio = len(matched_terms) / len(query_terms)
            multi_term_bonus = term_match_ratio * 0.5  # Bonus for matching multiple terms
            final_score = avg_tf_score + multi_term_bonus
            
            results.append({
                'id': recipe_title,
                'score': final_score,
                'tf_score': avg_tf_score,
                'multi_term_bonus': multi_term_bonus,
                'term_match_ratio': term_match_ratio,
                'matched_terms': matched_terms,
                'individual_tf_scores': individual_tf_scores,
                'title': recipe_title,
                'chef_name': doc['chef_name'],
                'prep_time': doc['prep_time'],
                'url': doc['url'],
                'ingredients': doc['ingredients'],
                'method_preview': searchable_text
            })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        final_results = results[:top_k]
        
        # Log best match details
        if final_results:
            best_result = final_results[0]
            self.logger.info(f"BEST MATCH for '{query}': {best_result['title']}")
            self.logger.info(f"  Final Score: {best_result['score']:.6f}")
            self.logger.info(f"  TF Score: {best_result['tf_score']:.6f}")
            self.logger.info(f"  Multi-term Bonus: {best_result['multi_term_bonus']:.6f}")
            self.logger.info(f"  Term Match Ratio: {best_result['term_match_ratio']:.2f} ({len(best_result['matched_terms'])}/{len(query_terms)})")
            self.logger.info(f"  Matched Terms: {best_result['matched_terms']}")
        
        return {'fulltext': final_results}
    
    def print_search_results(self, results: Dict, query: str):
        """
        Pretty print the search results (designed for single best result).
        
        Args:
            results: Search results dictionary
            query: Original search query
        """
        print(f"\nBest Match for: '{query}'")
        print("=" * 60)
        
        if 'fulltext' in results and results['fulltext']:
            result = results['fulltext'][0]
            
            print(f"{result['title']}")
            print(f"   Chef: {result['chef_name']}")
            print(f"   Time: {result['prep_time']}")
            print(f"   Final Score: {result.get('score', 0):.6f}")
            
            # Show individual term scores
            if 'individual_tf_scores' in result:
                print("   Individual Term Scores:")
                for term, score in result['individual_tf_scores'].items():
                    status = "+" if score > 0 else "-"
                    print(f"      {status} '{term}': {score:.6f}")
                            
            if result['ingredients']:
                ingredients_str = ", ".join(result['ingredients'])
                print(f"   Key ingredients: {ingredients_str}")
            
            if result['method_preview']:
                print(f"   Method preview: {result['method_preview']}")

            if result['url']:
                print(f"   URL: {result['url']}")
            print()
        else:
            self._print_no_results_help()
    
    def _print_no_results_help(self):
        """Print help message when no results are found."""
        print("No results found. Try different search terms.")
        print("\nSearch Tips:")
        print("• Single terms: 'chicken', 'chocolate', 'pasta'")
        print("• Multiple terms: 'pickled onion' - prioritizes documents matching all terms")
        print("• Returns only the single best matching document")
        print("• Multi-term queries get bonus scoring for matching more terms")
        print("• Title and ingredient matches receive additional scoring bonuses")
    
    def run_interactive_mode(self):
        """Run search engine in interactive console mode."""
        print("Recipe Search Engine - Interactive Mode")
        print("=" * 60)
        print("Enter your search queries. Type 'help' for commands or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nSearch> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! Happy cooking!")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if query.lower() == 'stats':
                    self.show_stats()
                    continue
                
                # Perform search and display results
                results = self.search(query, top_k=1)
                self.print_search_results(results, query)
                
            except KeyboardInterrupt:
                print("\nGoodbye! Happy cooking!")
                break
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                print(f"An error occurred: {e}")
    
    def show_help(self):
        """Display help information."""
        print("\nHelp - Available Commands:")
        print("-" * 40)
        print("• Simply type your search query (e.g., 'pickled onion')")
        print("• 'stats' - Show index statistics")
        print("• 'help' - Show this help")
        print("• 'quit' or 'exit' - Exit the search engine")
        print("\nSearch Features:")
        print("• TF scoring with multi-term optimization")
        print("• Returns only the single best matching document")
        print("• Multi-term queries prioritize documents matching all terms")
        print("• Title and ingredient matches receive bonus scoring")
        print("• Search matches words in title, ingredients, method, and chef name")
        print("• Try multi-term queries: 'pickled onion', 'blue cheese', 'chocolate cake'")
    
    def show_stats(self):
        """Display index statistics."""
        if not self.documents:
            print("No index loaded.")
            return
        
        print("\nIndex Statistics:")
        print("-" * 30)
        print(f"• Total Documents: {len(self.documents):,}")
        print(f"• Vocabulary Size: {len(self.inverted_index):,}")
        
        if self.documents:
            total_ingredients = sum(len(doc['ingredients']) for doc in self.documents.values())
            avg_ingredients = total_ingredients / len(self.documents)
            print(f"• Average Ingredients Per Recipe: {avg_ingredients:.1f}")
            
            unique_chefs = len(set(doc['chef_name'] for doc in self.documents.values()))
            print(f"• Unique Chefs: {unique_chefs:,}")

def main():
    """Main function - entry point for the search engine."""
    search_engine = RecipeSearchEngine()
    
    if not search_engine.documents:
        print("Failed to load documents. Please run 'python indexer.py' first to build the index.")
        sys.exit(1)
    
    search_engine.run_interactive_mode()

if __name__ == "__main__":
    main()
