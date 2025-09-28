#!/usr/bin/env python3
"""
Recipe Search Engine - Fulltext Search Only

This module provides a simple fulltext search interface for recipe documents.
It loads recipes from JSON and performs basic text matching.
"""

import json
import sys
from typing import Dict, List
import logging
from pathlib import Path

class RecipeSearchEngine:
    """Search engine for recipe documents using fulltext search with inverted index"""
    
    def __init__(self, index_file: str = 'data/index/fulltext_index.json', 
                 documents_file: str = 'data/index/recipe_documents.json'):
        self.documents = {}
        self.inverted_index = {}
        self.stats = {}
        self.logger = self._setup_logging()
        self.load_fulltext_index(index_file)
        self.load_documents(documents_file)
    
    def _setup_logging(self):
        """Setup logging configuration"""
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
        """Load fulltext index from JSON file"""
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
        """Load recipe documents from JSON file"""
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
        """Tokenize and normalize query terms (simple version)"""
        import re
        
        # Convert to lowercase and extract words
        query = query.lower()
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-\']*\b', query)
        
        # Simple stemming and filtering (basic version)
        processed_tokens = []
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'will', 'with', 'you', 'your', 'this', 'but', 'they',
                     'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how'}
        
        for token in tokens:
            if len(token) > 2 and token not in stop_words:
                # Simple stemming
                if token.endswith('ing') and len(token) > 5:
                    token = token[:-3]
                elif token.endswith('ed') and len(token) > 4:
                    token = token[:-2]
                elif token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
                    token = token[:-1]
                
                if len(token) > 2:
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def search(self, query: str, top_k: int = 10) -> Dict:
        """
        Search using the inverted index for efficient fulltext search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        if not self.documents or not self.inverted_index:
            self.logger.error("No index loaded. Cannot perform search.")
            return {}
        
        # Tokenize query
        query_terms = self.tokenize_query(query)
        
        if not query_terms:
            return {'fulltext': []}
        
        # Get candidate documents using inverted index
        candidate_docs = set()
        term_doc_counts = {}
        
        for term in query_terms:
            if term in self.inverted_index:
                doc_list = self.inverted_index[term]
                candidate_docs.update(doc_list)
                term_doc_counts[term] = len(doc_list)
        
        if not candidate_docs:
            return {'fulltext': []}
        
        # Score candidate documents
        results = []
        query_lower = query.lower()
        
        for doc_id in candidate_docs:
            doc = self.documents[doc_id]
            
            # Count matching terms
            match_count = 0
            for term in query_terms:
                if term in self.inverted_index and doc_id in self.inverted_index[term]:
                    match_count += 1
            
            # Calculate relevance score
            relevance_score = match_count / len(query_terms)
            
            # Combine all searchable text for additional scoring
            searchable_text = (
                doc['title'] + " " +
                " ".join(doc['ingredients']) + " " +
                doc['method'] + " " +
                doc['chef_name']
            ).lower()
            
            # Bonus for exact phrase match
            if query_lower in searchable_text:
                relevance_score += 0.5
            
            # Bonus for title matches
            if query_lower in doc['title'].lower():
                relevance_score += 0.3
            
            # Boost score for rare terms (inverse document frequency concept)
            rarity_boost = 0
            for term in query_terms:
                if term in term_doc_counts:
                    # Give higher score to documents containing rarer terms
                    rarity_boost += 1.0 / (1 + term_doc_counts[term] / len(self.documents))
            
            relevance_score += rarity_boost * 0.1
            
            results.append({
                'doc_id': doc_id,
                'score': relevance_score,
                'match_count': match_count,
                'title': doc['title'],
                'chef_name': doc['chef_name'],
                'prep_time': doc['prep_time'],
                'url': doc['url'],
                'ingredients': doc['ingredients'][:3],  # Show first 3 ingredients
                'method_preview': doc['method'][:200] + "..." if len(doc['method']) > 200 else doc['method']
            })
        
        # Sort by relevance score (descending) and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {'fulltext': results[:top_k]}
    
    def get_document_details(self, doc_id: str) -> Dict:
        """Get full details of a specific document"""
        if not self.documents:
            return {}
        
        return self.documents.get(doc_id, {})
    
    def print_search_results(self, results: Dict, query: str):
        """Pretty print search results"""
        print(f"\nSearch Results for: '{query}'")
        print("=" * 60)
        
        if 'fulltext' in results and results['fulltext']:
            print(f"\n� Full-Text Search Results ({len(results['fulltext'])} found):")
            print("-" * 50)
            for i, result in enumerate(results['fulltext'], 1):
                print(f"{i}. {result['title']}")
                print(f"   👨‍🍳 Chef: {result['chef_name']}")
                print(f"   ⏱️  Time: {result['prep_time']}")
                print(f"   📊 Relevance: {result['score']:.2f} ({result['match_count']}/{len(query.split())} terms matched)")
                if result['ingredients']:
                    ingredients_str = ", ".join(result['ingredients'])
                    print(f"   🥘 Key ingredients: {ingredients_str}")
                if result['method_preview']:
                    print(f"   � Method preview: {result['method_preview']}")
                if result['url']:
                    print(f"   🔗 URL: {result['url']}")
                print()
        
        if not results or not results.get('fulltext'):
            print("No results found. Try different search terms.")
            print("\n💡 Search Tips:")
            print("• Try ingredient names: 'chicken', 'chocolate', 'pasta'")
            print("• Try cooking methods: 'grilled', 'baked', 'fried'")
            print("• Try partial phrases: 'chicken curry', 'chocolate cake'")
    
    def run_test_mode(self):
        """Run search engine in test mode with predefined queries"""
        print("🧪 Running Recipe Search Engine in TEST MODE")
        print("=" * 60)
        
        test_queries = [
            "chicken curry",
            "chocolate cake", 
            "pasta sauce",
            "vegetarian recipes",
            "air fryer",
            "steak",
            "soup",
            "salad",
            "bread",
            "dessert"
        ]
        
        for query in test_queries:
            results = self.search(query, top_k=5)
            self.print_search_results(results, query)
            
            # Add separator between queries
            print("\n" + "="*60 + "\n")
    
    def run_interactive_mode(self):
        """Run search engine in interactive console mode"""
        print("🔍 Recipe Search Engine - Interactive Mode")
        print("=" * 60)
        print("Enter your search queries. Type 'help' for commands or 'quit' to exit.")
        
        while True:
            try:
                query = input("\n🔍 Search> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! Happy cooking! 👋")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if query.lower() == 'stats':
                    self.show_stats()
                    continue
                
                # Parse query options
                top_k = 10
                
                # Handle top:N:query format
                if ':' in query and query.startswith('top:'):
                    parts = query.split(':', 2)
                    if len(parts) == 3:
                        try:
                            top_k = int(parts[1].strip())
                            query = parts[2].strip()
                        except ValueError:
                            print("Invalid number format. Using default top 10.")
                
                # Perform search
                results = self.search(query, top_k=top_k)
                self.print_search_results(results, query)
                
            except KeyboardInterrupt:
                print("\nGoodbye! Happy cooking! 👋")
                break
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                print(f"An error occurred: {e}")
    
    def show_help(self):
        """Display help information"""
        print("\n📖 Help - Available Commands:")
        print("-" * 40)
        print("• Simply type your search query (e.g., 'chicken curry')")
        print("• 'top:N:your query' - Return top N results (e.g., 'top:3:pasta')")
        print("• 'stats' - Show index statistics")
        print("• 'help' - Show this help")
        print("• 'quit' or 'exit' - Exit the search engine")
        print("\n💡 Full-Text Search Tips:")
        print("• Search matches words in title, ingredients, method, and chef name")
        print("• Use multiple words to find recipes containing all terms")
        print("• Try ingredient names: 'chicken', 'chocolate', 'pasta'")
        print("• Try cooking methods: 'grilled', 'baked', 'fried'")
        print("• Try cuisine types: 'italian', 'indian', 'mexican'")
        print("• Try dietary preferences: 'vegetarian', 'vegan', 'gluten-free'")
        print("• Exact phrase matches get higher relevance scores")
    
    def show_stats(self):
        """Display fulltext index statistics"""
        if not self.documents:
            print("No index loaded.")
            return
        
        print("\n📈 Fulltext Index Statistics:")
        print("-" * 30)
        print(f"• Total Documents: {len(self.documents):,}")
        print(f"• Vocabulary Size: {len(self.inverted_index):,}")
        
        if self.stats:
            for key, value in self.stats.items():
                formatted_key = key.replace('_', ' ').title()
                print(f"• {formatted_key}: {value:,}")
        
        # Calculate some basic statistics
        if self.documents:
            total_ingredients = sum(len(doc['ingredients']) for doc in self.documents.values())
            avg_ingredients = total_ingredients / len(self.documents)
            print(f"• Average Ingredients Per Recipe: {avg_ingredients:.1f}")
            
            # Count unique chefs
            unique_chefs = len(set(doc['chef_name'] for doc in self.documents.values()))
            print(f"• Unique Chefs: {unique_chefs:,}")
            
            # Show some example terms from the index
            if self.inverted_index:
                print(f"• Sample Terms: {', '.join(list(self.inverted_index.keys())[:10])}")

def main():
    """Main function - entry point for the search engine"""
    # Initialize search engine
    search_engine = RecipeSearchEngine()
    
    if not search_engine.documents:
        print("❌ Failed to load documents. Please run 'python indexer.py' first to build the index.")
        sys.exit(1)
    
    # Check command line arguments for mode selection
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        search_engine.run_test_mode()
    else:
        # Interactive mode (default)
        search_engine.run_interactive_mode()

if __name__ == "__main__":
    main()
