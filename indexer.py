#!/usr/bin/env python3
"""
Recipe Indexer with TF-IDF and BM25 retrieval metrics

This module builds a primitive inverted index over recipe data from raw HTML files,
extracting recipe title, ingredients, method, and chef name. It implements two
retrieval metrics: TF-IDF with cosine similarity and BM25 (Okapi).
"""

import json
import re
import math
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import logging
from pathlib import Path

from bs4 import BeautifulSoup

class RecipeDocument:
    """Represents a recipe document with its metadata and content"""
    
    def __init__(self, doc_id: str, title: str, ingredients: List[str], 
                 method: str, chef_name: str = None, url: str = None,
                 prep_time: str = None):
        self.doc_id = doc_id
        self.title = title
        self.ingredients = ingredients
        self.method = method
        self.chef_name = chef_name or "Unknown"
        self.url = url
        self.prep_time = prep_time or "Not specified"
        
        # Combine all text for indexing
        self.description = self._combine_text()
        
    def _combine_text(self) -> str:
        """Combine all recipe components into a single text for indexing"""
        text_parts = [self.title]
        text_parts.extend(self.ingredients)
        text_parts.append(self.method)
        text_parts.append(self.chef_name)
        text_parts.append(self.prep_time)
        
        return " ".join(filter(None, text_parts))
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary for serialization"""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'ingredients': self.ingredients,
            'method': self.method,
            'chef_name': self.chef_name,
            'url': self.url,
            'prep_time': self.prep_time,
            'description': self.description
        }

class TextProcessor:
    """Handles text preprocessing for indexing using regex only"""
    
    def __init__(self):
        # Define stop words manually
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so',
            'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
            'more', 'very', 'when', 'come', 'may', 'use', 'than', 'first',
            'been', 'one', 'now', 'find', 'way', 'who', 'oil', 'water', 'can',
            'could', 'should', 'other', 'after', 'first', 'well', 'get', 'here',
            'all', 'new', 'just', 'see', 'only', 'know', 'take', 'also', 'back',
            'good', 'give', 'most', 'over', 'think', 'where', 'much', 'go', 'work',
            # Cooking-specific stop words
            'recipe', 'ingredients', 'method', 'instructions', 'cooking',
            'preparation', 'serves', 'minutes', 'hours', 'cup', 'cups',
            'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'ounce',
            'ounces', 'pound', 'pounds', 'gram', 'grams', 'ml', 'litre',
            'salt', 'pepper', 'add', 'mix', 'stir', 'heat', 'cook', 'serve',
            'large', 'small', 'medium', 'fresh', 'chopped', 'diced', 'sliced'
        }
    
    def simple_stem(self, word: str) -> str:
        """Simple stemming using regex patterns"""
        word = word.lower()
        
        # Remove common suffixes
        if word.endswith('ing') and len(word) > 5:
            word = word[:-3]
        elif word.endswith('ed') and len(word) > 4:
            word = word[:-2]
        elif word.endswith('er') and len(word) > 4:
            word = word[:-2]
        elif word.endswith('est') and len(word) > 5:
            word = word[:-3]
        elif word.endswith('ly') and len(word) > 4:
            word = word[:-2]
        elif word.endswith('ies') and len(word) > 5:
            word = word[:-3] + 'y'
        elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            word = word[:-1]
        
        return word
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenize, lowercase, remove stopwords, simple stemming
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of processed tokens
        """
        # Clean HTML tags if any
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Extract words using regex (alphanumeric + common punctuation in food)
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-\']*\b', text)
        
        # Filter and stem tokens
        processed_tokens = []
        for token in tokens:
            # Remove tokens that are too short or are stop words
            if len(token) > 2 and token not in self.stop_words:
                # Simple stemming
                stemmed = self.simple_stem(token)
                if len(stemmed) > 2:  # Check length after stemming
                    processed_tokens.append(stemmed)
        
        return processed_tokens

class InvertedIndex:
    """Inverted index implementation with TF-IDF and BM25 scoring"""
    
    def __init__(self):
        self.documents: Dict[str, RecipeDocument] = {}
        self.term_doc_freq: Dict[str, Set[str]] = defaultdict(set)  # term -> set of doc_ids
        self.doc_term_freq: Dict[str, Dict[str, int]] = defaultdict(dict)  # doc_id -> term -> count
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> document length
        self.vocab_size: int = 0
        self.total_docs: int = 0
        self.avg_doc_length: float = 0.0
        
        self.processor = TextProcessor()
        
    def add_document(self, document: RecipeDocument):
        """Add a document to the index"""
        doc_id = document.doc_id
        self.documents[doc_id] = document
        
        # Preprocess document text
        tokens = self.processor.preprocess_text(document.description)
        
        # Update document length
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update term frequencies
        term_counts = Counter(tokens)
        
        for term, count in term_counts.items():
            self.term_doc_freq[term].add(doc_id)
            if doc_id not in self.doc_term_freq:
                self.doc_term_freq[doc_id] = {}
            self.doc_term_freq[doc_id][term] = count
            
        self._update_stats()
    
    def _update_stats(self):
        """Update collection statistics"""
        self.total_docs = len(self.documents)
        self.vocab_size = len(self.term_doc_freq)
        
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
    
    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term"""
        return len(self.term_doc_freq.get(term, set()))
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Get term frequency in a specific document"""
        return self.doc_term_freq.get(doc_id, {}).get(term, 0)
        
    def get_document(self, doc_id: str) -> Optional[RecipeDocument]:
        """Retrieve a document by ID"""
        return self.documents.get(doc_id)
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_documents': self.total_docs,
            'vocabulary_size': self.vocab_size,
            'average_document_length': self.avg_doc_length
        }

class RecipeIndexer:
    """Main class for building recipe index from HTML files"""
    
    def __init__(self, html_dir: str):
        self.html_dir = Path(html_dir)
        self.index = InvertedIndex()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/indexer.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def extract_recipe_data(self, html_content: str, filename: str) -> Optional[RecipeDocument]:
        """
        Extract recipe data from HTML content
        
        Args:
            html_content: Raw HTML content
            filename: Name of the HTML file (used as doc_id)
            
        Returns:
            RecipeDocument object or None if extraction fails
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find JSON-LD structured data for recipe
            recipe_script = soup.find('script', type='application/ld+json', string=re.compile(r'"@type":\s*"Recipe"'))
            
            if recipe_script:
                recipe_data = json.loads(recipe_script.string)
                
                # Extract recipe components
                title = recipe_data.get('name', '').strip()
                ingredients = recipe_data.get('recipeIngredient', [])
                
                # Clean ingredients list
                if isinstance(ingredients, list):
                    ingredients = [ing.strip() for ing in ingredients if ing.strip()]
                else:
                    ingredients = []
                
                # Extract method/instructions - try JSON-LD first (most reliable), then HTML
                method = ""
                
                # Method 1: Use JSON-LD structured data (preferred)
                instructions = recipe_data.get('recipeInstructions', '')
                if isinstance(instructions, list):
                    # Handle array of instruction objects
                    method_steps = []
                    for instruction in instructions:
                        if isinstance(instruction, dict):
                            # Get step name and text
                            step_name = instruction.get('name', '')
                            step_text = instruction.get('text', '')
                            if step_name and step_text:
                                method_steps.append(f"{step_name}: {step_text}")
                            elif step_text:
                                method_steps.append(step_text)
                        else:
                            text = str(instruction)
                            if text:
                                # Remove HTML tags
                                text = re.sub(r'<[^>]+>', ' ', text)
                                text = re.sub(r'\s+', ' ', text).strip()
                                method_steps.append(text)
                    method = " ".join(method_steps)
                elif isinstance(instructions, str):
                    # Remove HTML tags from instructions
                    method = re.sub(r'<[^>]+>', ' ', instructions)
                    method = re.sub(r'\s+', ' ', method).strip()
                
                # Method 2: Fallback to HTML structure if JSON-LD doesn't provide good method
                if not method or len(method) < 50:  # If method is too short, try HTML
                    instructions_divs = soup.find_all('div', class_='mb-6 e-instructions')
                    if instructions_divs:
                        method_steps = []
                        for step_div in instructions_divs:
                            step_title_elem = step_div.find('h3')
                            step_content_elem = step_div.find('p')
                            
                            if step_title_elem and step_content_elem:
                                step_title = step_title_elem.get_text().strip()
                                step_content = step_content_elem.get_text().strip()
                                
                                # Format as "Step X: content" ensuring proper step formatting
                                if step_title.lower().startswith('step '):
                                    method_steps.append(f"{step_title}: {step_content}")
                                elif step_title:
                                    # For things like "Cook's Note:" etc.
                                    method_steps.append(f"{step_title}: {step_content}")
                                else:
                                    method_steps.append(step_content)
                        
                        if method_steps:
                            method = " ".join(method_steps)
                
                # Extract chef/author name
                author = recipe_data.get('author', {})
                if isinstance(author, dict):
                    chef_name = author.get('name', 'Unknown')
                elif isinstance(author, str):
                    chef_name = author
                else:
                    chef_name = 'Unknown'
                
                # Get URL if available
                url = recipe_data.get('url', '')
                
                # Extract preparation time
                prep_time = recipe_data.get('prepTime', '')
                if prep_time:
                    # PT150M -> 150 minutes, PT2H30M -> 2 hours 30 minutes
                    time_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?', prep_time)
                    if time_match:
                        hours = int(time_match.group(1)) if time_match.group(1) else 0
                        minutes = int(time_match.group(2)) if time_match.group(2) else 0
                        if hours and minutes:
                            prep_time = f"{hours} hours {minutes} minutes"
                        elif hours:
                            prep_time = f"{hours} hours"
                        elif minutes:
                            prep_time = f"{minutes} minutes"
                        else:
                            prep_time = "Not specified"
                    else:
                        prep_time = prep_time if prep_time else "Not specified"
                else:
                    prep_time = "Not specified"
                
                # Use filename (without extension) as document ID
                doc_id = Path(filename).stem
                
                if title:  # Only create document if we have a title
                    return RecipeDocument(
                        doc_id=doc_id,
                        title=title,
                        ingredients=ingredients,
                        method=method,
                        chef_name=chef_name,
                        url=url,
                        prep_time=prep_time
                    )
            
            # Fallback: try to extract from HTML structure
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
                # Remove common suffixes
                title = re.sub(r'\s*\|\s*Food Network.*$', '', title, flags=re.IGNORECASE)
                title = re.sub(r'\s*Recipe\s*$', '', title, flags=re.IGNORECASE)
                
                doc_id = Path(filename).stem
                return RecipeDocument(
                    doc_id=doc_id,
                    title=title,
                    ingredients=[],
                    method='',
                    chef_name='Unknown',
                    prep_time='Not specified'
                )
            
        except Exception as e:
            self.logger.error(f"Error extracting recipe data from {filename}: {e}")
        
        return None
    
    def build_index(self) -> bool:
        """
        Build the inverted index from all HTML files in the directory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            html_files = list(self.html_dir.glob('*.html'))
            self.logger.info(f"Found {len(html_files)} HTML files to process")
            
            processed = 0
            skipped = 0
            
            for html_file in html_files:
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    recipe_doc = self.extract_recipe_data(html_content, html_file.name)
                    
                    if recipe_doc:
                        self.index.add_document(recipe_doc)
                        processed += 1
                        
                        if processed % 100 == 0:
                            self.logger.info(f"Processed {processed} documents...")
                    else:
                        skipped += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing {html_file}: {e}")
                    skipped += 1
            
            self.logger.info("Index building completed:")
            self.logger.info(f"  Processed: {processed} documents")
            self.logger.info(f"  Skipped: {skipped} documents")
            self.logger.info(f"  Index stats: {self.index.get_stats()}")
            
            return processed > 0
            
        except Exception as e:
            self.logger.error(f"Error building index: {e}")
            return False
    
    def save_index(self, filename: str = 'data/index/recipe_index.pkl'):
        """Save the index to a pickle file"""
        try:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.index, f)
            self.logger.info(f"Index saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def load_index(self, filename: str = 'data/index/recipe_index.pkl') -> bool:
        """Load an index from a pickle file"""
        try:
            with open(filename, 'rb') as f:
                self.index = pickle.load(f)
            self.logger.info(f"Index loaded from {filename}")
            self.logger.info(f"Index stats: {self.index.get_stats()}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return False
    

    
    def export_documents(self, filename: str = 'data/index/recipe_documents.json'):
        """Export all documents to JSON for inspection"""
        try:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            docs_dict = {}
            for doc_id, doc in self.index.documents.items():
                docs_dict[doc_id] = doc.to_dict()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(docs_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Documents exported to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting documents: {e}")
    
    def export_fulltext_index(self, filename: str = 'data/index/fulltext_index.json'):
        """Export a simple fulltext inverted index to JSON (without documents)"""
        try:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Create simple inverted index: term -> list of doc_ids
            fulltext_index = {}
            for term, doc_ids_set in self.index.term_doc_freq.items():
                fulltext_index[term] = list(doc_ids_set)
            
            # Export index data (without documents - they're in recipe_documents.json)
            index_data = {
                'inverted_index': fulltext_index,
                'stats': {
                    'total_documents': len(self.index.documents),
                    'vocabulary_size': len(fulltext_index),
                    'total_terms': sum(len(doc_ids) for doc_ids in fulltext_index.values())
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Fulltext index exported to {filename}")
            self.logger.info(f"Index contains {len(fulltext_index)} unique terms")
            
        except Exception as e:
            self.logger.error(f"Error exporting fulltext index: {e}")

def main():
    """Main function for building the index"""
    # Initialize indexer
    html_dir = "data/raw_html"
    indexer = RecipeIndexer(html_dir)
    
    # Build index
    print("Building index from HTML files...")
    if indexer.build_index():
        # Save index
        indexer.save_index()
        
        # Export documents for inspection
        indexer.export_documents()
        
        # Export fulltext index for search
        indexer.export_fulltext_index()
        
        print("\n" + "="*60)
        print("Index Statistics:")
        stats = indexer.index.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nIndex building completed successfully!")
        print("Use search.py to search the indexed recipes.")
        
    else:
        print("Failed to build index!")

if __name__ == "__main__":
    main()
