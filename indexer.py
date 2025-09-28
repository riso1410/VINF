#!/usr/bin/env python3
"""
Recipe Indexer with TF-IDF Retrieval

This module builds an inverted index over recipe data from raw HTML files,
extracting recipe title, ingredients, method, and chef name. It implements
TF-IDF scoring for recipe search functionality.
"""

import json
import re
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional
import logging
from pathlib import Path

from bs4 import BeautifulSoup

class RecipeDocument:
    """
    Represents a recipe document with its metadata and content.
    
    Attributes:
        title (str): Recipe title
        ingredients (List[str]): List of ingredients
        method (str): Cooking method/instructions
        chef_name (str): Chef or author name
        url (str): Recipe URL
        prep_time (str): Preparation time
        description (str): Combined text for backward compatibility
    """
    
    def __init__(self, title: str, ingredients: List[str], 
                 method: str, chef_name: str = None, url: str = None,
                 prep_time: str = None):
        """
        Initialize a recipe document.
        
        Args:
            title: Recipe title
            ingredients: List of ingredients
            method: Cooking method/instructions
            chef_name: Chef or author name (optional)
            url: Recipe URL (optional)
            prep_time: Preparation time (optional)
        """
        self.title = title
        self.ingredients = ingredients
        self.method = method
        self.chef_name = chef_name or "Unknown"
        self.url = url
        self.prep_time = prep_time or "Not specified"
        self.description = self._combine_text()
        
    def _combine_text(self) -> str:
        """
        Combine all recipe components into a single text for indexing.
        
        Returns:
            str: Combined text from all recipe fields
        """
        text_parts = [self.title]
        text_parts.extend(self.ingredients)
        text_parts.append(self.method)
        text_parts.append(self.chef_name)
        text_parts.append(self.prep_time)
        
        return " ".join(filter(None, text_parts))
    
    def get_tf_text(self) -> str:
        """
        Combine title, ingredients, and method for TF calculation.
        
        Returns:
            str: Combined text for term frequency calculation
        """
        text_parts = [self.title]
        text_parts.extend(self.ingredients)
        text_parts.append(self.method)
        
        return " ".join(filter(None, text_parts))
    
    def to_dict(self) -> Dict:
        """
        Convert document to dictionary for serialization.
        
        Returns:
            dict: Document data as dictionary
        """
        return {
            'ingredients': self.ingredients,
            'method': self.method,
            'chef_name': self.chef_name,
            'url': self.url,
            'prep_time': self.prep_time,
            'description': self.description
        }

class TextProcessor:
    """Handles text preprocessing for indexing using regex-based operations."""
    
    def __init__(self):
        """Initialize text processor with stop words."""
        # Common English stop words plus cooking-specific terms
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so',
            'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
            'more', 'very', 'when', 'come', 'may', 'use', 'than', 'first',
            'been', 'one', 'now', 'find', 'way', 'who', 'oil', 'water', 'can',
            'could', 'should', 'other', 'after', 'well', 'get', 'here',
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
        """
        Apply simple stemming rules to a word.
        
        Args:
            word: Word to stem
            
        Returns:
            str: Stemmed word
        """
        word = word.lower()
        
        # Apply common suffix removal rules
        if word.endswith('ing') and len(word) > 5:
            word = word[:-3]
        elif word.endswith('ed') and len(word) > 4:
            word = word[:-2]
        elif word.endswith('est') and len(word) > 5:
            word = word[:-3]
        elif word.endswith('ly') and len(word) > 4:
            word = word[:-2] + 'e'
        elif word.endswith('ies') and len(word) > 5:
            word = word[:-3] + 'y'
        elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            word = word[:-1]
        
        return word
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenize, lowercase, remove stopwords, apply stemming.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            list: List of processed tokens
        """
        # Clean HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.lower()
        
        # Extract alphanumeric words with hyphens and apostrophes
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-\']*\b', text)
        
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                stemmed = self.simple_stem(token)
                if len(stemmed) > 2:
                    processed_tokens.append(stemmed)
        
        return processed_tokens

class InvertedIndex:
    """Inverted index implementation with TF scoring capabilities."""
    
    def __init__(self):
        """Initialize the inverted index."""
        self.documents: Dict[str, RecipeDocument] = {}
        self.term_doc_freq: Dict[str, Set[str]] = defaultdict(set)  # term -> set of doc_ids
        self.doc_term_freq: Dict[str, Dict[str, int]] = defaultdict(dict)  # doc_id -> term -> count
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> document length
        self.vocab_size: int = 0
        self.total_docs: int = 0
        self.avg_doc_length: float = 0.0
        
        self.processor = TextProcessor()
        
    def add_document(self, document: RecipeDocument):
        """
        Add a document to the index.
        
        Args:
            document: RecipeDocument to add to the index
        """
        doc_id = document.title
        self.documents[doc_id] = document
        
        # Process text for indexing
        tf_text = document.get_tf_text()
        tokens = self.processor.preprocess_text(tf_text)
        
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update term frequency mappings
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            self.term_doc_freq[term].add(doc_id)
            if doc_id not in self.doc_term_freq:
                self.doc_term_freq[doc_id] = {}
            self.doc_term_freq[doc_id][term] = count
            
        self._update_stats()
    
    def _update_stats(self):
        """Update collection statistics."""
        self.total_docs = len(self.documents)
        self.vocab_size = len(self.term_doc_freq)
        
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency for a term.
        
        Args:
            term: Term to get document frequency for
            
        Returns:
            int: Number of documents containing the term
        """
        return len(self.term_doc_freq.get(term, set()))
    
    def get_term_frequency(self, term: str, title: str) -> int:
        """
        Get term frequency in a specific document.
        
        Args:
            term: Term to get frequency for
            title: Document title (used as document ID)
            
        Returns:
            int: Frequency of term in the document
        """
        return self.doc_term_freq.get(title, {}).get(term, 0)
        
    def get_document(self, title: str) -> Optional[RecipeDocument]:
        """
        Retrieve a document by title.
        
        Args:
            title: Document title to retrieve
            
        Returns:
            RecipeDocument or None if not found
        """
        return self.documents.get(title)
    
    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            dict: Dictionary containing index statistics
        """
        return {
            'total_documents': self.total_docs,
            'vocabulary_size': self.vocab_size,
            'average_document_length': self.avg_doc_length
        }
    
    def calculate_tf(self, term: str, title: str) -> float:
        """
        Calculate normalized term frequency (TF) for a term in a document.
        TF = (frequency of term in document) / (total number of terms in document)
        
        Args:
            term: The term to calculate TF for
            title: Recipe title (used as document key)
            
        Returns:
            float: Normalized term frequency
        """
        if title not in self.doc_term_freq or title not in self.doc_lengths:
            return 0.0
            
        term_freq = self.get_term_frequency(term, title)
        doc_length = self.doc_lengths[title]
        
        return term_freq / doc_length if doc_length > 0 else 0.0
    
    def get_document_tf_vector(self, title: str) -> Dict[str, float]:
        """
        Get TF vector for a document (all terms and their TF values).
        
        Args:
            title: Recipe title (used as document key)
            
        Returns:
            dict: Dictionary mapping terms to their TF values
        """
        if title not in self.doc_term_freq:
            return {}
            
        doc_terms = self.doc_term_freq[title]
        doc_length = self.doc_lengths[title]
        
        if doc_length == 0:
            return {}
            
        return {term: count / doc_length for term, count in doc_terms.items()}
    
    def get_tf_calculation_details(self, title: str) -> Dict:
        """
        Get detailed information about TF calculation for a document.
        
        Args:
            title: Recipe title (used as document key)
            
        Returns:
            dict: Dictionary with TF calculation details
        """
        if title not in self.documents:
            return {}
            
        document = self.documents[title]
        tf_text = document.get_tf_text()
        processed_tokens = self.processor.preprocess_text(tf_text)
        
        return {
            'title': document.title,
            'tf_text': tf_text,
            'processed_tokens': processed_tokens,
            'total_tokens': len(processed_tokens),
            'unique_terms': len(set(processed_tokens)),
            'term_frequencies': dict(Counter(processed_tokens)),
            'tf_vector': self.get_document_tf_vector(title)
        }

class RecipeIndexer:
    """Main class for building recipe index from HTML files."""
    
    def __init__(self, html_dir: str):
        """
        Initialize the recipe indexer.
        
        Args:
            html_dir: Directory containing HTML files to index
        """
        self.html_dir = Path(html_dir)
        self.index = InvertedIndex()
        self.logger = self._setup_logging()
        
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
                logging.FileHandler('logs/indexer.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def extract_recipe_data(self, html_content: str, filename: str) -> Optional[RecipeDocument]:
        """
        Extract recipe data from HTML content using JSON-LD structured data.
        
        Args:
            html_content: Raw HTML content of the recipe page
            filename: Name of the HTML file for error logging
            
        Returns:
            RecipeDocument: Extracted recipe data or None if extraction fails
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for JSON-LD structured recipe data
            recipe_script = soup.find('script', type='application/ld+json', string=re.compile(r'"@type":\s*"Recipe"'))
            
            if recipe_script:
                recipe_data = json.loads(recipe_script.string)
                return self._extract_from_json_ld(recipe_data, soup)
            
            # Fallback: extract basic info from HTML title
            return self._extract_fallback(soup)
            
        except Exception as e:
            self.logger.error(f"Error extracting recipe data from {filename}: {e}")
            return None
    
    def _extract_from_json_ld(self, recipe_data: dict, soup) -> Optional[RecipeDocument]:
        """
        Extract recipe data from JSON-LD structured data.
        
        Args:
            recipe_data: Parsed JSON-LD recipe data
            soup: BeautifulSoup object for fallback HTML parsing
            
        Returns:
            RecipeDocument: Extracted recipe or None
        """
        title = recipe_data.get('name', '').strip()
        if not title:
            return None
            
        # Extract and clean ingredients
        ingredients = recipe_data.get('recipeIngredient', [])
        if isinstance(ingredients, list):
            ingredients = [ing.strip() for ing in ingredients if ing.strip()]
        else:
            ingredients = []
        
        # Extract cooking method/instructions
        method = self._extract_method(recipe_data, soup)
        
        # Extract author information
        author = recipe_data.get('author', {})
        if isinstance(author, dict):
            chef_name = author.get('name', 'Unknown')
        elif isinstance(author, str):
            chef_name = author
        else:
            chef_name = 'Unknown'
        
        # Extract other metadata
        url = recipe_data.get('url', '')
        prep_time = self._parse_prep_time(recipe_data.get('prepTime', ''))
        
        return RecipeDocument(
            title=title,
            ingredients=ingredients,
            method=method,
            chef_name=chef_name,
            url=url,
            prep_time=prep_time
        )
    
    def _extract_method(self, recipe_data: dict, soup) -> str:
        """
        Extract cooking method from JSON-LD or HTML fallback.
        
        Args:
            recipe_data: JSON-LD recipe data
            soup: BeautifulSoup object for HTML parsing
            
        Returns:
            str: Cooking method text
        """
        instructions = recipe_data.get('recipeInstructions', '')
        method_steps = []
        
        if isinstance(instructions, list):
            # Process instruction objects/strings
            for instruction in instructions:
                if isinstance(instruction, dict):
                    step_name = instruction.get('name', '')
                    step_text = instruction.get('text', '')
                    if step_name and step_text:
                        method_steps.append(f"{step_name}: {step_text}")
                    elif step_text:
                        method_steps.append(step_text)
                else:
                    # Clean HTML from text instructions
                    text = re.sub(r'<[^>]+>', ' ', str(instruction))
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text:
                        method_steps.append(text)
        elif isinstance(instructions, str):
            # Clean HTML tags from instruction text
            method = re.sub(r'<[^>]+>', ' ', instructions)
            method = re.sub(r'\s+', ' ', method).strip()
            return method
        
        method = " ".join(method_steps)
        
        # Fallback to HTML parsing if method is too short
        if not method or len(method) < 50:
            method = self._extract_method_from_html(soup)
        
        return method
    
    def _extract_method_from_html(self, soup) -> str:
        """
        Extract cooking method from HTML structure as fallback.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            str: Extracted method text
        """
        instructions_divs = soup.find_all('div', class_='mb-6 e-instructions')
        method_steps = []
        
        for step_div in instructions_divs:
            step_title_elem = step_div.find('h3')
            step_content_elem = step_div.find('p')
            
            if step_title_elem and step_content_elem:
                step_title = step_title_elem.get_text().strip()
                step_content = step_content_elem.get_text().strip()
                
                if step_title.lower().startswith('step '):
                    method_steps.append(f"{step_title}: {step_content}")
                elif step_title:
                    method_steps.append(f"{step_title}: {step_content}")
                else:
                    method_steps.append(step_content)
        
        return " ".join(method_steps)
    
    def _parse_prep_time(self, prep_time_str: str) -> str:
        """
        Parse ISO 8601 duration format to human-readable time.
        
        Args:
            prep_time_str: Duration string (e.g., "PT150M", "PT2H30M")
            
        Returns:
            str: Human-readable preparation time
        """
        if not prep_time_str:
            return "Not specified"
            
        # Parse ISO 8601 duration: PT150M -> 150 minutes, PT2H30M -> 2 hours 30 minutes
        time_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?', prep_time_str)
        if time_match:
            hours = int(time_match.group(1)) if time_match.group(1) else 0
            minutes = int(time_match.group(2)) if time_match.group(2) else 0
            
            if hours and minutes:
                return f"{hours} hours {minutes} minutes"
            elif hours:
                return f"{hours} hours"
            elif minutes:
                return f"{minutes} minutes"
            else:
                return "Not specified"
        
        return prep_time_str if prep_time_str else "Not specified"
    
    def _extract_fallback(self, soup) -> Optional[RecipeDocument]:
        """
        Extract basic recipe info from HTML title as fallback.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            RecipeDocument: Basic recipe document or None
        """
        title_elem = soup.find('title')
        if not title_elem:
            return None
            
        title = title_elem.get_text().strip()
        # Clean up title by removing common suffixes
        title = re.sub(r'\s*\|\s*Food Network.*$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*Recipe\s*$', '', title, flags=re.IGNORECASE)
        
        return RecipeDocument(
            title=title,
            ingredients=[],
            method='',
            chef_name='Unknown',
            prep_time='Not specified'
        )
    
    def build_index(self) -> bool:
        """
        Build the inverted index from all HTML files in the directory.
        
        Returns:
            bool: True if successful, False otherwise
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
        """
        Save the index to a pickle file.
        
        Args:
            filename: Path to save the index file
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.index, f)
            self.logger.info(f"Index saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def load_index(self, filename: str = 'data/index/recipe_index.pkl') -> bool:
        """
        Load an index from a pickle file.
        
        Args:
            filename: Path to the index file to load
            
        Returns:
            bool: True if successful, False otherwise
        """
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
        """
        Export all documents to JSON for inspection.
        
        Args:
            filename: Path to save the documents JSON file
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            docs_dict = {title: doc.to_dict() for title, doc in self.index.documents.items()}
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(docs_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Documents exported to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting documents: {e}")
    
    def export_fulltext_index(self, filename: str = 'data/index/fulltext_index.json'):
        """
        Export a simple fulltext inverted index to JSON.
        
        Args:
            filename: Path to save the fulltext index JSON file
        """
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Create inverted index mapping: term -> list of recipe titles
            fulltext_index = {term: list(titles_set) for term, titles_set in self.index.term_doc_freq.items()}
            
            index_data = {'inverted_index': fulltext_index}
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Fulltext index exported to {filename}")
            self.logger.info(f"Index contains {len(fulltext_index)} unique terms")
            
        except Exception as e:
            self.logger.error(f"Error exporting fulltext index: {e}")

def main():
    """Main function for building the recipe index."""
    html_dir = "data/raw_html"
    indexer = RecipeIndexer(html_dir)
    
    print("Building index from HTML files...")
    
    if indexer.build_index():
        # Save and export index files
        indexer.save_index()
        indexer.export_documents()
        indexer.export_fulltext_index()
        
        # Display statistics
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
