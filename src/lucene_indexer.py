"""
PyLucene indexer for wiki_matches.jsonl

Creates a searchable index with weighted fields:
- recipe_title (weight: 3.0)
- wiki_title (weight: 2.0)
- ingredients (weight: 2.5)
- description (weight: 1.5)
- wiki_description (weight: 1.0)
- method (weight: 1.0)
"""

import json
import sys
import os

# PyLucene imports
import lucene
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField, StoredField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from java.nio.file import Paths as JavaPaths


# Field weights for scoring
FIELD_WEIGHTS = {
    'recipe_title': 3.0,
    'wiki_title': 2.0,
    'ingredients': 2.5,
    'description': 1.5,
    'wiki_description': 1.0,
    'method': 1.0,
    'chef': 0.5,
    'difficulty': 0.3,
    'origin': 1.5,
}


def create_index(data_file, index_dir):
    """
    Create Lucene index from wiki_matches.jsonl
    
    Args:
        data_file: Path to wiki_matches.jsonl
        index_dir: Directory to store the index
    """
    # Initialize Lucene VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    
    print(f"Creating index from: {data_file}")
    print(f"Index location: {index_dir}")
    
    # Create index directory
    store = MMapDirectory(JavaPaths.get(index_dir))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    
    writer = IndexWriter(store, config)
    
    # Read and index documents
    indexed_count = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                recipe = json.loads(line)
                doc = Document()
                
                # Add recipe_url as ID (not analyzed, stored)
                recipe_url = recipe.get('recipe_url', '')
                if recipe_url:
                    doc.add(StringField('recipe_url', recipe_url, Field.Store.YES))
                
                # Add recipe_title (analyzed, stored)
                recipe_title = recipe.get('recipe_title', '')
                if recipe_title:
                    doc.add(TextField('recipe_title', recipe_title, Field.Store.YES))
                
                # Add wiki_title (analyzed, stored)
                wiki_title = recipe.get('wiki_title', '')
                if wiki_title:
                    doc.add(TextField('wiki_title', wiki_title, Field.Store.YES))
                
                # Add description (analyzed, stored)
                description = recipe.get('description', '')
                if description:
                    doc.add(TextField('description', description, Field.Store.YES))
                
                # Add wiki_description (analyzed, stored)
                wiki_description = recipe.get('wiki_description', '')
                if wiki_description:
                    doc.add(TextField('wiki_description', wiki_description, Field.Store.YES))
                
                # Add ingredients (analyzed, stored)
                ingredients = recipe.get('ingredients', [])
                if ingredients:
                    ingredients_text = ' '.join(ingredients) if isinstance(ingredients, list) else str(ingredients)
                    doc.add(TextField('ingredients', ingredients_text, Field.Store.YES))
                
                # Add method (analyzed, stored)
                method = recipe.get('method', '')
                if method:
                    doc.add(TextField('method', method, Field.Store.YES))
                
                # Add chef (analyzed, stored)
                chef = recipe.get('chef', '')
                if chef:
                    doc.add(TextField('chef', chef, Field.Store.YES))
                
                # Add difficulty (analyzed, stored)
                difficulty = recipe.get('difficulty', '')
                if difficulty:
                    doc.add(TextField('difficulty', difficulty, Field.Store.YES))
                
                # Add metadata fields (stored only)
                for field_name in ['prep_time', 'servings', 'wiki_url']:
                    value = recipe.get(field_name, '')
                    if value:
                        doc.add(StoredField(field_name, str(value)))
                
                # Add document to index
                writer.addDocument(doc)
                indexed_count += 1
                
                if indexed_count % 1000 == 0:
                    print(f"Indexed {indexed_count:,} recipes...")
            
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error indexing line {line_num}: {e}")
                continue
    
    # Commit and close
    writer.commit()
    writer.close()
    
    print("\n✓ Indexing complete!")
    print(f"Total recipes indexed: {indexed_count:,}")
    print("\nField weights:")
    for field, weight in sorted(FIELD_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: {weight}")


def main():
    """Main entry point"""
    data_file = os.getenv('DATA_FILE', '/app/data/wiki_matches.jsonl')
    index_dir = os.getenv('INDEX_DIR', '/app/index')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    # Create index directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    create_index(data_file, index_dir)
    print(f"\nIndex created successfully at: {index_dir}")


if __name__ == '__main__':
    main()
