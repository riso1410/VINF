"""
PyLucene interactive search CLI

Provides command-line interface for querying the recipe index.
"""
import os

# PyLucene imports
import lucene
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BoostQuery
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from java.nio.file import Paths as JavaPaths


# Field weights for multi-field search
SEARCH_FIELDS = [
    'recipe_title',
    'wiki_title', 
    'ingredients',
    'description',
    'wiki_description',
    'method',
    'chef',
]

FIELD_BOOSTS = {
    'recipe_title': 3.0,
    'wiki_title': 2.0,
    'ingredients': 2.5,
    'description': 1.5,
    'wiki_description': 1.0,
    'method': 1.0,
    'chef': 0.5,
}

# Global searcher
searcher = None
analyzer = None


def init_searcher(index_dir):
    """Initialize Lucene searcher"""
    global searcher, analyzer
    
    # Initialize Lucene VM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    
    print(f"Opening index from: {index_dir}")
    
    store = MMapDirectory(JavaPaths.get(index_dir))
    reader = DirectoryReader.open(store)
    searcher = IndexSearcher(reader)
    analyzer = StandardAnalyzer()
    
    print(f"Index loaded: {reader.numDocs():,} documents")
    return searcher


def search_recipes(query_text, max_results=10, fields=None):
    """
    Search recipes using multi-field query
    
    Args:
        query_text: User's search query
        max_results: Maximum number of results to return
        fields: List of fields to search (default: all searchable fields)
        
    Returns:
        List of search results with scores
    """
    if not searcher:
        return {'error': 'Index not initialized'}
    
    if fields is None:
        fields = SEARCH_FIELDS
    
    try:
        from org.apache.lucene.search import BooleanQuery, BooleanClause
        
        # Create a BooleanQuery with boosted field queries
        builder = BooleanQuery.Builder()
        
        # For each field, create individual queries with boosts
        for field in fields:
            field_parser = QueryParser(field, analyzer)
            try:
                field_query = field_parser.parse(query_text)
                boost = FIELD_BOOSTS.get(field, 1.0)
                
                # Wrap in BoostQuery if boost != 1.0
                if boost != 1.0:
                    boosted_query = BoostQuery(field_query, boost)
                    builder.add(boosted_query, BooleanClause.Occur.SHOULD)
                else:
                    builder.add(field_query, BooleanClause.Occur.SHOULD)
            except:
                # Skip fields that don't parse
                pass
        
        query = builder.build()
        
        # Search
        hits = searcher.search(query, max_results)
        
        # Process results
        results = []
        for i, hit in enumerate(hits.scoreDocs):
            doc = searcher.storedFields().document(hit.doc)
            
            result = {
                'rank': i + 1,
                'score': float(hit.score),
                'recipe_url': doc.get('recipe_url'),
                'recipe_title': doc.get('recipe_title'),
                'wiki_title': doc.get('wiki_title'),
                'description': doc.get('description'),
                'wiki_description': doc.get('wiki_description'),
                'ingredients': doc.get('ingredients'),
                'chef': doc.get('chef'),
                'difficulty': doc.get('difficulty'),
                'prep_time': doc.get('prep_time'),
                'servings': doc.get('servings'),
                'wiki_url': doc.get('wiki_url'),
            }
            
            results.append(result)
        
        return {
            'query': query_text,
            'total_hits': hits.totalHits.value,
            'results': results
        }
    
    except Exception as e:
        return {'error': str(e)}


def interactive_search():
    """Interactive search mode (console)"""
    print("\n" + "="*70)
    print("RECIPE SEARCH - Interactive Mode")
    print("="*70)
    print("\nEnter search queries (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input("Search> ").strip()
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            results = search_recipes(query, max_results=10)
            
            if 'error' in results:
                print(f"Error: {results['error']}\n")
                continue
            
            print(f"\nFound {results['total_hits']} results:\n")
            
            for result in results['results']:
                print(f"{result['rank']}. {result['recipe_title']} (score: {result['score']:.2f})")
                if result['wiki_title']:
                    print(f"   Wikipedia: {result['wiki_title']}")
                if result['chef']:
                    print(f"   Chef: {result['chef']}")
                if result['description']:
                    desc = result['description'][:150] + '...' if len(result['description']) > 150 else result['description']
                    print(f"   {desc}")
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point"""
    index_dir = os.getenv('INDEX_DIR', '/app/index')
    
    # Check if index exists, if not create it
    if not os.path.exists(os.path.join(index_dir, 'segments_1')):
        print("Index not found, creating it first...")
        from lucene_indexer import create_index
        data_file = os.getenv('DATA_FILE', '/app/data/wiki_matches.jsonl')
        create_index(data_file, index_dir)
    
    # Initialize searcher
    init_searcher(index_dir)
    
    # Run interactive search
    interactive_search()


if __name__ == '__main__':
    main()
