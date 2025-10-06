import os
import json
import re
import sys
from functools import partial
from typing import Dict, List
from dataclasses import dataclass, field
from uuid import uuid4

import config
import tiktoken
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def normalize_text(text: str) -> str:
    """
    Normalize text for tokenization by converting to lowercase and removing special characters.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized and trimmed text string
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(text: str, min_word_length: int, max_word_length: int, stop_words_broadcast) -> List[str]:
    """
    Tokenize and filter text based on word length, stop words, and numeric values.
    
    Args:
        text: Input text to tokenize
        min_word_length: Minimum allowed word length
        max_word_length: Maximum allowed word length
        stop_words_broadcast: Spark broadcast variable containing stop words set
        
    Returns:
        List of filtered tokens
    """
    if not text:
        return []
    normalized = normalize_text(text)
    tokens = normalized.split()
    stop_words = stop_words_broadcast.value
    return [token for token in tokens if min_word_length <= len(token) <= max_word_length and token not in stop_words and not token.isdigit()]


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken encoding.
    
    Args:
        text: Input text to count tokens
        
    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0
    return len(TOKENIZER.encode(text))

@dataclass
class DocumentStats:
    """
    Statistics for a document in the index.
    
    Attributes:
        doc_id: Unique document identifier
        url: Document URL
        title: Document title
        word_count: Total number of words
        unique_words: Number of unique words
        token_count: Total token count
        chef: Recipe chef name
        difficulty: Recipe difficulty level
        prep_time: Recipe preparation time
        servings: Number of servings
    """
    doc_id: str
    url: str
    title: str
    word_count: int
    unique_words: int
    token_count: int = 0
    chef: str = ""
    difficulty: str = ""
    prep_time: str = ""
    servings: str = ""


@dataclass
class IndexEntry:
    """
    Single entry in the inverted index.
    
    Attributes:
        term: The indexed term
        document_frequency: Number of documents containing the term
        postings: Dictionary mapping document IDs to term frequencies
    """
    term: str
    document_frequency: int
    postings: Dict[str, int] = field(default_factory=dict)


@dataclass
class SearchIndex:
    """
    Complete search index structure.
    
    Attributes:
        inverted_index: Dictionary of terms to IndexEntry objects
        document_stats: Dictionary of document IDs to DocumentStats objects
        total_documents: Total number of indexed documents
        vocabulary_size: Total number of unique terms
    """
    inverted_index: Dict[str, IndexEntry] = field(default_factory=dict)
    document_stats: Dict[str, DocumentStats] = field(default_factory=dict)
    total_documents: int = 0
    vocabulary_size: int = 0

class RecipeIndexer:
    """
    Inverted indexer for recipe data with Spark processing.
    
    Creates an inverted index from recipe documents. Uses Apache Spark for
    efficient distributed data processing, tokenization, and aggregation.
    Generates document statistics and vocabulary metrics.
    """
    
    def __init__(self, 
                 recipes_file: str = None,
                 index_dir: str = None,
                 log_level: int = None):
        """
        Initialize the RecipeIndexer.
        
        Args:
            recipes_file: Path to input recipes JSONL file
            index_dir: Directory to save index files
            log_level: Logging level
        """
        self.recipes_file = recipes_file if recipes_file is not None else config.RECIPES_FILE
        self.index_dir = index_dir if index_dir is not None else config.INDEX_DIR
        log_level = log_level if log_level is not None else config.LOG_LEVEL
        
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.logger = config.setup_logging(config.INDEXER_LOG, log_level)
        
        self.min_word_length = config.MIN_WORD_LENGTH
        self.max_word_length = config.MAX_WORD_LENGTH
        self.stop_words = config.STOP_WORDS
        
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        
        self.spark = SparkSession.builder \
            .appName("RecipeIndexer") \
            .master("local[*]") \
            .config("spark.python.worker.faulthandler.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.ui.port", "4050") \
            .getOrCreate()
        
        self.stop_words_broadcast = self.spark.sparkContext.broadcast(self.stop_words)

    def calculate_html_size(self, df) -> int:
        """
        Calculate total size of HTML files referenced in the dataframe.
        
        Args:
            df: Spark DataFrame containing html_file column
            
        Returns:
            Total size in bytes
        """
        html_files_df = df.select("html_file").distinct()
        html_file_paths = [row.html_file for row in html_files_df.collect() if row.html_file]
        
        total_size = sum(
            os.path.getsize(path) 
            for path in html_file_paths 
            if path and os.path.exists(path)
        )
        
        return total_size

    def save_metadata(self, total_documents: int, vocabulary_size: int, total_html_size: int, total_tokens: int):
        """
        Save index metadata to JSONL file.
        
        Args:
            total_documents: Total number of indexed documents
            vocabulary_size: Total number of unique terms
            total_html_size: Total size of HTML files in bytes
            total_tokens: Total token count across all documents
        """
        metadata_path = os.path.join(self.index_dir, "metadata.jsonl")
        metadata = {
            "total_documents": total_documents,
            "vocabulary_size": vocabulary_size,
            "total_tokens": total_tokens,
            "size_bytes": total_html_size,
            "size_mb": round(total_html_size / (1024 * 1024), 2)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
            f.write('\n')

    def save_document_mapping(self, doc_stats_list):
        """
        Save document statistics to JSONL file.
        
        Args:
            doc_stats_list: List of document statistics rows from Spark
        """
        mapping_path = os.path.join(self.index_dir, "mapping.jsonl")
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for row in doc_stats_list:
                doc_stat = {
                    "doc_id": row.doc_id,
                    "url": row.url,
                    "title": row.title,
                    "word_count": row.word_count,
                    "unique_words": row.unique_words,
                    "token_count": row.token_count,
                    "chef": row.chef or "",
                    "difficulty": row.difficulty or "",
                    "prep_time": row.prep_time or "",
                    "servings": row.servings or ""
                }
                json.dump(doc_stat, f)
                f.write('\n')

    def save_inverted_index(self, inverted_index_list):
        """
        Save inverted index to JSONL file.
        
        Args:
            inverted_index_list: List of inverted index entries from Spark
        """
        index_path = os.path.join(self.index_dir, "index.jsonl")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            for row in inverted_index_list:
                term_entry = {
                    "term": row.term,
                    "document_frequency": row.document_frequency,
                    "postings": row.postings
                }
                json.dump(term_entry, f)
                f.write('\n')

    def build_index(self):
        """
        Build the complete inverted index from recipe data.
        
        Processes recipe JSONL file with Spark to create:
        - Inverted index with term frequencies
        - Document statistics and metadata
        - Vocabulary and token counts
        """
        if not os.path.exists(self.recipes_file):
            self.logger.error(f"Recipes file not found: {self.recipes_file}")
            return
        
        self.logger.info(f"Building index from {self.recipes_file}")

        uuid_udf = F.udf(lambda: str(uuid4()), T.StringType())
        df = self.spark.read.json(self.recipes_file).withColumn("doc_id", uuid_udf())
        df.cache()

        total_documents = df.count()
        total_html_size = self.calculate_html_size(df)
        
        self.logger.info(f"Indexing {total_documents} documents ({total_html_size / (1024*1024):.2f} MB)")

        df = df.withColumn("full_text", F.concat_ws(" ",
            F.col("title"),
            F.col("description"),
            F.concat_ws(" ", F.col("ingredients")),
            F.col("method")
        ))

        tokenize_udf = F.udf(
            partial(tokenize_text, 
                    min_word_length=self.min_word_length, 
                    max_word_length=self.max_word_length, 
                    stop_words_broadcast=self.stop_words_broadcast),
            T.ArrayType(T.StringType())
        )
        df = df.withColumn("tokens", tokenize_udf(F.col("full_text")))

        term_doc_df = df.select("doc_id", F.explode("tokens").alias("term"))
        tf_df = term_doc_df.groupBy("term", "doc_id").count().withColumnRenamed("count", "tf")
        
        postings_df = tf_df.groupBy("term").agg(
            F.collect_list(F.struct(F.col("doc_id"), F.col("tf"))).alias("postings_struct")
        )
        
        inverted_index_df = postings_df.withColumn(
            "postings", F.map_from_entries(F.col("postings_struct"))
        ).withColumn(
            "document_frequency", F.size(F.col("postings_struct"))
        ).select("term", "document_frequency", "postings")

        inverted_index_df.cache()
        vocabulary_size = inverted_index_df.count()

        tiktoken_len_udf = F.udf(count_tokens, T.IntegerType())

        doc_stats_df = df.withColumn("word_count", F.size(F.col("tokens"))) \
                         .withColumn("unique_words", F.size(F.array_distinct(F.col("tokens")))) \
                         .withColumn("token_count", tiktoken_len_udf(F.col("full_text"))) \
                         .select(
                             "doc_id", "url", "title", "word_count", "unique_words", "token_count",
                               "chef", "difficulty", "prep_time", "servings"
                         )
        
        total_tokens = doc_stats_df.agg(F.sum("token_count")).collect()[0][0] or 0
        
        doc_stats_list = doc_stats_df.collect()
        inverted_index_list = inverted_index_df.collect()

        self.save_metadata(total_documents, vocabulary_size, total_html_size, total_tokens)
        self.save_document_mapping(doc_stats_list)
        self.save_inverted_index(inverted_index_list)

        df.unpersist()
        inverted_index_df.unpersist()
        self.stop_words_broadcast.unpersist()

        self.logger.info("Index building completed successfully!")
        self.spark.stop()


def main():
    """Main entry point for building the recipe index."""
    indexer = RecipeIndexer()
    indexer.build_index()


if __name__ == "__main__":
    main()
