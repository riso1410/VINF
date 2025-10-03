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

def normalize_text(text: str) -> str:
    """Normalize text for tokenization."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_text(text: str, min_word_length: int, max_word_length: int, stop_words: set) -> List[str]:
    """Tokenize text with filtering."""
    if not text: 
        return []
    normalized = normalize_text(text)
    tokens = normalized.split()
    return [token for token in tokens if min_word_length <= len(token) <= max_word_length and token not in stop_words and not token.isdigit()]

def count_tiktoken_tokens(text: str) -> int:
    """Count tokens using tiktoken encoding."""
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

@dataclass
class DocumentStats:
    """Statistics for a document in the index"""
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
    """Single entry in the inverted index"""
    term: str
    document_frequency: int
    postings: Dict[str, int] = field(default_factory=dict)  # doc_id -> tf

@dataclass
class SearchIndex:
    """Complete search index structure"""
    inverted_index: Dict[str, IndexEntry] = field(default_factory=dict)
    document_stats: Dict[str, DocumentStats] = field(default_factory=dict)
    total_documents: int = 0
    vocabulary_size: int = 0

class RecipeIndexer:
    """Spark-based inverted indexer for recipe data."""
    
    def __init__(self, 
                 recipes_file: str = None,
                 index_dir: str = None,
                 log_level: int = None):
        
        self.recipes_file = recipes_file if recipes_file is not None else config.RECIPES_FILE
        self.index_dir = index_dir if index_dir is not None else config.INDEX_DIR
        log_level = log_level if log_level is not None else config.LOG_LEVEL
        
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.logger = config.setup_logging(config.INDEXER_LOG, log_level)
        
        self.min_word_length = config.MIN_WORD_LENGTH
        self.max_word_length = config.MAX_WORD_LENGTH
        self.stop_words = config.STOP_WORDS
        self.tokenizer_encoding = tiktoken.get_encoding("cl100k_base")
        
        # Set Python executable for PySpark workers
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        
        self.spark = SparkSession.builder \
            .appName("RecipeIndexer") \
            .master("local[*]") \
            .config("spark.python.worker.faulthandler.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .getOrCreate()

    def normalize_text(self, text: str) -> str:
        """Instance method for backward compatibility."""
        return normalize_text(text)

    def tokenize(self, text: str) -> List[str]:
        """Instance method for backward compatibility."""
        return tokenize_text(text, self.min_word_length, self.max_word_length, self.stop_words)

    def _calculate_html_size(self, df) -> tuple:
        """Calculate total size of HTML files."""
        html_files_df = df.select("html_file").distinct()
        html_file_paths = [row.html_file for row in html_files_df.collect() if row.html_file]
        
        total_size = sum(
            os.path.getsize(path) 
            for path in html_file_paths 
            if path and os.path.exists(path)
        )
        
        return len(html_file_paths), total_size

    def save_metadata(self, total_documents: int, vocabulary_size: int, total_html_size: int):
        """Save index metadata to file."""
        metadata_path = os.path.join(self.index_dir, "metadata.jsonl")
        metadata = {
            "total_documents": total_documents,
            "vocabulary_size": vocabulary_size,
            "size_bytes": total_html_size,
            "size_mb": round(total_html_size / (1024 * 1024), 2)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
            f.write('\n')
        
        self.logger.info(f"Metadata saved to {metadata_path}")

    def save_document_mapping(self, doc_stats_list):
        """Save document statistics to file."""
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
        
        self.logger.info(f"Document mapping saved to {mapping_path}")

    def save_inverted_index(self, inverted_index_list):
        """Save inverted index to file."""
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
        
        self.logger.info(f"Inverted index saved to {index_path}")

    def build_index(self):
        if not os.path.exists(self.recipes_file):
            self.logger.error(f"Recipes file not found: {self.recipes_file}")
            return
        
        self.logger.info(f"Building Spark-based inverted index from {self.recipes_file} using DataFrames")

        # Read data and add doc_id
        uuid_udf = F.udf(lambda: str(uuid4()), T.StringType())
        df = self.spark.read.json(self.recipes_file).withColumn("doc_id", uuid_udf())
        df.cache()

        total_documents = df.count()
        html_files_count, total_html_size = self._calculate_html_size(df)
        
        self.logger.info(f"Total HTML files: {html_files_count}, Total size: {total_html_size:,} bytes ({total_html_size / (1024*1024):.2f} MB)")

        # Create full text and tokenize
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
                    stop_words=self.stop_words),
            T.ArrayType(T.StringType())
        )
        df = df.withColumn("tokens", tokenize_udf(F.col("full_text")))

        # Build inverted index
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

        tiktoken_len_udf = F.udf(count_tiktoken_tokens, T.IntegerType())

        doc_stats_df = df.withColumn("word_count", F.size(F.col("tokens"))) \
                         .withColumn("unique_words", F.size(F.array_distinct(F.col("tokens")))) \
                         .withColumn("token_count", tiktoken_len_udf(F.col("full_text"))) \
                         .select(
                             "doc_id", "url", "title", "word_count", "unique_words", "token_count",
                             "chef", "difficulty", "prep_time", "servings"
                         )
        
        # Collect results
        doc_stats_list = doc_stats_df.collect()
        inverted_index_list = inverted_index_df.collect()

        # Save all outputs
        self.save_metadata(total_documents, vocabulary_size, total_html_size)
        self.save_document_mapping(doc_stats_list)
        self.save_inverted_index(inverted_index_list)

        # Cleanup
        df.unpersist()
        inverted_index_df.unpersist()

        self.logger.info("Index building completed successfully!")
        self.spark.stop()

def main():
    indexer = RecipeIndexer()
    indexer.build_index()

if __name__ == "__main__":
    main()
