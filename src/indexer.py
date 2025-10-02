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

# Global tokenization functions that can be serialized
def normalize_text(text: str) -> str:
    """Normalize text for tokenization."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Keep only alphanumeric and space characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def tokenize_text(text: str, min_word_length: int, max_word_length: int, stop_words: set) -> List[str]:
    """Tokenize text with filtering."""
    if not text: 
        return []
    normalized = normalize_text(text)
    tokens = normalized.split()
    return [token for token in tokens if min_word_length <= len(token) <= max_word_length and token not in stop_words and not token.isdigit()]

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

    def build_index(self):
        if not os.path.exists(self.recipes_file):
            self.logger.error(f"Recipes file not found: {self.recipes_file}")
            return
        
        self.logger.info(f"Building Spark-based inverted index from {self.recipes_file} using DataFrames")

        # 1. Read data and add doc_id
        df = self.spark.read.json(self.recipes_file)
        
        # Create UUID UDF
        uuid_udf = F.udf(lambda: str(uuid4()), T.StringType())
        df = df.withColumn("doc_id", uuid_udf())
        df.cache()

        total_documents = df.count()

        # 2. Tokenization
        df = df.withColumn("full_text", F.concat_ws(" ",
            F.col("title"),
            F.col("description"),
            F.concat_ws(" ", F.col("ingredients")),
            F.col("method")
        ))

        tokenize_partial = partial(tokenize_text, 
                                   min_word_length=self.min_word_length, 
                                   max_word_length=self.max_word_length, 
                                   stop_words=self.stop_words)
        tokenize_udf = F.udf(tokenize_partial, T.ArrayType(T.StringType()))
        
        df = df.withColumn("tokens", tokenize_udf(F.col("full_text")))

        # 3. Build Inverted Index
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

        # 4. Build Document Stats
        def tiktoken_len(text):
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        tiktoken_len_udf = F.udf(tiktoken_len, T.IntegerType())

        doc_stats_df = df.withColumn("word_count", F.size(F.col("tokens"))) \
                         .withColumn("unique_words", F.size(F.array_distinct(F.col("tokens")))) \
                         .withColumn("token_count", tiktoken_len_udf(F.col("full_text"))) \
                         .select(
                             "doc_id", "url", "title", "word_count", "unique_words", "token_count",
                             "chef", "difficulty", "prep_time", "servings"
                         )

        # 5. Save results
        self.logger.info("Saving index to JSONL files...")

        # Collect results to driver
        doc_stats_list = doc_stats_df.collect()
        inverted_index_list = inverted_index_df.collect()

        # Save metadata.jsonl
        metadata_path = os.path.join(self.index_dir, "metadata.jsonl")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            metadata = {
                "total_documents": total_documents,
                "vocabulary_size": vocabulary_size
            }
            json.dump(metadata, f)
            f.write('\n')
        
        self.logger.info(f"Metadata saved to {metadata_path}")

        # Save mapping.jsonl (document stats only)
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
                    "chef": row.chef if row.chef else "",
                    "difficulty": row.difficulty if row.difficulty else "",
                    "prep_time": row.prep_time if row.prep_time else "",
                    "servings": row.servings if row.servings else ""
                }
                json.dump(doc_stat, f)
                f.write('\n')
        
        self.logger.info(f"Document mapping saved to {mapping_path}")

        # Save index.jsonl (term entries only)
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

        df.unpersist()
        inverted_index_df.unpersist()

        self.logger.info("Index building completed successfully!")
        self.spark.stop()

def main():
    indexer = RecipeIndexer()
    indexer.build_index()

if __name__ == "__main__":
    main()
