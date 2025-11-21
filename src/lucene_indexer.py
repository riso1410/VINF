import json
import os
import sys

import lucene
from java.nio.file import Paths as JavaPaths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import (
    Document,
    Field,
    StoredField,
    StringField,
    TextField,
)
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import MMapDirectory

FIELD_WEIGHTS = {
    "recipe_title": 3.0,
    "wiki_title": 2.0,
    "ingredients": 2.5,
    "description": 1.5,
    "wiki_description": 1.0,
    "method": 1.0,
    "chef": 0.5,
    "difficulty": 0.3,
    "origin": 1.5,
    "wiki_course": 1.0,
    "wiki_origin": 1.5,
    "wiki_serving_temp": 0.5,
    "wiki_categories": 1.0,
}


# Create Lucene index from JSONL data
def create_index(data_file, index_dir):
    lucene.initVM(vmargs=["-Djava.awt.headless=true"])

    print(f"Creating index from: {data_file}")
    print(f"Index location: {index_dir}")

    store = MMapDirectory(JavaPaths.get(index_dir))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(store, config)

    indexed_count = 0

    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                recipe = json.loads(line)
                doc = Document()

                recipe_url = recipe.get("recipe_url", "")
                if recipe_url:
                    doc.add(StringField("recipe_url", recipe_url, Field.Store.YES))

                recipe_title = recipe.get("recipe_title", "")
                if recipe_title:
                    doc.add(TextField("recipe_title", recipe_title, Field.Store.YES))

                wiki_title = recipe.get("wiki_title", "")
                if wiki_title:
                    doc.add(TextField("wiki_title", wiki_title, Field.Store.YES))

                description = recipe.get("description", "")
                if description:
                    doc.add(TextField("description", description, Field.Store.YES))

                wiki_description = recipe.get("wiki_description", "")
                if wiki_description:
                    doc.add(
                        TextField("wiki_description", wiki_description, Field.Store.YES)
                    )

                ingredients = recipe.get("ingredients", [])
                if ingredients:
                    ingredients_text = (
                        " ".join(ingredients)
                        if isinstance(ingredients, list)
                        else str(ingredients)
                    )
                    doc.add(TextField("ingredients", ingredients_text, Field.Store.YES))

                method = recipe.get("method", "")
                if method:
                    doc.add(TextField("method", method, Field.Store.YES))

                chef = recipe.get("chef", "")
                if chef:
                    doc.add(TextField("chef", chef, Field.Store.YES))

                difficulty = recipe.get("difficulty", "")
                if difficulty:
                    doc.add(TextField("difficulty", difficulty, Field.Store.YES))

                for field in ["wiki_course", "wiki_origin", "wiki_serving_temp"]:
                    val = recipe.get(field, "")
                    if val:
                        doc.add(TextField(field, val, Field.Store.YES))

                categories = recipe.get("wiki_categories", [])
                if categories:
                    cat_text = (
                        " ".join(categories)
                        if isinstance(categories, list)
                        else str(categories)
                    )
                    doc.add(TextField("wiki_categories", cat_text, Field.Store.YES))

                for field_name in ["prep_time", "servings", "wiki_url"]:
                    value = recipe.get(field_name, "")
                    if value:
                        doc.add(StoredField(field_name, str(value)))

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

    writer.commit()
    writer.close()

    print("\n✓ Indexing complete!")
    print(f"Total recipes indexed: {indexed_count:,}")
    print("\nField weights:")
    for field, weight in sorted(
        FIELD_WEIGHTS.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {field}: {weight}")


def main():
    data_file = os.getenv("DATA_FILE", "/app/data/wiki_recipes.jsonl")
    index_dir = os.getenv("INDEX_DIR", "/app/index")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    if os.path.exists(index_dir) and not os.path.isdir(index_dir):
        print(f"Warning: {index_dir} exists but is not a directory. Removing it.")
        os.remove(index_dir)

    os.makedirs(index_dir, exist_ok=True)

    create_index(data_file, index_dir)
    print(f"\nIndex created successfully at: {index_dir}")


if __name__ == "__main__":
    main()
