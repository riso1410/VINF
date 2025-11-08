import bz2
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

import config


def sanitize_ingredient_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.lower().split())
    return text.strip()


def find_balanced_braces(text: str, start_pos: int) -> int:
    if start_pos >= len(text) - 1 or not text[start_pos:].startswith('{{'):
        return -1

    depth = 0
    i = start_pos

    while i < len(text) - 1:
        if text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif text[i:i+2] == '}}':
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1

    return -1


def extract_ingredients_from_wikitext(wikitext: str) -> list[str]:
    ingredients = set()
    infobox_pattern = r'\{\{[Ii]nfobox\s+'
    infobox_starts = [m.start() for m in re.finditer(infobox_pattern, wikitext)]

    for start_pos in infobox_starts:
        end_pos = find_balanced_braces(wikitext, start_pos)
        if end_pos == -1:
            continue

        infobox = wikitext[start_pos:end_pos]

        if not re.search(r'\|\s*\w*ingredient[s]?\s*=', infobox, re.IGNORECASE):
            continue

        field_pattern = r'\|\s*\w*ingredient[s]?\s*=\s*(.+?)(?=\s*\||$)'
        fields = re.findall(field_pattern, infobox, re.IGNORECASE | re.DOTALL)

        for field in fields:
            field = field.strip()

            while '{{' in field:
                template_start = field.find('{{')
                template_end = find_balanced_braces(field, template_start)

                if template_end > 0:
                    template = field[template_start:template_end]
                    template_content = re.sub(r'\{\{[^|]+\|', '', template, count=1)
                    template_content = template_content.rstrip('}')
                    field = field[:template_start] + template_content + field[template_end:]
                else:
                    field = re.sub(r'\{\{[^}]*$', '', field)
                    break

            text = re.sub(r'\[\[(?:[^\]]*\|)?([^\]]+)\]\]', r'\1', field)

            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)

            text = re.sub(r'\{\{[^}]*\}\}', '', text)

            text = re.sub(r'^\s*\*\s*', '', text, flags=re.MULTILINE)

            for part in re.split(r'[,;\n]', text):
                token = sanitize_ingredient_text(part)
                if len(token) > 2:
                    ingredients.add(token)

    return list(ingredients)


def sanitize_filename(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", title)


def compute_title_similarity(title1: str, title2: str) -> float:
    if not title1 or not title2:
        return 0.0
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()


def compute_jaccard_similarity(set1: list, set2: list) -> float:
    if not set1 or not set2:
        return 0.0
    s1 = set(set1)
    s2 = set(set2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def parse_multistream_index(index_path: str) -> dict[int, list[tuple[int, str]]]:
    stream_map = defaultdict(list)

    with bz2.open(index_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":", 2)
            if len(parts) == 3:
                offset = int(parts[0])
                article_id = int(parts[1])
                title = parts[2]
                stream_map[offset].append((article_id, title))

    return dict(stream_map)


def read_stream_at_offset(dump_path: str, offset: int, next_offset: int = None) -> str:
    with open(dump_path, "rb") as f:
        f.seek(offset)

        if next_offset:
            compressed_data = f.read(next_offset - offset)
        else:
            compressed_data = bytearray()
            chunk = f.read(1)
            if not chunk:
                return ""
            compressed_data.extend(chunk)

            while True:
                chunk = f.read(8192)
                if not chunk:
                    break

                bz_marker = b'BZh'
                marker_pos = chunk.find(bz_marker)
                if marker_pos > 0:
                    compressed_data.extend(chunk[:marker_pos])
                    break
                compressed_data.extend(chunk)

    try:
        decompressed = bz2.decompress(bytes(compressed_data))
        return decompressed.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def parse_xml_pages(xml_text: str) -> list[dict]:
    pages = []

    if not xml_text.strip().startswith("<mediawiki"):
        xml_text = f"<mediawiki>{xml_text}</mediawiki>"

    try:
        root = ET.fromstring(xml_text)

        for page in root.findall(".//page"):
            title_elem = page.find("title")
            ns_elem = page.find("ns")
            text_elem = page.find(".//revision/text")

            if title_elem is not None and text_elem is not None:
                title = title_elem.text or ""
                ns = ns_elem.text or "0"
                wikitext = text_elem.text or ""

                pages.append({
                    "title": title,
                    "ns": ns,
                    "wikitext": wikitext
                })
    except ET.ParseError:
        page_pattern = r"<page>.*?</page>"
        for page_match in re.finditer(page_pattern, xml_text, re.DOTALL):
            page_xml = page_match.group(0)
            title_match = re.search(r"<title>(.*?)</title>", page_xml)
            ns_match = re.search(r"<ns>(.*?)</ns>", page_xml)
            text_match = re.search(r"<text[^>]*>(.*?)</text>", page_xml, re.DOTALL)

            if title_match and text_match:
                pages.append({
                    "title": title_match.group(1),
                    "ns": ns_match.group(1) if ns_match else "0",
                    "wikitext": text_match.group(1)
                })

    return pages


def save_wiki_page_xml(title: str, wikitext: str, output_dir: str):
    filename = f"{sanitize_filename(title)}.xml"
    filepath = os.path.join(output_dir, filename)

    # Escape XML special characters
    wikitext_escaped = (
        wikitext.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

    page_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<page>
  <title>{title}</title>
  <text>{wikitext_escaped}</text>
</page>
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(page_xml)


def process_stream_with_recipes(args):
    offset, next_offset, dump_path, recipes_broadcast, wiki_pages_dir = args

    xml_text = read_stream_at_offset(dump_path, offset, next_offset)
    if not xml_text:
        return (offset, [], [])

    pages = parse_xml_pages(xml_text)

    recipes = recipes_broadcast.value

    pages_with_ingredients = []
    matches = []

    for page in pages:
        if page["ns"] != "0":
            continue

        wiki_ingredients = extract_ingredients_from_wikitext(page["wikitext"])
        if not wiki_ingredients:
            continue

        wiki_title = page["title"]
        wiki_url = f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"

        pages_with_ingredients.append({
            "title": wiki_title,
            "ingredients": wiki_ingredients,
            "url": wiki_url
        })

        wiki_ingredients_set = set(wiki_ingredients)
        page_saved = False

        for recipe in recipes:
            recipe_title = recipe["title"]
            recipe_ingredients = recipe["ingredients"]

            recipe_ingredients_set = set(recipe_ingredients)
            if not wiki_ingredients_set & recipe_ingredients_set:
                continue

            title_sim = compute_title_similarity(recipe_title, wiki_title)
            ingredient_sim = compute_jaccard_similarity(recipe_ingredients, wiki_ingredients)

            match_score = (title_sim * 0.3) + (ingredient_sim * 0.7)

            if match_score > 0:
                if not page_saved:
                    try:
                        save_wiki_page_xml(wiki_title, page["wikitext"], wiki_pages_dir)
                        page_saved = True
                    except Exception:
                        pass

                matches.append({
                    "recipe_title": recipe_title,
                    "wiki_title": wiki_title,
                    "wiki_url": wiki_url,
                    "wiki_ingredients": wiki_ingredients,
                    "match_score": match_score,
                    "ingredient_similarity": ingredient_sim
                })

    return (offset, pages_with_ingredients, matches)


class WikiMapper:
    def __init__(self):
        self.recipes_file = config.RECIPES_FILE
        self.wiki_dump_path = config.WIKI_DUMP_PATH
        self.wiki_index_path = config.WIKI_INDEX_PATH
        self.wiki_pages_dir = config.WIKI_PAGES_DIR
        self.output_file = config.WIKI_RECIPES_OUTPUT
        self.checkpoint_dir = config.WIKI_CHECKPOINT_DIR

        self.pages_checkpoint = os.path.join(self.checkpoint_dir, "wiki_pages.parquet")
        self.matches_checkpoint = os.path.join(self.checkpoint_dir, "matches.jsonl")
        self.processed_streams_file = os.path.join(self.checkpoint_dir, "processed_streams.txt")

        os.makedirs(self.wiki_pages_dir, exist_ok=True)
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = config.setup_logging(config.WIKI_SPARK_LOG, config.LOG_LEVEL)

        self.spark = (
            SparkSession.builder
            .appName("WikiMapper")
            .master("local[*]")
            .config("spark.hadoop.io.compression.codecs",
                    "org.apache.hadoop.io.compress.BZip2Codec")
            .getOrCreate()
        )

        self.sc = self.spark.sparkContext
        self.logger.info("WikiMapper initialized with streaming architecture")

    def run(self):
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" WIKIPEDIA-RECIPE MAPPING PIPELINE (STREAMING)")
        self.logger.info("=" * 70)

        if os.path.exists(self.output_file):
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info(" OUTPUT ALREADY EXISTS")
            self.logger.info("=" * 70)
            self.logger.info(f"Found existing output: {self.output_file}")
            self.logger.info("Delete the file to reprocess or use checkpoint to resume")
            self.logger.info("=" * 70)
            return

        if not os.path.exists(self.wiki_dump_path):
            self.logger.error(f"Wikipedia dump not found: {self.wiki_dump_path}")
            return

        if not os.path.exists(self.wiki_index_path):
            self.logger.error(f"Multistream index not found: {self.wiki_index_path}")
            return

        if not os.path.exists(self.recipes_file):
            self.logger.error(f"Recipes file not found: {self.recipes_file}")
            return

        self.logger.info("Loading recipes...")
        with open(self.recipes_file, 'r', encoding='utf-8') as f:
            recipes = [json.loads(line) for line in f]
        self.logger.info(f"Loaded {len(recipes):,} recipes")

        recipes_broadcast = self.sc.broadcast(recipes)

        processed_streams = set()
        if os.path.exists(self.processed_streams_file):
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info(" RESUMING FROM CHECKPOINT")
            self.logger.info("=" * 70)
            with open(self.processed_streams_file, 'r') as f:
                processed_streams = set(int(line.strip()) for line in f)
            self.logger.info(f"Found {len(processed_streams):,} already processed streams")

            if os.path.exists(self.pages_checkpoint) and os.path.exists(self.matches_checkpoint):
                self.logger.info("Found existing checkpoints - loading...")

                with open(self.matches_checkpoint, 'r', encoding='utf-8') as f:
                    existing_matches = [json.loads(line) for line in f]

                self.logger.info(f"Loaded {len(existing_matches):,} existing matches")
                self.logger.info("Will continue processing remaining streams...")
            else:
                existing_matches = []
        else:
            existing_matches = []

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" PROCESSING WIKIPEDIA STREAMS")
        self.logger.info("=" * 70)
        self.logger.info(f"Reading index from: {self.wiki_index_path}")

        stream_map = parse_multistream_index(self.wiki_index_path)
        offsets = sorted(stream_map.keys())
        self.logger.info(f"Found {len(offsets):,} streams")

        offset_ranges = []
        for i, offset in enumerate(offsets):
            if offset not in processed_streams:
                next_offset = offsets[i + 1] if i + 1 < len(offsets) else None
                offset_ranges.append((offset, next_offset, self.wiki_dump_path, recipes_broadcast, self.wiki_pages_dir))

        if not offset_ranges:
            self.logger.info("All streams already processed!")
            self.logger.info("Creating final output from checkpoint...")
            self._create_final_output(existing_matches, recipes)
            return

        self.logger.info(f"Streams to process: {len(offset_ranges):,} (skipping {len(processed_streams):,} already done)")

        num_partitions = min(500, len(offset_ranges))
        self.logger.info(f"Using {num_partitions} partitions")
        self.logger.info("Processing streams and matching to recipes...")
        self.logger.info("This will take 30-60 minutes...")
        self.logger.info("Progress will be checkpointed incrementally")

        streams_rdd = self.sc.parallelize(offset_ranges, numSlices=num_partitions)
        results_rdd = streams_rdd.map(process_stream_with_recipes)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" COLLECTING RESULTS")
        self.logger.info("=" * 70)

        all_results = results_rdd.collect()

        new_pages = []
        new_matches = []
        new_processed_offsets = []

        for offset, pages, matches in all_results:
            new_pages.extend(pages)
            new_matches.extend(matches)
            new_processed_offsets.append(offset)

        self.logger.info(f"Collected {len(new_pages):,} Wikipedia pages with ingredients")
        self.logger.info(f"Collected {len(new_matches):,} new matches")

        if new_pages:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info(" CHECKPOINTING WIKIPEDIA PAGES (PARQUET)")
            self.logger.info("=" * 70)
            self.logger.info(f"Checkpoint path: {self.pages_checkpoint}")
            self.logger.info(f"New pages to save: {len(new_pages):,}")

            schema = T.StructType([
                T.StructField("title", T.StringType(), False),
                T.StructField("ingredients", T.ArrayType(T.StringType()), False),
                T.StructField("url", T.StringType(), False),
            ])

            new_pages_df = self.spark.createDataFrame(new_pages, schema)

            if os.path.exists(self.pages_checkpoint):
                self.logger.info("Appending to existing checkpoint...")
                existing_pages_df = self.spark.read.parquet(self.pages_checkpoint)
                existing_count = existing_pages_df.count()
                self.logger.info(f"Existing pages: {existing_count:,}")

                combined_pages_df = existing_pages_df.union(new_pages_df)
                combined_pages_df.write.mode("overwrite").parquet(self.pages_checkpoint)

                total_count = combined_pages_df.count()
                self.logger.info(f"Total pages after merge: {total_count:,}")
            else:
                self.logger.info("Creating new checkpoint (first write)...")
                new_pages_df.write.mode("overwrite").parquet(self.pages_checkpoint)
                self.logger.info(f"Saved {len(new_pages):,} pages")

            checkpoint_size = 0
            for root, dirs, files in os.walk(self.pages_checkpoint):
                for file in files:
                    checkpoint_size += os.path.getsize(os.path.join(root, file))

            self.logger.info(f"Checkpoint size: {checkpoint_size / (1024*1024):.2f} MB")
            self.logger.info("✅ Pages checkpoint saved successfully!")
            self.logger.info("=" * 70)

        if new_matches:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info(" CHECKPOINTING MATCHES (JSONL)")
            self.logger.info("=" * 70)
            self.logger.info(f"Checkpoint path: {self.matches_checkpoint}")
            self.logger.info(f"New matches to save: {len(new_matches):,}")

            mode = 'a' if os.path.exists(self.matches_checkpoint) else 'w'
            if mode == 'a':
                with open(self.matches_checkpoint, 'r', encoding='utf-8') as f:
                    existing_count = sum(1 for _ in f)
                self.logger.info(f"Existing matches: {existing_count:,}")
                self.logger.info("Appending new matches...")
            else:
                self.logger.info("Creating new checkpoint (first write)...")

            with open(self.matches_checkpoint, mode, encoding='utf-8') as f:
                for match in new_matches:
                    json.dump(match, f, ensure_ascii=False)
                    f.write('\n')

            file_size = os.path.getsize(self.matches_checkpoint)
            self.logger.info(f"Checkpoint size: {file_size / 1024:.2f} KB")

            with open(self.matches_checkpoint, 'r', encoding='utf-8') as f:
                total_count = sum(1 for _ in f)
            self.logger.info(f"Total matches in checkpoint: {total_count:,}")
            self.logger.info("✅ Matches checkpoint saved successfully!")
            self.logger.info("=" * 70)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" CHECKPOINTING PROCESSED STREAMS")
        self.logger.info("=" * 70)
        self.logger.info(f"Checkpoint path: {self.processed_streams_file}")
        self.logger.info(f"New streams processed: {len(new_processed_offsets):,}")

        if os.path.exists(self.processed_streams_file):
            with open(self.processed_streams_file, 'r') as f:
                existing_count = sum(1 for _ in f)
            self.logger.info(f"Previously processed: {existing_count:,}")

        with open(self.processed_streams_file, 'a') as f:
            for offset in new_processed_offsets:
                f.write(f"{offset}\n")

        with open(self.processed_streams_file, 'r') as f:
            total_processed = sum(1 for _ in f)

        self.logger.info(f"Total streams processed: {total_processed:,} / {len(offsets):,}")
        progress_pct = (total_processed / len(offsets)) * 100 if offsets else 0
        self.logger.info(f"Overall progress: {progress_pct:.1f}%")

        if total_processed > len(processed_streams):
            streams_per_run = total_processed - len(processed_streams)
            remaining_streams = len(offsets) - total_processed
            self.logger.info(f"Remaining streams: {remaining_streams:,}")
            self.logger.info(f"(Processing ~{streams_per_run:,} streams per run)")

        self.logger.info("✅ Streams checkpoint saved successfully!")
        self.logger.info("=" * 70)

        all_matches = existing_matches + new_matches
        self.logger.info("")
        self.logger.info(f"📊 Total matches accumulated: {len(all_matches):,}")
        self.logger.info("")

        self._create_final_output(all_matches, recipes)

    def _create_final_output(self, all_matches, recipes):
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" COMPUTING BEST MATCHES")
        self.logger.info("=" * 70)

        schema = T.StructType([
            T.StructField("recipe_title", T.StringType(), False),
            T.StructField("wiki_title", T.StringType(), False),
            T.StructField("wiki_url", T.StringType(), False),
            T.StructField("wiki_ingredients", T.ArrayType(T.StringType()), False),
            T.StructField("match_score", T.FloatType(), False),
            T.StructField("ingredient_similarity", T.FloatType(), False),
        ])

        matches_df = self.spark.createDataFrame(all_matches, schema)

        total_matches = matches_df.count()
        self.logger.info(f"Total recipe-Wikipedia match candidates: {total_matches:,}")

        from pyspark.sql.window import Window
        window = Window.partitionBy("recipe_title").orderBy(F.desc("match_score"))
        best_matches = matches_df.withColumn(
            "rank", F.row_number().over(window)
        ).filter(F.col("rank") == 1).drop("rank")

        match_count = best_matches.count()
        self.logger.info(f"Best matches (one per recipe): {match_count:,}")

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" CREATING ENRICHED RECIPES")
        self.logger.info("=" * 70)

        recipes_df = self.spark.read.json(self.recipes_file)

        enriched_df = recipes_df.join(
            best_matches.select(
                F.col("recipe_title").alias("title"),
                "wiki_title",
                "wiki_url",
                "wiki_ingredients"
            ),
            on="title",
            how="left"
        )

        self.logger.info(f"Writing enriched recipes to: {self.output_file}")

        temp_output_dir = self.output_file + "_temp"
        enriched_df.coalesce(1).write.mode("overwrite").json(temp_output_dir)

        import glob
        import shutil
        part_files = glob.glob(os.path.join(temp_output_dir, "part-*.json"))
        if part_files:
            shutil.move(part_files[0], self.output_file)
            shutil.rmtree(temp_output_dir)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(" PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 70)
        self.logger.info(f"Enriched recipes: {self.output_file}")
        self.logger.info(f"Wiki pages saved: {self.wiki_pages_dir}")
        self.logger.info(f"Total recipes: {len(recipes):,}")
        self.logger.info(f"Matched recipes: {match_count:,}")
        self.logger.info(f"Match rate: {(match_count/len(recipes)*100):.1f}%")
        self.logger.info("=" * 70)

        self.spark.stop()


def main():
    mapper = WikiMapper()
    mapper.run()


if __name__ == "__main__":
    main()
