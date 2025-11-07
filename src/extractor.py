import json
import os
from typing import Optional

import config
from recipe_parser import (
    RecipeMetadata,
    html_filename_to_url,
    metadata_to_dict,
    parse_recipe_html,
    should_skip_metadata,
    url_to_html_path,
)


class RecipeScraper:
    def __init__(self):
        self.html_dir = config.RAW_HTML_DIR
        self.output_dir = config.SCRAPED_DIR
        self.recipes_file = os.path.join(self.output_dir, "recipes.jsonl")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)

        self.logger = config.setup_logging(config.SCRAPER_LOG)

    def extract_recipe_metadata(
        self, url: str = None, html_file: str = None
    ) -> Optional[RecipeMetadata]:
        if html_file:
            if not os.path.exists(html_file):
                self.logger.warning(f"HTML file not found: {html_file}")
                return None
        elif url:
            html_file = url_to_html_path(url, self.html_dir)
            if not html_file:
                self.logger.warning(f"HTML file not found for URL: {url}")
                return None
        else:
            self.logger.error("Either url or html_file must be provided")
            return None

        return parse_recipe_html(
            html_file,
            url=url,
            logger=self.logger,
        )

    def save_to_jsonl(self, metadata: RecipeMetadata):
        if should_skip_metadata(metadata):
            return

        data_dict = metadata_to_dict(metadata)

        with open(self.recipes_file, "a", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False)
            f.write("\n")

    def scrape_all_recipes(self):
        if not os.path.exists(self.html_dir):
            self.logger.error(f"HTML directory not found: {self.html_dir}")
            return

        html_files = [f for f in os.listdir(self.html_dir) if f.endswith(".html")]

        if not html_files:
            self.logger.error(f"No valid recipe HTML files found in: {self.html_dir}")
            return

        if os.path.exists(self.recipes_file):
            os.remove(self.recipes_file)

        self.logger.info(
            f"Starting to scrape {len(html_files)} HTML files from {self.html_dir}"
        )

        processed_count = 0
        successful_count = 0

        total_files = len(html_files)
        for idx, html_filename in enumerate(html_files, start=1):
            if idx % 10 == 0 or idx == total_files:
                self.logger.info(f"Processing file {idx} out of {total_files}")
            html_path = os.path.join(self.html_dir, html_filename)
            url = html_filename_to_url(html_filename)

            metadata = self.extract_recipe_metadata(url=url, html_file=html_path)
            if metadata:
                self.save_to_jsonl(metadata)
                successful_count += 1

            processed_count += 1

        self.logger.info(
            f"Scraping completed. Processed {processed_count} files, extracted {successful_count} recipes"
        )
        self.logger.info(f"All recipes saved to: {self.recipes_file}")


def main():
    scraper = RecipeScraper()
    scraper.scrape_all_recipes()


if __name__ == "__main__":
    main()
