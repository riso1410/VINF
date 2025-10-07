import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional
from markitdown import MarkItDown

import config


@dataclass
class RecipeMetadata:
    url: str = ""
    html_file: str = ""
    title: str = ""
    description: str = ""
    ingredients: List[str] = field(default_factory=list)
    method: str = ""
    prep_time: str = ""
    servings: str = ""
    difficulty: str = ""
    chef: str = ""


class RecipeScraper:
    
    def __init__(self):
        self.html_dir = config.RAW_HTML_DIR
        self.output_dir = config.SCRAPED_DIR
        self.recipes_file = os.path.join(self.output_dir, "recipes.jsonl")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.logger = config.setup_logging(config.SCRAPER_LOG)

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def extract_title(self, markdown_content: str) -> str:
        pattern = r'^# (.+)$'
        match = re.search(pattern, markdown_content, re.MULTILINE)
        if match:
            return self.clean_text(match.group(1))
        
        return ""

    def extract_description(self, markdown_content: str) -> str:
        pattern = r'\[Rate\]\(#vote\).*?(?:\n\s*\* !\[A fallback image for Food Network UK\][^\n]*)+\s*\n\s*(.*?)(?=\n\nFeatured In:)'
        match = re.search(pattern, markdown_content, re.DOTALL)
        if match:
            description = match.group(1).strip()
            # Clean up any remaining noise
            description = re.sub(r'!\[.*?\]\(.*?\)', '', description)  # Remove images
            # Remove any links
            description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', description)
            return self.clean_text(description)
        
        return ""

    def extract_ingredients(self, markdown_content: str) -> List[str]:
        ingredients = []
        
        pattern = r'## Ingredients\s*\n(.*?)(?=\n## |\n\nRead More|$)'
        match = re.search(pattern, markdown_content, re.DOTALL | re.IGNORECASE)
        
        if match:
            ingredients_section = match.group(1)
            ingredient_pattern = r'\* \[ \] (.+?)(?=\n|$)'
            matches = re.findall(ingredient_pattern, ingredients_section)
            
            for match in matches:
                ingredient_text = self.clean_text(match)
                if not ingredient_text.endswith(':'):
                    ingredients.append(ingredient_text)
        
        return ingredients
    
    def extract_method(self, markdown_content: str) -> str:
        pattern = r'## Method\s*\n(.*?)(?=\n\*Copyright|\n\*From Food Network|\nRead More\s*\n\s*Rate this recipe|## Related Recipes)'
        match = re.search(pattern, markdown_content, re.DOTALL | re.IGNORECASE)
        
        if match:
            method_text = match.group(1).strip()
            method_text = re.sub(r'\s*Read More\s*$', '', method_text, flags=re.IGNORECASE)
            return self.clean_text(method_text)
        
        return ""

    def extract_author(self, markdown_content: str) -> str:
        pattern = r'\[([^\]]+)\]\(https://foodnetwork\.co\.uk/chefs/[^\)]*?"Go to Author"\)'
        match = re.search(pattern, markdown_content)
        if match:
            return self.clean_text(match.group(1))
        
        return ""

    def extract_prep_time(self, markdown_content: str) -> str:
        # Look for time after time icon - handles formats like "15 MINS", "1 HRS 30 MINS", etc.
        pattern = r'!\[A fallback image for Food Network UK\]\(/images/time-icon\.svg\)((?:\d+\s+HRS?)?\s*\d+\s+(?:MINS?|HRS?))'
        match = re.search(pattern, markdown_content, re.IGNORECASE)
        if match:
            time_str = match.group(1).strip()
            # Normalize time units
            time_str = re.sub(r'\bMINS?\b', 'min', time_str, flags=re.IGNORECASE)
            time_str = re.sub(r'\bHRS?\b', 'hr', time_str, flags=re.IGNORECASE)
            return self.clean_text(time_str)
        
        return ""

    def extract_servings(self, markdown_content: str) -> str:
        pattern = r'!\[A fallback image for Food Network UK\]\(/images/serves-icon\.svg\)(\d+)'
        match = re.search(pattern, markdown_content)
        if match:
            return self.clean_text(match.group(1))
        
        return ""
    
    def extract_difficulty(self, markdown_content: str) -> str:
        pattern = r'!\[A fallback image for Food Network UK\]\(/images/difficulty-icon\.svg\)([A-Za-z\s]+?)(?=\n|!\[)'
        match = re.search(pattern, markdown_content)
        if match:
            difficulty = self.clean_text(match.group(1))
            if not difficulty.isdigit() and len(difficulty) < 50:
                return difficulty
        
        return ""
    
    def html_filename_to_url(self, filename: str) -> str:
        url = filename.replace('.html', '')
        url = url.replace('_', '/')
        
        if not url.startswith('http'):
            url = 'https://' + url
        
        return url
    
    def get_html_file_path(self, url: str) -> Optional[str]:
        filename = url.replace('https://', '').replace('http://', '')
        filename = filename.replace('/', '_')
        
        if not filename.endswith('.html'):
            filename += '.html'
        
        html_path = os.path.join(self.html_dir, filename)
        
        if os.path.exists(html_path):
            return html_path
        
        return None

    def extract_recipe_metadata(self, url: str = None, html_file: str = None) -> Optional[RecipeMetadata]:
        if html_file:
            if not os.path.exists(html_file):
                self.logger.warning(f"HTML file not found: {html_file}")
                return None
            if not url:
                filename = os.path.basename(html_file)
                url = self.html_filename_to_url(filename)
        elif url:
            html_file = self.get_html_file_path(url)
            if not html_file:
                self.logger.warning(f"HTML file not found for URL: {url}")
                return None
        else:
            self.logger.error("Either url or html_file must be provided")
            return None

        normalized_html_file = html_file.replace('\\', '/')

        # Read the markdown content directly
        md = MarkItDown()
        result = md.convert(html_file)
        markdown_content = result.text_content if hasattr(result, 'text_content') else str(result)

        # Save markdown file
        # md_file_path = html_file.replace('.html', '.md')
        # try:
        #     with open(md_file_path, 'w', encoding='utf-8') as md_file:
        #         md_file.write(markdown_content)
        #     self.logger.info(f"Saved markdown file: {md_file_path}")
        # except Exception as e:
        #     self.logger.warning(f"Failed to save markdown file {md_file_path}: {e}")

        metadata = RecipeMetadata(
            url=url,
            html_file=normalized_html_file,
            title=self.extract_title(markdown_content),
            description=self.extract_description(markdown_content),
            method=self.extract_method(markdown_content),
            ingredients=self.extract_ingredients(markdown_content),
            prep_time=self.extract_prep_time(markdown_content),
            servings=self.extract_servings(markdown_content),
            difficulty=self.extract_difficulty(markdown_content),
            chef=self.extract_author(markdown_content)
        )
        
        return metadata

    def save_to_jsonl(self, metadata: RecipeMetadata):
        data_dict = {
            'url': metadata.url,
            'html_file': metadata.html_file,
            'title': metadata.title,
            'description': metadata.description,
            'method': metadata.method,
            'ingredients': metadata.ingredients,
            'prep_time': metadata.prep_time,
            'servings': metadata.servings,
            'difficulty': metadata.difficulty,
            'chef': metadata.chef
        }
        
        with open(self.recipes_file, 'a', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False)
            f.write('\n')

    def is_valid_recipe_file(self, filename: str) -> bool:
        name_without_ext = filename.replace('.html', '')
        
        parts = name_without_ext.split('_')
        
        if len(parts) != 3:
            return False
        
        try:
            recipes_index = parts.index('recipes')
        except ValueError:
            return False
        
        if recipes_index != 1:
            return False
        
        if len(parts) - recipes_index != 2:
            return False
        
        return True
    
    def scrape_all_recipes(self):
        if not os.path.exists(self.html_dir):
            self.logger.error(f"HTML directory not found: {self.html_dir}")
            return
        
        all_html_files = [f for f in os.listdir(self.html_dir) if f.endswith('.html')]
        
        html_files = [f for f in all_html_files if self.is_valid_recipe_file(f)]

        if not html_files:
            self.logger.error(f"No valid recipe HTML files found in: {self.html_dir}")
            return
                
        if os.path.exists(self.recipes_file):
            os.remove(self.recipes_file)
        
        self.logger.info(f"Starting to scrape {len(html_files)} HTML files from {self.html_dir}")
        
        processed_count = 0
        successful_count = 0
        
        for html_filename in html_files:
            html_path = os.path.join(self.html_dir, html_filename)
            url = self.html_filename_to_url(html_filename)

            metadata = self.extract_recipe_metadata(url=url, html_file=html_path)
            if metadata:
                self.save_to_jsonl(metadata)
                successful_count += 1
                            
            processed_count += 1
                    
        self.logger.info(f"Scraping completed. Processed {processed_count} files, extracted {successful_count} recipes")
        self.logger.info(f"All recipes saved to: {self.recipes_file}")


def main():
    scraper = RecipeScraper()
    scraper.scrape_all_recipes()

if __name__ == "__main__":
    main()