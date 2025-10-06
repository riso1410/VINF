"""
Recipe scraper for extracting structured data from HTML files.
Parses JSON-LD structured data and HTML patterns to extract recipe information.
"""

import os
import json
import re
import html
from dataclasses import dataclass, field
from typing import List, Optional
import config


@dataclass
class RecipeMetadata:
    """Data class representing structured recipe information."""
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
    """
    Scraper for extracting recipe data from HTML files.
    Supports JSON-LD structured data and HTML pattern matching.
    """
    
    def __init__(self, html_dir: str = None, output_dir: str = None):
        """
        Initialize the recipe scraper.
        
        Args:
            html_dir: Directory containing HTML files to scrape
            output_dir: Directory to save scraped recipe data
        """
        self.html_dir = html_dir if html_dir is not None else config.RAW_HTML_DIR
        self.output_dir = output_dir if output_dir is not None else config.SCRAPED_DIR
        self.recipes_file = os.path.join(self.output_dir, "recipes.jsonl")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.logger = config.setup_logging(config.SCRAPER_LOG)

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags and normalizing whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def extract_json_ld_recipe(self, html_content: str) -> dict:
        """
        Extract recipe from JSON-LD structured data.
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            dict: Recipe data from JSON-LD or empty dict
        """
        pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_match = match.strip()
            try:
                data = json.loads(clean_match)
                if data and data.get('@type') == 'Recipe':
                    return data
            except json.JSONDecodeError:
                continue
        
        return {}

    def extract_recipe_from_js_match(self, js_text: str) -> dict:
        """
        Extract recipe data from JavaScript text match.
        
        Args:
            js_text: JavaScript text containing potential recipe data
            
        Returns:
            dict: Recipe data or empty dict
        """
        try:
            data = json.loads(js_text)
            if data and isinstance(data, dict):
                if data.get('@type') == 'Recipe':
                    return data
                
                recipe = self.find_recipe_in_nested_object(data)
                if recipe:
                    return recipe
        except json.JSONDecodeError:
            pass
        
        return {}
                
    def find_recipe_in_nested_object(self, obj: dict) -> dict:
        """
        Recursively search for recipe data in nested objects.
        
        Args:
            obj: Object to search
            
        Returns:
            dict: Recipe data or empty dict
        """
        if isinstance(obj, dict):
            if obj.get('@type') == 'Recipe':
                return obj
        return {}

    def extract_title(self, json_data: dict) -> str:
        """
        Extract recipe title.
        
        Args:
            json_data: Recipe JSON-LD data
            
        Returns:
            str: Recipe title
        """
        return self.clean_text(json_data.get('name', ''))

    def extract_description(self, json_data: dict, html_content: str = "") -> str:
        """
        Extract recipe description.
        
        Args:
            json_data: Recipe JSON-LD data
            html_content: HTML content for fallback extraction
            
        Returns:
            str: Recipe description
        """
        description = self.clean_text(json_data.get('description', ''))
        
        if not description and html_content:
            og_desc_pattern = r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']'
            og_match = re.search(og_desc_pattern, html_content, re.IGNORECASE)
            if og_match:
                description = self.clean_text(og_match.group(1))
            
            if not description:
                desc_pattern = r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']'
                desc_match = re.search(desc_pattern, html_content, re.IGNORECASE)
                if desc_match:
                    description = self.clean_text(desc_match.group(1))
        
        return description

    def extract_ingredients(self, json_data: dict) -> List[str]:
        """
        Extract ingredients from JSON-LD structured data.
        
        Args:
            json_data: Recipe JSON-LD data
            
        Returns:
            List[str]: List of ingredient strings
        """
        if json_data and 'recipeIngredient' in json_data:
            ingredients = self.filter_ingredients(json_data['recipeIngredient'])
            return ingredients
        
        return []
    
    def filter_ingredients(self, recipe_ingredients: list) -> List[str]:
        """
        Filter and clean ingredient list.
        
        Args:
            recipe_ingredients: Raw ingredient list
            
        Returns:
            List[str]: Filtered ingredient list
        """
        ingredients = []
        for ingredient in recipe_ingredients:
            ingredient_text = str(ingredient).strip()
            
            if (ingredient_text and 
                not ingredient_text.endswith(':') and 
                not ingredient_text.startswith('For the')):
                ingredients.append(ingredient_text)
        
        return ingredients
    
    def extract_method(self, html_content: str, json_data: dict) -> str:
        """
        Extract recipe method as clean text.
        
        Args:
            html_content: HTML content
            json_data: Recipe JSON-LD data
            
        Returns:
            str: Recipe instructions
        """
        instructions = json_data.get('recipeInstructions', [])
        if instructions:
            instruction_texts = []
            
            if isinstance(instructions, str):
                return self.clean_text(instructions)
            elif isinstance(instructions, list):
                for instruction in instructions:
                    if isinstance(instruction, dict):
                        text = instruction.get('text', '')
                        if text:
                            instruction_texts.append(self.clean_text(text))
                    elif isinstance(instruction, str):
                        instruction_texts.append(self.clean_text(instruction))
                
                if instruction_texts:
                    return ' '.join(instruction_texts)
            
        return ""

    def extract_author(self, json_data: dict) -> str:
        """
        Extract recipe author.
        
        Args:
            json_data: Recipe JSON-LD data
            
        Returns:
            str: Author name
        """
        author = json_data.get('author', {})
        return self.clean_text(author.get('name', ''))

    def extract_prep_time(self, html_content: str) -> str:
        """
        Extract preparation time from HTML.
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            str: Formatted preparation time
        """
        pattern = r'<img[^>]*time-icon\.svg[^>]*>.*?<span[^>]*>\s*([^<]+?(?:MINS?|HRS?))\s*</span>'
        match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
        if match:
            time_str = self.clean_text(match.group(1))
            time_str = re.sub(r'\bMINS?\b', 'min', time_str, flags=re.IGNORECASE)
            time_str = re.sub(r'\bHRS?\b', 'hr', time_str, flags=re.IGNORECASE)
            return time_str
        return ""

    def extract_servings(self, json_data: dict) -> str:
        """
        Extract servings information.
        
        Args:
            json_data: Recipe JSON-LD data
            
        Returns:
            str: Servings information
        """
        if json_data.get('recipeYield'):
            yield_value = json_data['recipeYield']
            return self.clean_text(str(yield_value))
                
        return ""
    
    def extract_difficulty(self, html_content: str) -> str:
        """
        Extract difficulty from HTML patterns.
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            str: Recipe difficulty level
        """
        pattern = r'<li[^>]*class="[^"]*p-2[^"]*"[^>]*>\s*<span[^>]*>\s*<img[^>]*difficulty-icon[^>]*>\s*<span[^>]*>\s*([^<]+)\s*</span>'
        match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_text(match.group(1))
        
        return ""
    
    def html_filename_to_url(self, filename: str) -> str:
        """
        Convert HTML filename back to URL.
        
        Args:
            filename: HTML filename
            
        Returns:
            str: Reconstructed URL
        """
        url = filename.replace('.html', '')
        url = url.replace('_', '/')
        
        if not url.startswith('http'):
            url = 'https://' + url
        
        return url
    
    def get_html_file_path(self, url: str) -> Optional[str]:
        """
        Get the HTML file path for a given URL.
        
        Args:
            url: URL to find corresponding HTML file for
            
        Returns:
            str or None: File path if found, None otherwise
        """
        filename = url.replace('https://', '').replace('http://', '')
        filename = filename.replace('/', '_')
        
        if not filename.endswith('.html'):
            filename += '.html'
        
        html_path = os.path.join(self.html_dir, filename)
        
        if os.path.exists(html_path):
            return html_path
        
        return None

    def extract_recipe_metadata(self, url: str = None, html_file: str = None) -> Optional[RecipeMetadata]:
        """
        Extract complete recipe metadata from HTML file.
        
        Args:
            url: Recipe URL (optional if html_file is provided)
            html_file: HTML file path (optional if url is provided)
            
        Returns:
            RecipeMetadata or None: Extracted recipe data or None if extraction fails
        """
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
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        json_data = self.extract_json_ld_recipe(html_content)
        normalized_html_file = html_file.replace('\\', '/')
        
        metadata = RecipeMetadata(
            url=url,
            html_file=normalized_html_file,
            title=self.extract_title(json_data),
            description=self.extract_description(json_data, html_content),
            method=self.extract_method(html_content, json_data),
            ingredients=self.extract_ingredients(json_data),
            prep_time=self.extract_prep_time(html_content),
            servings=self.extract_servings(json_data),
            difficulty=self.extract_difficulty(html_content),
            chef=self.extract_author(json_data)
        )
        
        return metadata

    def save_to_jsonl(self, metadata: RecipeMetadata):
        """
        Save single recipe to the main recipes.jsonl file.
        
        Args:
            metadata: Recipe metadata to save
        """
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
        """
        Check if filename matches recipe file pattern.
        
        Args:
            filename: Filename to validate
            
        Returns:
            bool: True if filename is a valid recipe file
        """
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
        """Main method to scrape all recipes to a single recipes.jsonl file."""
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
    """Main function to run the recipe scraper."""
    scraper = RecipeScraper()
    scraper.scrape_all_recipes()


if __name__ == "__main__":
    main()