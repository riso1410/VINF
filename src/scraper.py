import os
import json
import re
import html
from dataclasses import dataclass, field
from typing import List, Optional
import config

@dataclass
class RecipeMetadata:
    """Recipe object structure"""
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
    def __init__(self, html_dir: str = None, output_dir: str = None):
        self.html_dir = html_dir if html_dir is not None else config.RAW_HTML_DIR
        self.output_dir = output_dir if output_dir is not None else config.SCRAPED_DIR
        self.recipes_file = os.path.join(self.output_dir, "recipes.jsonl")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.logger = config.setup_logging(config.SCRAPER_LOG)

    def clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and normalizing whitespace"""
        if not text:
            return ""
        
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def extract_json_ld_recipe(self, html_content: str) -> dict:
        """Extract Recipe from JSON-LD structured data"""
        pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_match = match.strip()
            data = self._try_parse_json(clean_match)
            
            if data:
                recipe = self._extract_recipe_from_data(data)
                if recipe:
                    return recipe
        
        return {}
    
    def _try_parse_json(self, json_str: str) -> Optional[dict]:
        """Try to parse JSON string, return None if it fails"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.debug(f"Failed to parse JSON-LD: {e}")
            return None
    
    def _extract_recipe_from_data(self, data) -> Optional[dict]:
        """Extract recipe from parsed JSON data"""
        if data.get('@type') == 'Recipe':
            return data
        
        return None

    def extract_recipe_from_js_match(self, js_text: str) -> dict:
        """Extract recipe data from JavaScript text match"""
        data = self._try_parse_json(js_text)
        if data and isinstance(data, dict):
            if data.get('@type') == 'Recipe':
                return data
            
            recipe = self.find_recipe_in_nested_object(data)
            if recipe:
                return recipe
                
    def find_recipe_in_nested_object(self, obj: dict) -> dict:
        """Recursively search for recipe data in nested objects"""
        if isinstance(obj, dict):
            if obj.get('@type') == 'Recipe':
                return obj
        return {}

    def extract_title(self, json_data: dict) -> str:
        """Extract recipe title"""
        return self.clean_text(json_data.get('name', ''))

    def extract_description(self, json_data: dict, html_content: str = "") -> str:
        """Extract recipe description"""
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
        """Extract ingredients from JSON-LD structured data"""
        if json_data and 'recipeIngredient' in json_data:
            ingredients = self._filter_ingredients(json_data['recipeIngredient'])
            return ingredients
        
        return []
    
    def _filter_ingredients(self, recipe_ingredients: list) -> List[str]:
        """Filter and clean ingredient list"""
        ingredients = []
        for ingredient in recipe_ingredients:
            ingredient_text = str(ingredient).strip()
            
            if (ingredient_text and 
                not ingredient_text.endswith(':') and 
                not ingredient_text.startswith('For the')):
                ingredients.append(ingredient_text)
        
        return ingredients
    
    def extract_method(self, html_content: str, json_data: dict) -> str:
        """Extract recipe method as clean text"""
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
        """Extract recipe author"""
        author = json_data.get('author', {})
        return self.clean_text(author.get('name', ''))

    def format_duration(self, duration_str: str) -> str:
        """Convert ISO 8601 duration format to readable format"""
        if not duration_str:
            return ""
        
        if duration_str.startswith('PT'):
            duration = duration_str[2:]  
            
            hours = 0
            minutes = 0
            
            if 'H' in duration:
                hours_match = re.search(r'(\d+)H', duration)
                if hours_match:
                    hours = int(hours_match.group(1))
            
            if 'M' in duration:
                minutes_match = re.search(r'(\d+)M', duration)
                if minutes_match:
                    minutes = int(minutes_match.group(1))
            
            # Format the output
            result_parts = []
            if hours > 0:
                result_parts.append(f"{hours} hr" if hours == 1 else f"{hours} hrs")
            if minutes > 0:
                result_parts.append(f"{minutes} min")
            
            return " ".join(result_parts) if result_parts else duration_str
        
        return duration_str

    def extract_prep_time(self, json_data: dict) -> str:
        """Extract preparation time"""
        if json_data.get('prepTime'):
            return self.format_duration(self.clean_text(json_data['prepTime']))
        return ""

    def extract_servings(self, json_data: dict) -> str:
        """Extract servings information"""
        if json_data.get('recipeYield'):
            yield_value = json_data['recipeYield']
            return self.clean_text(str(yield_value))
                
        return ""
    
    def extract_difficulty(self, html_content: str) -> str:
        """Extract difficulty from HTML patterns"""
        pattern = r'<li[^>]*class="[^"]*p-2[^"]*"[^>]*>\s*<span[^>]*>\s*<img[^>]*difficulty-icon[^>]*>\s*<span[^>]*>\s*([^<]+)\s*</span>'
        match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_text(match.group(1))
        
        return ""
    
    def html_filename_to_url(self, filename: str) -> str:
        """Convert HTML filename back to URL"""
        url = filename.replace('.html', '')
        url = url.replace('_', '/')
        
        if not url.startswith('http'):
            url = 'https://' + url
        
        return url
    
    def get_html_file_path(self, url: str) -> Optional[str]:
        """Get the HTML file path for a given URL"""
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
            prep_time=self.extract_prep_time(json_data),
            servings=self.extract_servings(json_data),
            difficulty=self.extract_difficulty(html_content),
            chef=self.extract_author(json_data)
        )
        
        return metadata

    def save_to_jsonl(self, metadata: RecipeMetadata):
        """Save single recipe to the main recipes.jsonl file"""
        # Convert to dict
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
        
        # Append to the single JSONL file
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
        """Main method to scrape all recipes to a single recipes.jsonl file"""
        if not os.path.exists(self.html_dir):
            self.logger.error(f"HTML directory not found: {self.html_dir}")
            return
        
        all_html_files = [f for f in os.listdir(self.html_dir) if f.endswith('.html')]
        
        html_files = [f for f in all_html_files if self.is_valid_recipe_file(f)]
        
        if not html_files:
            self.logger.error(f"No valid recipe HTML files found in: {self.html_dir}")
            return
                
        # Clear the recipes file if it exists to start fresh
        if os.path.exists(self.recipes_file):
            os.remove(self.recipes_file)
        
        self.logger.info(f"Starting to scrape {len(html_files)} HTML files from {self.html_dir}")
        self.logger.info(f"Saving all recipes to: {self.recipes_file}")
        
        processed_count = 0
        successful_count = 0
        
        for html_filename in html_files:
            html_path = os.path.join(self.html_dir, html_filename)
            url = self.html_filename_to_url(html_filename)
            
            self.logger.info(f"Processing file {processed_count + 1}/{len(html_files)}: {html_filename}")
            
            metadata = self.extract_recipe_metadata(url=url, html_file=html_path)
            if metadata:
                self.save_to_jsonl(metadata)
                successful_count += 1
                            
            processed_count += 1
                    
        self.logger.info(f"Scraping completed. Processed {processed_count} HTML files, successfully extracted {successful_count} recipes")
        self.logger.info(f"All recipes saved to: {self.recipes_file}")

def main():
    scraper = RecipeScraper()
    scraper.scrape_all_recipes()

if __name__ == "__main__":
    main()