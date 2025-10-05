import unittest
import os
import sys
import json
from datetime import datetime

# Add parent directory and src directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from src.scraper import RecipeScraper  # noqa: E402


class TestScraperRegexExtractors(unittest.TestCase):
    """
    Unit tests for regex extractors in scraper.py
    Tests are performed on 20 real HTML files from data/raw_html directory
    """
    
    # Class variable to store test results
    test_results = {
        'timestamp': None,
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'test_details': []
    }
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - initialize scraper and select 20 test files"""
        cls.test_results['timestamp'] = datetime.now().isoformat()
        cls.scraper = RecipeScraper()
        
        # Get the path to raw_html directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.raw_html_dir = os.path.join(base_dir, 'data', 'raw_html')
        
        cls.test_files = [
            'foodnetwork.co.uk_recipes_simple-pancakes.html',
            'foodnetwork.co.uk_recipes_spaghetti-bolognese.html',
            'foodnetwork.co.uk_recipes_simple-fish-pie-peas.html',
            'foodnetwork.co.uk_recipes_shepherds-pie.html',
            'foodnetwork.co.uk_recipes_simple-coq-au-vin.html',
            'foodnetwork.co.uk_recipes_thai-green-chicken-curry-0.html',
            'foodnetwork.co.uk_recipes_tuna-pasta-salad.html',
            'foodnetwork.co.uk_recipes_vegan-pancakes.html',
            'foodnetwork.co.uk_recipes_slow-cooker-beef-stew.html',
            'foodnetwork.co.uk_recipes_vegetable-curry-0.html',
            'foodnetwork.co.uk_recipes_spanish-tortilla.html',
            'foodnetwork.co.uk_recipes_sticky-toffee-pudding-0-1.html',
            'foodnetwork.co.uk_recipes_smoked-salmon-and-egg-wraps.html',
            'foodnetwork.co.uk_recipes_strawberry-cheesecake.html',
            'foodnetwork.co.uk_recipes_tarte-tatin.html',
            'foodnetwork.co.uk_recipes_traditional-gaelic-irish-steak-with-irish-whiskey.html',
            'foodnetwork.co.uk_recipes_turkey-burgers.html',
            'foodnetwork.co.uk_recipes_vegan-banana-bread.html',
            'foodnetwork.co.uk_recipes_vanilla-cheesecake.html',
            'foodnetwork.co.uk_recipes_vegan-chocolate-cupcakes.html'
        ]
        
        # Load HTML content for all test files
        cls.html_contents = {}
        for filename in cls.test_files:
            filepath = os.path.join(cls.raw_html_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    cls.html_contents[filename] = f.read()
    
    def test_extract_json_ld_recipe(self):
        """Test extraction of JSON-LD recipe data from HTML"""
        print("\n=== Testing extract_json_ld_recipe ===")
        
        results = []
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            result = self.scraper.extract_json_ld_recipe(html_content)
            
            # Verify that we got a dictionary back
            self.assertIsInstance(result, dict, f"Failed for {filename}")
            
            # For recipe pages, we expect to find recipe data
            if 'recipes' in filename:
                # Check if we found recipe data
                has_recipe = bool(result)
                results.append({
                    'file': filename,
                    'url': self.scraper.html_filename_to_url(filename),
                    'success': has_recipe,
                    'has_data': has_recipe
                })
                print(f"{filename}: {'✓ Found recipe data' if has_recipe else '✗ No recipe data'}")
        
        self.__class__.test_results['test_details'].append({
            'test_name': 'extract_json_ld_recipe',
            'files_tested': len(results),
            'results': results
        })
    
    def test_clean_text(self):
        """Test the clean_text method removes HTML tags and normalizes whitespace"""
        print("\n=== Testing clean_text ===")
        
        test_cases = [
            ('<p>Hello <b>World</b></p>', 'Hello World'),
            ('Multiple    spaces   here', 'Multiple spaces here'),
            ('  Leading and trailing  ', 'Leading and trailing'),
            ('<script>alert("bad")</script>Good text', 'Good text'),
            ('Line\n\nbreaks\n\nhere', 'Line breaks here'),
            ('&lt;HTML&gt; entities &amp; stuff', '<HTML> entities & stuff'),
        ]
        
        for input_text, expected in test_cases:
            result = self.scraper.clean_text(input_text)
            self.assertEqual(result, expected, f"Failed for input: {input_text}")
            print(f"✓ '{input_text[:30]}...' -> '{result}'")
    
    def test_extract_title(self):
        """Test extraction of recipe title from JSON-LD data"""
        print("\n=== Testing extract_title ===")
        
        results = []
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            if json_data:
                title = self.scraper.extract_title(json_data)
                
                # Title should be a string
                self.assertIsInstance(title, str, f"Failed for {filename}")
                
                # For recipe pages, title should not be empty
                if 'recipes' in filename:
                    results.append({
                        'file': filename,
                        'url': self.scraper.html_filename_to_url(filename),
                        'success': bool(title),
                        'title': title[:50] if title else ''
                    })
                    print(f"{filename}: {'✓' if title else '✗'} Title: '{title[:50]}...'")
        
        self.__class__.test_results['test_details'].append({
            'test_name': 'extract_title',
            'files_tested': len(results),
            'results': results
        })
    
    def test_extract_description(self):
        """Test extraction of recipe description"""
        print("\n=== Testing extract_description ===")
        
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            description = self.scraper.extract_description(json_data, html_content)
            
            # Description should be a string
            self.assertIsInstance(description, str, f"Failed for {filename}")
            
            if description:
                # Description should not contain HTML tags
                self.assertNotIn('<', description, f"HTML tags found in {filename}")
                print(f"{filename}: ✓ Description: '{description[:60]}...'")
    
    def test_extract_ingredients(self):
        """Test extraction of ingredients list from JSON-LD data"""
        print("\n=== Testing extract_ingredients ===")
        
        results = []
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            ingredients = self.scraper.extract_ingredients(json_data)
            
            # Should return a list
            self.assertIsInstance(ingredients, list, f"Failed for {filename}")
            
            # Each ingredient should be a string
            for ing in ingredients:
                self.assertIsInstance(ing, str, f"Non-string ingredient in {filename}")
                # Ingredients should not end with colon (filtered out)
                self.assertFalse(ing.endswith(':'), f"Ingredient ends with colon in {filename}")
            
            if ingredients:
                results.append({
                    'file': filename,
                    'url': self.scraper.html_filename_to_url(filename),
                    'success': True,
                    'ingredient_count': len(ingredients)
                })
                print(f"{filename}: ✓ Found {len(ingredients)} ingredients")
        
        self.__class__.test_results['test_details'].append({
            'test_name': 'extract_ingredients',
            'files_tested': len(results),
            'results': results
        })
    
    def test_extract_method(self):
        """Test extraction of cooking method/instructions"""
        print("\n=== Testing extract_method ===")
        
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            method = self.scraper.extract_method(html_content, json_data)
            
            # Method should be a string
            self.assertIsInstance(method, str, f"Failed for {filename}")
            
            if method:
                # Method should not contain HTML tags
                self.assertNotIn('<', method, f"HTML tags found in {filename}")
                print(f"{filename}: ✓ Method length: {len(method)} chars")
    
    def test_extract_author(self):
        """Test extraction of recipe author"""
        print("\n=== Testing extract_author ===")
        
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            author = self.scraper.extract_author(json_data)
            
            # Author should be a string
            self.assertIsInstance(author, str, f"Failed for {filename}")
            
            if author:
                # Author should not contain HTML tags
                self.assertNotIn('<', author, f"HTML tags found in {filename}")
                print(f"{filename}: ✓ Author: '{author}'")
    
    def test_format_duration(self):
        """Test ISO 8601 duration format conversion"""
        print("\n=== Testing format_duration ===")
        
        test_cases = [
            ('PT30M', '30 min'),
            ('PT1H', '1 hr'),
            ('PT1H30M', '1 hr 30 min'),
            ('PT2H', '2 hrs'),
            ('PT45M', '45 min'),
            ('PT2H15M', '2 hrs 15 min'),
            ('', ''),
        ]
        
        for input_duration, expected in test_cases:
            result = self.scraper.format_duration(input_duration)
            self.assertEqual(result, expected, f"Failed for {input_duration}")
            print(f"✓ '{input_duration}' -> '{result}'")
    
    def test_extract_prep_time(self):
        """Test extraction and formatting of preparation time"""
        print("\n=== Testing extract_prep_time ===")
        
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            prep_time = self.scraper.extract_prep_time(json_data)
            
            # Prep time should be a string
            self.assertIsInstance(prep_time, str, f"Failed for {filename}")
            
            if prep_time:
                # Should not contain "PT" if properly formatted
                self.assertNotIn('PT', prep_time, f"Duration not formatted in {filename}")
                print(f"{filename}: ✓ Prep time: '{prep_time}'")
    
    def test_extract_servings(self):
        """Test extraction of servings information"""
        print("\n=== Testing extract_servings ===")
        
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            json_data = self.scraper.extract_json_ld_recipe(html_content)
            
            servings = self.scraper.extract_servings(json_data)
            
            # Servings should be a string
            self.assertIsInstance(servings, str, f"Failed for {filename}")
            
            if servings:
                print(f"{filename}: ✓ Servings: '{servings}'")
    
    def test_extract_difficulty(self):
        """Test extraction of difficulty level using regex pattern"""
        print("\n=== Testing extract_difficulty ===")
        
        for filename in self.test_files:
            if filename not in self.html_contents:
                continue
            
            html_content = self.html_contents[filename]
            difficulty = self.scraper.extract_difficulty(html_content)
            
            # Difficulty should be a string
            self.assertIsInstance(difficulty, str, f"Failed for {filename}")
            
            if difficulty:
                # Difficulty extracted successfully
                print(f"{filename}: ✓ Difficulty: '{difficulty}'")
    
    def test_html_filename_to_url(self):
        """Test conversion of HTML filename to URL"""
        print("\n=== Testing html_filename_to_url ===")
        
        test_cases = [
            ('foodnetwork.co.uk_recipes_simple-pancakes.html', 
             'https://foodnetwork.co.uk/recipes/simple-pancakes'),
            ('foodnetwork.co.uk_recipes_spaghetti-bolognese.html',
             'https://foodnetwork.co.uk/recipes/spaghetti-bolognese'),
        ]
        
        for filename, expected_url in test_cases:
            result = self.scraper.html_filename_to_url(filename)
            self.assertEqual(result, expected_url, f"Failed for {filename}")
            print(f"✓ '{filename}' -> '{result}'")
    
    def test_is_valid_recipe_file(self):
        """Test validation of recipe filenames"""
        print("\n=== Testing is_valid_recipe_file ===")
        
        valid_files = [
            'foodnetwork.co.uk_recipes_simple-pancakes.html',
            'foodnetwork.co.uk_recipes_spaghetti-bolognese.html',
        ]
        
        invalid_files = [
            'foodnetwork.co.uk_about.html',
            'foodnetwork.co.uk_article_some-article.html',
            'foodnetwork.co.uk_shows.html',
        ]
        
        for filename in valid_files:
            result = self.scraper.is_valid_recipe_file(filename)
            self.assertTrue(result, f"Should be valid: {filename}")
            print(f"✓ Valid: {filename}")
        
        for filename in invalid_files:
            result = self.scraper.is_valid_recipe_file(filename)
            self.assertFalse(result, f"Should be invalid: {filename}")
            print(f"✓ Invalid: {filename}")
    
    def test_extract_recipe_metadata_full_integration(self):
        """Integration test: Extract complete metadata from all 20 HTML files"""
        print("\n=== Testing extract_recipe_metadata (Full Integration) ===")
        
        successful = 0
        failed = 0
        detailed_results = []
        
        for filename in self.test_files:
            filepath = os.path.join(self.raw_html_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"⚠ File not found: {filename}")
                failed += 1
                detailed_results.append({
                    'file': filename,
                    'url': self.scraper.html_filename_to_url(filename),
                    'success': False,
                    'error': 'File not found'
                })
                continue
            
            # Extract metadata
            url = self.scraper.html_filename_to_url(filename)
            metadata = self.scraper.extract_recipe_metadata(url=url, html_file=filepath)
            
            if metadata:
                # Verify all fields are present
                self.assertIsNotNone(metadata.url, f"URL missing for {filename}")
                self.assertIsNotNone(metadata.html_file, f"HTML file missing for {filename}")
                self.assertIsNotNone(metadata.title, f"Title missing for {filename}")
                self.assertIsNotNone(metadata.description, f"Description is None for {filename}")
                self.assertIsNotNone(metadata.ingredients, f"Ingredients is None for {filename}")
                self.assertIsNotNone(metadata.method, f"Method is None for {filename}")
                
                # Check that at least title and ingredients exist (core recipe data)
                if metadata.title and len(metadata.ingredients) > 0:
                    print(f"✓ {filename}:")
                    print(f"  - Title: {metadata.title[:50]}...")
                    print(f"  - Ingredients: {len(metadata.ingredients)} items")
                    print(f"  - Method: {len(metadata.method)} chars")
                    print(f"  - Prep time: {metadata.prep_time}")
                    print(f"  - Servings: {metadata.servings}")
                    print(f"  - Difficulty: {metadata.difficulty}")
                    print(f"  - Chef: {metadata.chef}")
                    successful += 1
                    detailed_results.append({
                        'file': filename,
                        'url': url,
                        'success': True,
                        'title': metadata.title,
                        'ingredient_count': len(metadata.ingredients),
                        'method_length': len(metadata.method),
                        'prep_time': metadata.prep_time,
                        'servings': metadata.servings,
                        'difficulty': metadata.difficulty,
                        'chef': metadata.chef
                    })
                else:
                    print(f"⚠ Incomplete data for {filename}")
                    failed += 1
                    detailed_results.append({
                        'file': filename,
                        'url': url,
                        'success': False,
                        'error': 'Incomplete data'
                    })
            else:
                print(f"✗ Failed to extract metadata from {filename}")
                failed += 1
                detailed_results.append({
                    'file': filename,
                    'url': self.scraper.html_filename_to_url(filename),
                    'success': False,
                    'error': 'Extraction failed'
                })
        
        print("\n=== Summary ===")
        print(f"Successful: {successful}/{len(self.test_files)}")
        print(f"Failed: {failed}/{len(self.test_files)}")
        
        # Store summary
        self.__class__.test_results['test_details'].append({
            'test_name': 'extract_recipe_metadata_full_integration',
            'successful': successful,
            'failed': failed,
            'total_files': len(self.test_files),
            'results': detailed_results
        })
        
        # Update overall statistics
        self.__class__.test_results['total_tests'] = len(self.test_files)
        self.__class__.test_results['passed'] = successful
        self.__class__.test_results['failed'] = failed
        
        # At least 50% should be successful
        self.assertGreaterEqual(successful, len(self.test_files) * 0.5, 
                               "Less than 50% of files were successfully processed")
    
    @classmethod
    def tearDownClass(cls):
        """Save test results to JSON file after all tests complete"""
        # Update timestamp
        cls.test_results['timestamp'] = datetime.now().isoformat()
        
        # Save to results.json
        results_file = os.path.join(os.path.dirname(__file__), 'results.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(cls.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Test results saved to: {results_file}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
