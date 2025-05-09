from together import Together
from sentence_transformers import SentenceTransformer
import base64
import re
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET
import copy
import xml.sax.saxutils as su

# Load environment variables from .env file
load_dotenv()

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response(messages, max_tokens=1000, temperature=0.7):
    # Access the API key from environment variables
    api_key = os.getenv("TOGETHER_API_KEY")         
    client = Together(api_key=api_key)


    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content

def parse_recipe_xml(xml_string):
    """
    Parse a recipe XML string based on the defined structure in the prompt.
    
    The expected structure is:
    <recipe>
      <title>The dish name</title>
      <ingredients>
        <ingredient>1 cup whole milk</ingredient>
        <ingredient>2 tbsp sugar</ingredient>
        <!-- More ingredients -->
      </ingredients>
      <instructions>
        <step>1. Preheat the oven to 350°F.</step>
        <step>2. Mix all ingredients in a bowl.</step>
        <!-- More steps -->
      </instructions>
    </recipe>
    """
    try:
        # # 1) Extract <think> if present
        # think_text = None
        # think_match = re.search(r'<think>(.*?)</think>', xml_string, re.DOTALL)
        # if think_match:
        #     think_text = think_match.group(1).strip()


        xml_string = xml_string.replace('&', 'and')
            
        # Extract the XML part if there's text before or after it
        xml_match = re.search(r'<recipe>.*?</recipe>', xml_string, re.DOTALL)
        if xml_match:
            xml_string = xml_match.group(0)
        
        # Try to parse the XML
        root = ET.fromstring(xml_string)
        
        # Extract the title
        title = root.find('title').text if root.find('title') is not None else "Unknown"
        
        ingredients = []
        ingredients_section = root.find('ingredients')
        if ingredients_section is not None:
            for ing_elem in ingredients_section.findall('ingredient'):
                if ing_elem.text:
                    ingredients.append(ing_elem.text.strip())
        
        # Extract instruction steps
        steps = []
        instructions_section = root.find('instructions')
        if instructions_section is not None:
            for step_elem in instructions_section.findall('step'):
                if step_elem.text:
                    steps.append(step_elem.text.strip())
        
        # Return structured data in the format matching our dataset
        if(xml_string.find("<think>") != -1):
            print('think found')
            return {
                'title': title,
                'ingredients': ingredients,
                'steps': steps,
                'think' : "include"
            }
        else:
            return {
                'title': title,
                'ingredients': ingredients,
                'steps': steps,
                'think' : "None"
            }
    except Exception as e:
        print(f"Error parsing XML: {e}")
        print(f"Problematic XML: {xml_string}")
        return None
    
# Function to display recipe in a nicely formatted way
def display_recipe(recipe_dict):
    if not recipe_dict:
        print("Could not parse recipe.")
        return
    
    print(f"🍽️ DISH: {recipe_dict['title']}")
    
    print("\n📋 INGREDIENTS:")
    for i, ing in enumerate(recipe_dict['ingredients'], 1):
        qty = ing['quantity']
        unit = ing['unit']
        name = ing['name']
        
        if qty and unit:
            print(f"  {i}. {qty} {unit} {name}")
        elif qty:
            print(f"  {i}. {qty} {name}")
        else:
            print(f"  {i}. {name}")
    
    print("\n👨‍🍳 INSTRUCTIONS:")
    for i, step in enumerate(recipe_dict['steps'], 1):
        print(f"  {i}. {step}")

def prase_ingridients_to_embeddings(ingredients_list):
    """
    Vectorize ingredients. To caclucalte the cosine similarity later.
    """
    if not ingredients_list:
        return []
    
    embeddings = []
    for ingredient in ingredients_list:
        embedding = embedder.encode(ingredient)
        embeddings.append(embedding)
    return embeddings
    
def parse_instructions_to_embeddings(instructions_list):
    """
    Vectorize instructions. To caclucalte the cosine similarity later.
    """
    if not instructions_list:
        return []
    
    embeddings = []
    for instruction in instructions_list:
        embedding = embedder.encode(instruction)
        embeddings.append(embedding)
    return embeddings

def parse_instructions(instructions_text):
    """
    Parse instructions text into a list of individual instruction steps.
    Each step is separated by a newline character.
    """
    if not instructions_text:
        return []
    
    # Split by newline
    steps = instructions_text.split('\n')
    
    # Remove any empty steps
    steps = [step.strip() for step in steps if step.strip()]
    
    return steps

def parse_ingredients(ingredients_text):
    """
    Parse ingredients text which is a string representation of a list
    using regular expressions to extract each ingredient
    """
    if not ingredients_text:
        return []
        
    # Check if it's already a list
    if isinstance(ingredients_text, list):
        return ingredients_text
        
    # Use regex to match everything between single quotes
    pattern = r"'([^'\\]*(?:\\.[^'\\]*)*)'"
    matches = re.findall(pattern, ingredients_text)
    
    return matches

def encode_image(image_path):
    """Encode an image as base64 string, with error handling."""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None
        
        # Open and encode the image
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None
    
def preprocess_dataset(hf_dataset):
    """
    Preprocess the Hugging Face dataset by adding new columns for:
    - Parsed ingredients
    - Parsed cleaned ingredients (using parse_ingredients for now)
    - Parsed instruction steps
    - Base64 encoded images
    
    Filters out examples where image encoding fails.
    
    Args:
        hf_dataset: The original Hugging Face dataset object.
        
    Returns:
        The processed Hugging Face dataset with only valid images.
    """

    def _process_example(example):
        """Helper function to process a single dataset example."""
        # Process ingredients
        if 'Ingredients' in example:
            example['parsed_ingredients'] = parse_ingredients(example['Ingredients'])
        else:
             example['parsed_ingredients'] = [] # Add empty list if key missing
        
        # Process cleaned ingredients
        if 'Cleaned_Ingredients' in example:
            example['parsed_cleaned_ingredients'] = parse_ingredients(example['Cleaned_Ingredients'])
        else:
             example['parsed_cleaned_ingredients'] = [] # Add empty list if key missing
        
        # Process instructions
        if 'Instructions' in example:
            example['instruction_steps'] = parse_instructions(example['Instructions'])
        else:
            example['instruction_steps'] = [] # Add empty list if key missing
        
        # Validate and process image paths
        if 'full_image_path' in example:
            example['base64_image'] = encode_image(example['full_image_path'])
        else:
            example['base64_image'] = None # Add None if key missing
        
        # Vectorize ingredients and instructions
        example['ingredients_embeddings'] = prase_ingridients_to_embeddings(example['parsed_ingredients'])
        example['instructions_embeddings'] = parse_instructions_to_embeddings(example['instruction_steps'])
        
        return example

    print(f"Preprocessing dataset with {len(hf_dataset)} examples...")
    
    # First, process all examples
    processed_dataset = hf_dataset.map(_process_example)
    
    # Then filter out examples with missing images
    valid_examples = processed_dataset.filter(lambda example: example['base64_image'] is not None)
    
    print(f"Preprocessing complete. {len(valid_examples)} examples with valid images (filtered out {len(processed_dataset) - len(valid_examples)} examples)")
    return valid_examples

def convert_recipe_to_xml(recipe_entry):
    """
    Convert a recipe dataset entry to XML format that matches LLM output.
    
    Args:
        recipe_entry: A dictionary-like object containing recipe data
        
    Returns:
        str: XML formatted recipe
    """
    # Extract the required fields
    title = recipe_entry["Title"]
    ingredients = recipe_entry["parsed_cleaned_ingredients"]
    steps = recipe_entry["instruction_steps"]

    print("converting to xml ...example:", recipe_entry)
    
    # Start building the XML structure
    xml = [
        "<recipe>",
        f"  <title>{title}</title>",
        "  <ingredients>"
    ]
    
    # Add all ingredients
    for ingredient in ingredients:
        xml.append(f"    <ingredient>{ingredient}</ingredient>")
    
    xml.append("  </ingredients>")
    xml.append("  <instructions>")
    
    # Add all instruction steps
    for i, step in enumerate(steps, 1):
        # Clean any newlines or extra commas in the step text
        clean_step = step.replace("\n", " ").strip()
        xml.append(f"    <step>{i}. {clean_step}</step>")
    
    xml.append("  </instructions>")
    xml.append("</recipe>")

    xml_string = "\n".join(xml)
    
    return {"xml_recipe": xml_string}
