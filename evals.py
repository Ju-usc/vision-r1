from ast import main
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from importlib.resources import contents
import json
from logging import config
from pyexpat import model
from PIL import Image
from utils import parse_recipe_xml, generate_response, convert_recipe_to_xml
from google import genai
from google.genai import types
from pydantic import BaseModel


embedder = SentenceTransformer('all-MiniLM-L6-v2')

def compute_top_cosine_similarity(pred_string, reference_strings, reference_embeddings):
    """
    Compute the highest cosine similarity between an input string embedding and a list of string embeddings.
    ex) input_string = "Add 1 cup of rice"
        list_of_strings = ["Add 1 cup of rice", "Add 1 cup of pasta", "Add 1 cup of chicken"]
        output = 0.95
        best_match = "Add 1 cup of rice"
        best_match_index = 0

        reference_embeddings: list of vectorized ingredients and instructions
        reference_strings: list of the full ingredient+instruction strings from csv file
        pred_string: 

    Args:
        string: the string that we want to find the best match for (string)
        list_of_strings: List of strings (list of strings)
    
    Returns:
        A tuple containing:
        - best_score: The highest cosine similarity score
        - best_idx: The index of the best matching string
        - best_string: The text of the best matching string
    """
    # Ensure the step_embedding is 2D (batch size 1)


    pred_sentence_embedding = embedder.encode([pred_string])

    
    if len(pred_sentence_embedding.shape) == 1:
        pred_sentence_embedding = pred_sentence_embedding.reshape(1, -1)
    
    best_score = -1  # Initialize with impossible score
    best_idx = -1
    
    # Compute similarity with each golden step
    for i, reference_embedding in enumerate(reference_embeddings):
        # Convert list to numpy array
        reference_embedding_np = np.array(reference_embedding)
        
        # Ensure golden embedding is 2D
        if len(reference_embedding_np.shape) == 1:
            reference_embedding_np = reference_embedding_np.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(pred_sentence_embedding, reference_embedding_np)[0][0]
        
        # Update if this is the best score so far
        if similarity > best_score:
            best_score = similarity
            best_idx = i
    
    return best_score, best_idx, reference_strings[best_idx]


def compute_bleu_score(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)


def compute_rouge_scores(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)

def compute_best_item_bleu(pred_item, reference_items):
    best_score = 0
    for ref_item in reference_items:
        score = compute_bleu_score(ref_item, pred_item) 
        if score > best_score:
            best_score = score
    return best_score

def compute_ingredient_bleu_score(pred_ingredients, golden_ingredients):
    scores = []
    for pred in pred_ingredients:
        best = compute_best_item_bleu(pred, golden_ingredients)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0


def compute_best_item_rouge(pred_item, reference_items):
    """
    For a single predicted ingredient, compute ROUGE scores against each reference and return the best match.
    Here we take the best average of ROUGE-1 and ROUGE-L f-measures.
    """
    best_avg = 0
    best_rouge1 = 0
    best_rougeL = 0
    for ref_item in reference_items:
        scores = compute_rouge_scores(ref_item, pred_item)
        avg_score = (scores['rouge1'].fmeasure + scores['rougeL'].fmeasure) / 2
        if avg_score > best_avg:
            best_avg = avg_score
            best_rouge1 = scores['rouge1'].fmeasure
            best_rougeL = scores['rougeL'].fmeasure
    return best_rouge1, best_rougeL

def compute_ingredient_rouge_score(pred_ingredients, reference_ingredients):
    """
    Compute the average best-match ROUGE scores (both rouge1 and rougeL) for all predicted ingredients.
    """
    rouge1_scores = []
    rougeL_scores = []
    for pred in pred_ingredients:
        r1, rL = compute_best_item_rouge(pred, reference_ingredients)
        rouge1_scores.append(r1)
        rougeL_scores.append(rL)
    avg_r1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    return avg_r1, avg_rL

# Example recipe entry
"""
3116,Our Favorite Chocolate Chip Cookies,"['1 1/2 cups all-purpose flour', '1 teaspoon baking powder', '1/2 teaspoon kosher salt', '1/4 teaspoon baking soda', '2 large eggs', '1 teaspoon vanilla extract', '10 tablespoons unsalted butter, room temperature', '3/4 cup granulated sugar', '3/4 cup (packed) light brown sugar', '8 ounces chopped high-quality semisweet chocolate or chocolate chips (about 1 1/2 cups)']","Place racks in upper and lower thirds of oven; preheat to 375°F. Line 2 rimmed baking sheets with parchment paper.
Whisk flour, baking powder, salt, and baking soda in a medium bowl. Lightly beat eggs and vanilla in a small bowl.
Using an electric mixer on medium speed, beat butter, granulated sugar, and brown sugar in a large bowl until light and fluffy, 3–4 minutes. Add egg mixture and beat, occasionally scraping down sides of bowl, until mixture is pale and fluffy, 3–4 minutes. Reduce mixer speed to low and gradually add dry ingredients, beating until just incorporated. Fold in chocolate with a spatula.
Spoon about 12 heaping tablespoonfuls of dough onto each prepared sheet, spacing 1"" apart. Chill dough 15 minutes on baking sheets before transferring to oven to prevent dough from spreading too much as it bakes.
Bake cookies, rotating sheets halfway through, until just golden brown around the edges, 10–12 minutes (cookies will firm up as they cool). Let cookies cool on baking sheets 3 minutes, then transfer to wire racks and let cool completely. Let baking sheet cool completely before lining with fresh parchment and spooning on dough for the third batch of cookies.
Dough can be made 3 months ahead. Wrap tightly and freeze. Cookies can be made 1 day ahead. Store in an airtight container at room temperature, or freeze up to 3 months.",our-favorite-chocolate-chip-cookies,"['1 1/2 cups all-purpose flour', '1 teaspoon baking powder', '1/2 teaspoon kosher salt', '1/4 teaspoon baking soda', '2 large eggs', '1 teaspoon vanilla extract', '10 tablespoons unsalted butter, room temperature', '3/4 cup granulated sugar', '3/4 cup (packed) light brown sugar', '8 ounces chopped high-quality semisweet chocolate or chocolate chips (about 1 1/2 cups)']"
"""

# Example recipe entry in xml
input_xml = "<recipe> <title>Our Favorite Chocolate Chip Cookies</title> <ingredients> <ingredient>1 1/2 cups all-purpose flour</ingredient> <ingredient>1 teaspoon baking powder</ingredient> <ingredient>1/2 teaspoon kosher salt</ingredient> <ingredient>1/4 teaspoon baking soda</ingredient> <ingredient>2 large eggs</ingredient> <ingredient>1 teaspoon vanilla extract</ingredient> <ingredient>10 tablespoons unsalted butter, room temperature</ingredient> <ingredient>3/4 cup granulated sugar</ingredient> <ingredient>3/4 cup packed light brown sugar</ingredient> <ingredient>8 ounces chopped high-quality semisweet chocolate or chocolate chips</ingredient> </ingredients> <instructions> <step>1. Place racks in the upper and lower thirds of the oven and preheat to 375°F; line two rimmed baking sheets with parchment paper.</step> <step>2. In a medium bowl, whisk together flour, baking powder, salt, and baking soda; in a small bowl, lightly beat eggs with vanilla extract.</step> <step>3. Using an electric mixer on medium speed, beat butter, granulated sugar, and brown sugar until light and fluffy, about 3–4 minutes.</step> <step>4. Add the egg and vanilla mixture and beat until pale and fluffy, about 3–4 minutes, scraping down the bowl as needed.</step> <step>5. Reduce mixer speed to low and gradually add the dry ingredients, mixing until just incorporated; fold in the chopped chocolate with a spatula.</step> <step>6. Scoop about 12 heaping tablespoonfuls of dough onto each prepared sheet, spacing them 1 inch apart; chill on baking sheets for 15 minutes.</step> <step>7. Bake at 375°F, rotating sheets halfway through, until edges are just golden, 10–12 minutes.</step> <step>8. Let cookies cool on the baking sheets for 3 minutes, then transfer to wire racks to cool completely; cool sheets before lining for additional batches.</step> <step>9. Make-ahead and storage: dough can be frozen up to 3 months; baked cookies can be stored in an airtight container at room temperature for up to 1 day or frozen for up to 3 months.</step> </instructions> </recipe>"

# Example recipe xml model output

output_xml = "<recipe> <title>Favorite Chocolate Chip Cookies</title> <ingredients> <ingredient>2 1/4 cups all-purpose flour</ingredient> <ingredient>1 teaspoon baking soda</ingredient> <ingredient>1 teaspoon salt</ingredient> <ingredient>1 cup unsalted butter, softened</ingredient> <ingredient>3/4 cup granulated sugar</ingredient> <ingredient>3/4 cup packed brown sugar</ingredient> <ingredient>1 teaspoon vanilla extract</ingredient> <ingredient>2 large eggs</ingredient> <ingredient>2 cups semisweet chocolate chips</ingredient> </ingredients> <instructions> <step>1. Preheat oven to 350°F (175°C) and line a baking sheet with parchment paper.</step> <step>2. In a medium bowl, whisk together flour, baking soda, and salt. In a large bowl, cream butter and sugars until light and fluffy. Beat in eggs one at a time, then stir in vanilla. Gradually blend in dry ingredients until just combined.</step> <step>3. Fold in chocolate chips. Drop rounded tablespoons of dough onto prepared baking sheet, spacing 2 inches apart. Bake 10–12 minutes until edges are golden. Cool on sheet for 5 minutes before transferring to wire rack.</step> </instructions> </recipe>"

    
client = genai.Client(api_key="AIzaSyD1-Ms788wc8CEu2CqwKjH3x7m95txv7w8")
ex_cookies = "/Users/BenChung/Downloads/FavoriteChocolateChipCookies.jpg"

def llm_eval_score(golden_xml: str, pred_xml: str, image_base64: str = None) -> str:
    """
    Uses an LLM to act as a professional chef, judging whether the predicted recipe
    would feasibly produce the dish shown in the image and align with the ground-truth.
    Returns an XML string of the form:
    <evaluation>
      <feasibility_score>0.85</feasibility_score>
      <comment>Brief strengths/issues</comment>
    </evaluation>
    """
    # Parse both recipes into structured text
    gt = parse_recipe_xml(golden_xml)
    pred = parse_recipe_xml(pred_xml)

    # Build the system prompt: chef-style holistic evaluation, XML output
    # SYSTEM MESSAGE PROMPT - FINE TUNE
    system_message = (
        "You are a professional chef.\n"
        "Evaluate whether the predicted recipe would realistically produce the dish shown in the image, "
        "and how well it aligns with the ground-truth recipe. Consider ingredient correctness, cooking steps, "
        "order, practicality, and visual appearance.\n"
        "Determine a single feasibility_score of the predicted recipe between 0.0 (completely infeasible) and 1.0 (perfectly feasible) based on the aforementioned criteria,"
        "and a comment describing strengths and potential issues, with insight into the chosen feasability score.\n"
        "Return a parseable JSON string/snippet with the entry: evaluation = { \"feasability_score\": string, \"comment\": string }"
    )

    # class Evaluation(BaseModel):
    #     feasability_score: str
    #     comment: str


    # Build the user prompt including XMLs and optional image
    user_parts = {'ground-truth': gt,
                  'predicted': pred}
    gr = "<ground-truth>" + golden_xml + "</ground-truth>"
    pr = "<predicted>" + pred_xml + "</predicted>"

    image = Image.open(image_base64)
    llm_eval_response = client.models.generate_content( # can change add more config like temperature etc
        model="gemini-2.5-pro-exp-03-25",
        config=types.GenerateContentConfig(
            system_instruction=system_message
        ),
        contents=[image, gr, pr]
    )
    # if image_base64:
    #     user_parts += ["<image_base64>", image_base64, "</image_base64>"]
    # user_message = "\n".join(user_parts)

    # # Query the LLM with deterministic output
    # llm_output = generate_response(
    #     messages=[
    #         {"role": "system", "content": system_message},
    #         {"role": "user",   "content": user_message}
    #     ],
    #     max_tokens=256,
    #     temperature=0.0
    # )

    # Return the raw XML from the LLM (trim whitespace)
    # print(llm_eval_response.parsed)
    return llm_eval_response.text

# response = llm_eval(input_xml, output_xml, ex_cookies)

# print(response.feasability_score)
# print(response.comment)
# print(response)

# # If run as a script, allow quick CLI testing
# if __name__ == "__main__":
#     import sys
#     # if len(sys.argv) < 3:
#     #     print("Usage: python llm_evaluator.py ground_truth.xml predicted.xml [image_base64.txt]")
#     #     sys.exit(1)
#     print(parse_recipe_xml(sys.argv[1]))
#     # gt_path = sys.argv[1]
#     # pred_path = sys.argv[2]
#     # image_b64 = None
#     # if len(sys.argv) == 4:
#     #     image_b64 = open(sys.argv[3]).read()

#     # gt_xml = open(gt_path).read()
#     # pred_xml = open(pred_path).read()

#     # xml_eval = score_recipe_llm(gt_xml, pred_xml, image_base64=image_b64)
#     # print(xml_eval)


def compute_evals(pred_recipe, golden_recipe, image_path):
    """
    Compute evaluation metrics for the predicted recipe against the golden recipe.
    - For instructions (steps), we compare the entire concatenated string.
    - For ingredients (unordered), we use a per-item best-match approach.
    
    Metrics computed:
      * Cosine Similarity (per-item best match)
      * BLEU Score (steps: whole string; ingredients: per-item average)
      * ROUGE Scores (steps: per-item; ingredients: per-item best match averaged)
    """
    # Assume these fields are lists of strings.
    pred_steps_list = pred_recipe['steps']
    golden_steps_list = golden_recipe['instruction_steps']
    pred_ingredients_list = pred_recipe['ingredients']
    golden_ingredients_list = golden_recipe['parsed_ingredients']

    # Concatenate entire instructions for full-string comparisons.
    pred_steps_string = " ".join(pred_steps_list)
    golden_steps_string = " ".join(golden_steps_list)
    # For ingredients, we'll use the list directly for per-item matching.

    golden_steps_embeddings = golden_recipe['instructions_embeddings']
    golden_ingredients_embeddings = golden_recipe['ingredients_embeddings']

    print("Calculating cosine similarity...")
    # --- Cosine Similarity ---
    cosine_scores = {"steps": [], "ingredients": []}
    for step in pred_steps_list:
        score, _, _ = compute_top_cosine_similarity(step, golden_steps_list, golden_steps_embeddings)
        cosine_scores["steps"].append(score)
    for ingredient in pred_ingredients_list:
        score, _, _ = compute_top_cosine_similarity(ingredient, golden_ingredients_list, golden_ingredients_embeddings)
        cosine_scores["ingredients"].append(score)

    print("Calculating BLEU scores...")
    # --- BLEU Scores ---
    bleu_scores = {"steps": None, "ingredients": None}
    # For steps, use the whole concatenated string.
    step_bleu_list = []
    for step in pred_steps_list:
        score = compute_bleu_score(golden_steps_string, step)
        step_bleu_list.append(score)
    bleu_steps = sum(step_bleu_list) / len(step_bleu_list) if step_bleu_list else 0
    # For ingredients, use per-item best-match BLEU.
    bleu_ingredients = compute_ingredient_bleu_score(pred_ingredients_list, golden_ingredients_list)
    bleu_scores["steps"] = bleu_steps
    bleu_scores["ingredients"] = bleu_ingredients

    print("Calculating ROUGE scores...")
    # --- ROUGE Scores ---
    rouge_scores = {"steps": {"rouge1": [], "rougeL": []}, "ingredients": {"rouge1": None, "rougeL": None}}
    # For steps, compute ROUGE per step against the whole instructions.
    for step in pred_steps_list:
        scores = compute_rouge_scores(golden_steps_string, step)
        rouge_scores["steps"]["rouge1"].append(scores['rouge1'].fmeasure)
        rouge_scores["steps"]["rougeL"].append(scores['rougeL'].fmeasure)
    rouge_steps_r1 = sum(rouge_scores["steps"]["rouge1"]) / len(rouge_scores["steps"]["rouge1"]) if rouge_scores["steps"]["rouge1"] else 0
    rouge_steps_rL = sum(rouge_scores["steps"]["rougeL"]) / len(rouge_scores["steps"]["rougeL"]) if rouge_scores["steps"]["rougeL"] else 0

    # For ingredients, use per-item best-match ROUGE.
    rouge_ing_r1, rouge_ing_rL = compute_ingredient_rouge_score(pred_ingredients_list, golden_ingredients_list)

    # --- Aggregate and Return ---
    def average(lst):
        return sum(lst) / len(lst) if lst else 0
    

    # Asking gemini-2.5-pro-exp-03-25 to judge/evaluate the predicted recipe and its feasability

    print("Calculating LLM evaluation...")
    llm_evaluation = llm_eval_score(convert_recipe_to_xml(golden_recipe), convert_recipe_to_xml(pred_recipe), image_path)
    json_string = llm_evaluation[8: -4]
    data = json.loads(json_string)
    eval = data.get("evaluation", {})
    feas = eval.get("feasability_score")
    comm = eval.get("comment")

    aggregated = {
        'cosine_similarity': {
            'steps': average(cosine_scores["steps"]),
            'ingredients': average(cosine_scores["ingredients"])
        },
        # 'bleu_score': bleu_scores,
        'rouge_scores': {
            'steps': {
                'rouge1': rouge_steps_r1,
                'rougeL': rouge_steps_rL,
            },
            'ingredients': {
                'rouge1': rouge_ing_r1,
                'rougeL': rouge_ing_rL,
            }
        },
        'llm_evaluation':{
            'feasability_score': feas,
            'comment': comm,
        }
    }
    print("Evaluation metrics calculated.")
    return aggregated
