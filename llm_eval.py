from importlib.resources import contents
import json
from logging import config
from pyexpat import model
from PIL import Image
from utils import parse_recipe_xml, convert_recipe_to_xml, preprocess_dataset
from google import genai
from google.genai import types
from pydantic import BaseModel
from pathlib import Path
# may need more libraries

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
        "order, practicality, and visual appearance. "
        #"There will also be a <thinking>...</thinking> section in the prediction XML input, so also comment on whether the model's \"thinking\" is valid.\n"
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

"""
Usage: llm_eval_score(input1, input2, image):
input1: xml formatted recipe from csv database entry
input2: xml formatted recipe from gpt(emulates the model output)
image: image path string
"""