import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import patch_qwen  # noqa: F401
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoProcessor,
)
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
from qwen_vl_utils import process_vision_info
from trl import GRPOConfig, GRPOTrainer
from trl.models.utils import unwrap_model_for_generation
from typing import Any
from io import BytesIO
from PIL import Image

from load_helper import get_dev_stage_datasets

SYSTEM_PROMPT = """Look at the food image and create a recipe in following XML format:

<think>think step by step to infer the recipe from the image</think>
<recipe>
  <title>Name of the dish</title>
  <ingredients>
    <ingredient>Quantity and ingredient (e.g., 2 tbsp olive oil)</ingredient>
    <!-- Add more ingredients as needed -->
  </ingredients>
  <instructions>
    <step>1. Preheat the oven to 350°F.</step>
    <!-- Add more steps as needed -->
  </instructions>
</recipe>

Format requirements:
- Follow the exact XML structure shown above
- List each ingredient in a separate <ingredient> tag with quantity
- Present each cooking step in a separate <step> tag
- Use clear, concise language throughout
- Do not generate any text outside of the <recipe> tags and <think> tags
"""

# model_name = "/home/jackson/vision-r1/outputs/Qwen-0.5B-GRPO-Count-SFT/checkpoint-1500"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

LOG_FILE = "response_log.txt"
output_dir = "outputs/Qwen-3B-GRPO-Count-R1"
run_name = "Qwen-3B-GRPO-Count-R1"
max_pixels = 256 * 256
processor = AutoProcessor.from_pretrained(
    model_name, max_pixels=max_pixels, use_cache=False
)
# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # Use float32 for CPU
    device_map="auto" if device == "cuda" else None,  # Don't use device_map for CPU
    # use_flash_attention_2=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=False)

model.gradient_checkpointing_enable() # Enable gradient checkpointing to save memory

tokenizer.padding_size = "left"
processor.tokenizer.padding_side = "left"

for param in model.parameters():
    param.requires_grad = False

for param in model.lm_head.parameters():
    param.requires_grad = True

for layer in model.model.layers[:5]:
    for param in layer.parameters():
        param.requires_grad = True

for name, param in model.visual.named_parameters():
    if "merger.mlp.2" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


# def extract_xml_answer(text: str) -> str:
#     answer = text.split("<answer>")[-1]
#     answer = answer.split("</answer>")[0]
#     return answer.strip()
def extract_xml_recipe(response_text: str) -> dict:
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
        # Make sure <recipe> tag exists
        if "<recipe>" not in response_text or "</recipe>" not in response_text:
            print("No <recipe> tag found in text during extract_xml_recipe")
            return None
            
        # Extract just the recipe XML portion
        recipe_start = response_text.find("<recipe>")
        recipe_end = response_text.find("</recipe>") + len("</recipe>")
        if recipe_start == -1 or recipe_end == -1:
            return None
            
        xml_text = response_text[recipe_start:recipe_end]
        
        # Parse XML
        try:
            root = ET.fromstring(xml_text)
        except Exception as e:
            print(f"Error parsing XML: {str(e)}")
            print(f"Problematic XML: {xml_text}")
            return None
            
        # Extract recipe components
        recipe = {
            "title": "",
            "ingredients": [],
            "steps": []
        }
        
        # Get title
        title_elem = root.find("title")
        if title_elem is not None and title_elem.text:
            recipe["title"] = title_elem.text.strip()
            
        # Get ingredients
        ingredients = root.find("ingredients")
        if ingredients is not None:
            for ingredient in ingredients.findall("ingredient"):
                if ingredient.text:
                    recipe["ingredients"].append(ingredient.text.strip())
                    
        # Get instructions/steps
        instructions = root.find("instructions")
        if instructions is not None:
            for step in instructions.findall("step"):
                if step.text:
                    recipe["steps"].append(step.text.strip())
                    
        return recipe
        
    except Exception as e:
        print(f"Unexpected error in extract_xml_recipe: {str(e)}")
        return None

# def format_data(sample):
#     return {
#         "prompt": [
#             {
#                 "role": "system",
#                 "content": [{"type": "text", "text": SYSTEM_PROMPT}],
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": sample["image"],
#                     },
#                     {
#                         "type": "text",
#                         "text": f"{sample['problem']}",
#                     },
#                 ],
#             },
#         ],
#         "answer": {
#             "role": "assistant",
#             "content": [{"type": "text", "text": sample["solution"]}],
#         },
#     }


def format_data(sample):
    return {
        "prompt": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"], #PIL Image object
                    },
                ],
            },
        ],
        "answer": {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["title"]}],
            "metadata": {
                "ingredients": sample["ingredients"],  # list of string
                "instructions": sample["instructions"],  # list of string
                "ingredients_embeddings": sample["ingredients_embeddings"],  # list of float
                "instructions_embeddings": sample["instructions_embeddings"],  # list of float
            },
        },
    }

def get_count_data(root: str = "data/dev_stage"):
    train_ds, dev_ds, test_ds = get_dev_stage_datasets(
        root=root,
        map_to_prompt_fn=format_data
    )
    return train_ds, dev_ds, test_ds


train_dataset, eval_dataset, test_dataset = get_count_data()


# def detect_format(text: str) -> bool:
#     """
#     Returns True if the text exactly follows the format:

#     <think>...</think>
#     <answer>...</answer>

#     where '...' can be any content (including newlines), and there is a newline separating
#     the closing </think> tag and the opening <answer> tag.
#     """
#     pattern = r"^<think>([\s\S]*?)</think>\n<answer>([\s\S]*?)</answer>$"
#     return re.fullmatch(pattern, text) is not None

def detect_format(completion: str) -> bool:
    """
    Strictly validate that the model’s text is in the form

        <think> … </think>
        <recipe>
          <title>…</title>
          <ingredients>
            <ingredient> … </ingredient>+
          </ingredients>
          <instructions>
            <step> … </step>+
          </instructions>
        </recipe>

    If – and only if – **all** structural constraints are met, return 1.0;
    otherwise return 0.0.  (GRPO expects a deterministic binary reward.)
    """

    text = completion.strip()

    # ──────────────────────────────────────────────────────────────
    # Regex design notes
    # ──────────────────────────────────────────────────────────────
    #  • ^ … $         → anchor entire string (no junk before/after)
    #  • (?:(?!<tag>).)*?  → tempered greedy token: consumes anything
    #                       *except* the opening of that <tag> again,
    #                       so we guarantee “exactly one” occurrence.
    #  • [^<]+          → at least one character inside the leaf tags
    #  • DOTALL flag    → "." matches newlines so pretty printing works
    #  • IGNORECASE     → tags are case-insensitive (conservative choice)
    #
    # We only insist on *one* <ingredient> and *one* <step> to keep the
    # check simple; the generator is free to add more – the look-ahead
    # guards allow multiple as long as the outer structure is intact.
    # ──────────────────────────────────────────────────────────────
    recipe_regex = r"""
            ^<think>                             # single think section …
                (?:(?!<think>).)*?               #   … no nested <think>
            </think>\s*                          # end </think> (trim ws)
            <recipe>                             # single recipe section
                (?:(?!<recipe>).)*?              #   … no nested <recipe>

                <title>[^<]+</title>             # required title text

                (?:(?!<recipe>).)*?              # anything until ingredients
                <ingredients>                    # open ingredients
                    (?:(?!</ingredients>).)*?    #   stuff but not </ingredients>
                    <ingredient>[^<]+</ingredient> # ≥1 ingredient
                    (?:(?!</ingredients>).)*?    #   (possibly more)
                </ingredients>                   # close ingredients

                (?:(?!<recipe>).)*?              # anything until instructions
                <instructions>                   # open instructions
                    (?:(?!</instructions>).)*?   #   stuff but not </instructions>
                    <step>[^<]+</step>           # ≥1 step
                    (?:(?!</instructions>).)*?   #   (possibly more)
                </instructions>                  # close instructions

                (?:(?!<recipe>).)*?              # anything else but no new <recipe>
            </recipe>$                           # close recipe – end of string
        """

    # Compile once for speed and readability
    pattern = re.compile(recipe_regex, re.DOTALL | re.IGNORECASE | re.VERBOSE)
    return pattern.search(text) is not None

def cosine_ingredients_reward_func(guess_ingredients_list, answer_ingredients_embeddings) -> float:
    """Calculate cosine similarity between guess ingredients and answer ingredients.
    
    Args:
        guess_ingredients_list: List of extracted ingredients from model output
        answer_ingredients_embeddings: List of embeddings for ground truth ingredients
        
    Returns:
        float: Average max cosine similarity score between 0 and 1
    """

    
    # If no ingredients were extracted, return 0
    if not guess_ingredients_list or not answer_ingredients_embeddings:
        return 0.0
        
    # If we need to generate embeddings for the guess ingredients
    # This assumes answer_ingredients_embeddings are already vectorized
    # If they're not vectorized yet, you'll need to adjust this code
    
    # Create a same vectorizer as the answer ingredients embeddings
    vectorizer = SentenceTransformer("all-MiniLM-L6-v2")
    
    try:
        guess_embeddings = vectorizer.encode(guess_ingredients_list)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(guess_embeddings, answer_ingredients_embeddings)
        
        # For each guess ingredient, find its max similarity with any answer ingredient
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Average these max similarities
        avg_similarity = np.mean(max_similarities)
        
        return float(avg_similarity)
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return 0.0  # Return 0 on error

def cosine_steps_reward_func(guess_steps_list, answer_steps_embeddings):
    """Calculate cosine similarity between guess steps and answer steps.
    
    Args:
        guess_steps_list: List of extracted steps from model output
        answer_steps_embeddings: List of embeddings for ground truth steps
        
    Returns:
        float: Average max cosine similarity score between 0 and 1
    """

    vectorizer = SentenceTransformer("all-MiniLM-L6-v2")

    if not guess_steps_list or not answer_steps_embeddings:
        return 0.0
    
    try:
        guess_embeddings = vectorizer.encode(guess_steps_list)
        similarity_matrix = cosine_similarity(guess_embeddings, answer_steps_embeddings)
        max_similarities = np.max(similarity_matrix, axis=1)
        avg_similarity = np.mean(max_similarities)
        return float(avg_similarity)
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return 0.0



    

# Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     q = prompts[0][-1]["content"]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     log_dir = os.path.dirname(LOG_FILE)
#     if log_dir and not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write("-" * 20 + "\n")
#         f.write(f"Question:\n{q[1]['text']}\n")
#         f.write(f"Answer:\n{extract_xml_answer(answer[0]['content'][0]['text'])}\n")
#         f.write(f"Response:\n{responses[0]}\n")
#         f.write(f"Extracted:\n{extracted_responses[0]}\n")
#     reward = [
#         2.0 if r == extract_xml_answer(a["content"][0]["text"]) else 0.0
#         for r, a in zip(extracted_responses, answer)
#     ]
#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(f"Correctness reward: {reward}\n\n")
#     return reward
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Compute composite reward for recipe generation using multiple components.
    
    Args:
        prompts: The prompts given to the model
        completions: The model outputs to evaluate
        answer: The ground truth answers with metadata
        
    Returns:
        list[float]: Reward scores between 0.0 and 2.0 for each completion
    """
    # Extract the raw text responses
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]

    # Extract recipe from XML where each response has 3 parts: title, ingredients, instructions
    extracted_responses = [extract_xml_recipe(r) for r in responses]
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("-" * 20 + "\n")
        f.write(f"Question:\n{q}\n")
        f.write(f"Answer (ingredients):\n{(answer[0]['metadata']['ingredients'])}\n")
        f.write(f"Answer (instructions):\n{(answer[0]['metadata']['instructions'])}\n")
        
        # Handle case where extraction might have failed
        if extracted_responses[0] is not None:
            f.write(f"Extracted Response (ingredients):\n{extracted_responses[0]['ingredients']}\n")
            f.write(f"Extracted Response (steps):\n{extracted_responses[0]['steps']}\n")
        else:
            f.write("Failed to extract valid recipe from response\n")
            
        f.write(f"Raw Response:\n{responses[0]}\n")
    reward = []

    for r, a in zip(extracted_responses, answer):
        if r is None or a is None:
            reward.append(0.0)
        else:
            # extract ingredients and steps from the answer and response
            answer_ingredients_embeddings = a['metadata']['ingredients_embeddings']
            answer_steps_embeddings = a['metadata']['instructions_embeddings']
            response_ingredients_list = r['ingredients']
            response_steps_list = r['steps']
            
            # calculate rewards
            ingredient_reward = cosine_ingredients_reward_func(response_ingredients_list, answer_ingredients_embeddings)
            step_reward = cosine_steps_reward_func(response_steps_list, answer_steps_embeddings)
            print(f"Ingredient reward: {ingredient_reward}")
            print(f"Step reward: {step_reward}")
            reward.append(ingredient_reward + step_reward)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Correctness reward: {reward}\n\n")
    return reward


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    matches = [detect_format(r) for r in responses]
    reward = [1 if match else 0.0 for match in matches]
    print(f"Strict format reward: {reward}")
    return reward


def is_conversational(example: dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`: `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if (
                isinstance(maybe_message, dict)
                and "role" in maybe_message
                and "content" in maybe_message
            ):
                return True

    return False


def collate_fn(examples):
    for idx, item in enumerate(examples):
        # Extract the system prompt
        system_prompt = item["prompt"][0]["content"][0]["text"]
        
        # Create a new prompt structure
        new_prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # Use the image directly if it's already a PIL Image, otherwise try to load it from bytes
                        "image": item["prompt"][1]["content"][0]["image"] if isinstance(item["prompt"][1]["content"][0]["image"], Image.Image) else Image.open(BytesIO(item["prompt"][1]["content"][0]["image"])),
                    },
                    # Check if there's a text prompt in the user content
                    # We don't have this in our format_data, but this accommodates the original structure
                    *([{
                        "type": "text",
                        "text": item["prompt"][1]["content"][1]["text"],
                    }] if len(item["prompt"][1]["content"]) > 1 else []),
                ],
            },
        ]
        examples[idx]["prompt"] = new_prompt
    texts = [
        processor.apply_chat_template(example["prompt"], tokenize=False)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        image_inputs.append(process_vision_info(example["prompt"])[0])

    batch = processor(
        text=texts, images=image_inputs, videos=None, return_tensors="pt", padding=True
    ).to(device)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if isinstance(
        processor, Qwen2_5_VLProcessor
    ):  # Check if the processor is Qwen2_5_VLProcessor
        image_tokens = [
            151652,
            151653,
            151655,
        ]  # Specific image token IDs for Qwen2_5_VLProcessor
    else:
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch


class VLGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, scale_rewards=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_rewards = scale_rewards

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompt_inputs = collate_fn(inputs)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -self.max_prompt_length :
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -self.max_prompt_length :
            ]
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_mask_repeated = prompt_inputs["attention_mask"].repeat_interleave(
            self.num_generations, dim=0
        )
        attention_mask = torch.cat(
            [prompt_mask_repeated, completion_mask], dim=1
        )  # (B*G, P+C)

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits  # (B, L, V)
            logits = logits[
                :, -(logits_to_keep + 1) : -1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(
                logits, input_ids[:, -logits_to_keep:]
            ):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        per_token_logps = get_per_token_logps(
            model, prompt_completion_ids, attention_mask, logits_to_keep
        )

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {
                key: []
                for key in inputs[0].keys()
                if key not in ["prompt", "completion"]
            }
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages /= std_grouped_rewards + 1e-4

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )

        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss


training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    beta=0.12,
    weight_decay=0.1,
    lr_scheduler_type="constant",
    warmup_ratio=0.05,
    logging_steps=1,
    bf16=True if device == "cuda" else False,  # Only use bf16 on CUDA
    fp16=False,  # Don't use fp16 on CPU
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2 if device == "cuda" else 2,  # Reduce for CPU
    max_prompt_length=None,
    max_completion_length=500,
    num_train_epochs=2,
    save_steps=50,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    report_to="mlflow",
)


trainer = VLGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[strict_format_reward_func, correctness_reward_func],
    args=training_args,
    train_dataset=train_dataset,
)
for name, params in model.named_parameters():
    if not params.requires_grad:
        print(name)
trainer.train()
