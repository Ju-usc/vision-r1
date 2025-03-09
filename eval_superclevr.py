from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os

MODEL_PATH = "./outputs/Qwen-0.5B-GRPO-Count-SFT-base/checkpoint-500"
BSZ = 64  # reduce it if GPU OOM
OUTPUT_PATH = "./logs/counting_results_superclevr.json"
PROMPT_PATH = "./superclevr_test_200_counting.jsonl"
SYSTEM_PROMPT = """Respond in the following format:
<think>...</think>
<answer>...</answer>"""

max_pixels = 256 * 256
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, max_pixels=max_pixels, use_cache=False
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    use_cache=False,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))

messages = []

for i in data:
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{i['image_path']}"},
                {"type": "text", "text": i["question"]},
            ],
        },
    ]
    messages.append(message)


all_outputs = []  # List to store all answers

# Process data in batches
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i : i + BSZ]

    # Preparation for inference
    text = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]

    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, use_cache=True, max_new_tokens=1024, do_sample=False
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")


def extract_number_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r"<answer>\s*(\d+)\s*</answer>"
    match = re.search(answer_pattern, output_str)

    if match:
        return int(match.group(1))
    return None


final_output = []
correct_number = 0

for input_example, model_output in zip(data, all_outputs):
    original_output = model_output
    ground_truth = input_example["ground_truth"]
    model_answer = extract_number_answer(original_output)

    # Create a result dictionary for this example
    result = {
        "question": input_example,
        "ground_truth": ground_truth,
        "model_output": original_output,
        "extracted_answer": model_answer,
    }
    final_output.append(result)

    # Count correct answers
    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

output_dir = os.path.dirname(OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)

output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({"accuracy": accuracy, "results": final_output}, f, indent=2)

print(f"Results saved to {output_path}")
