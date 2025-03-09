import re
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datasets import load_dataset, Dataset
import transformers
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoProcessor,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import rotate_half
from qwen_vl_utils import process_vision_info
from trl import GRPOConfig, GRPOTrainer
from trl.models.utils import unwrap_model_for_generation
from typing import Any
from io import BytesIO
from PIL import Image


def custom_apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section, unsqueeze_dim=1
):
    # Removed: mrope_section = mrope_section * 2 otherwise will cause error
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Monkey patching the function
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = (
    custom_apply_multimodal_rotary_pos_emb
)

SYSTEM_PROMPT = """Respond in the following format:
<think>...</think>
<answer>...</answer>"""

model_name = "./outputs/Qwen-0.5B-GRPO-Count-SFT-v2/checkpoint-9500"
LOG_FILE = "response_log.txt"
output_dir = "outputs/Qwen-0.5B-GRPO-Count-R1"
run_name = "Qwen-0.5B-GRPO-Count-R1"
max_pixels = 256 * 256
processor = AutoProcessor.from_pretrained(
    model_name, max_pixels=max_pixels, use_cache=False
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    use_cache=False,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=False)
tokenizer.padding_size = "left"
processor.tokenizer.padding_side = "left"

for param in model.parameters():
    param.requires_grad = False

for layer in model.model.layers[:5]:
    for param in layer.parameters():
        param.requires_grad = True

for name, param in model.visual.named_parameters():
    if "merger.mlp.2" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


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
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": f"{sample['problem']}",
                    },
                ],
            },
        ],
        "answer": {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["solution"]}],
        },
    }


def get_count_data() -> Dataset:
    data = load_dataset(
        "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex", split="train[:10000]"
    )
    data = data.map(format_data, remove_columns=["image"], num_proc=2)
    return data


train_dataset = get_count_data()


def detect_format(text: str) -> bool:
    """
    Returns True if the text exactly follows the format:

    <think>...</think>
    <answer>...</answer>

    where '...' can be any content (including newlines), and there is a newline separating
    the closing </think> tag and the opening <answer> tag.
    """
    pattern = r"^<think>([\s\S]*?)</think>\n<answer>([\s\S]*?)</answer>$"
    return re.fullmatch(pattern, text) is not None


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("-" * 20 + "\n")
        f.write(f"Question:\n{q[1]['text']}\n")
        f.write(f"Answer:\n{extract_xml_answer(answer[0]['content'][0]['text'])}\n")
        f.write(f"Response:\n{responses[0]}\n")
        f.write(f"Extracted:\n{extracted_responses[0]}\n")
    reward = [
        2.0 if r == extract_xml_answer(a["content"][0]["text"]) else 0.0
        for r, a in zip(extracted_responses, answer)
    ]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Correctness reward: {reward}\n\n")
    return reward


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    matches = [detect_format(r) for r in responses]
    reward = [0.5 if match else 0.0 for match in matches]
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
        new_prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": item["prompt"][0]["content"][0]["text"]}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(
                            BytesIO(item["prompt"][1]["content"][0]["image"]["bytes"])
                        ),
                    },
                    {
                        "type": "text",
                        "text": item["prompt"][1]["content"][1]["text"],
                    },
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
    ).to("cuda")

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
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

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
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
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
    reward_funcs=[correctness_reward_func],
    args=training_args,
    train_dataset=train_dataset,
)
for name, params in model.named_parameters():
    if not params.requires_grad:
        print(name)
trainer.train()
