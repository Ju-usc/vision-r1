import torch

from datasets import load_dataset, Dataset, Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from trl import SFTConfig, SFTTrainer
from datasets import disable_caching
from PIL import Image as PILImage, UnidentifiedImageError
from io import BytesIO

disable_caching()

SYSTEM_PROMPT = """Respond in the following format:
<think>...</think>
<answer>...</answer>"""

model_name = "jacksonkek/qwen-0.5-vl-custom"

output_dir = "outputs/Qwen-0.5B-GRPO-Count-SFT-base"
run_name = "Qwen-0.5B-GRPO-Count-SFT-base"
max_pixels = 256 * 256
processor = AutoProcessor.from_pretrained(
    model_name, max_pixels=max_pixels, use_cache=False
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
    use_cache=False,
).to("cuda")
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


def is_valid_image(image_dict):
    """
    Given an image dictionary (with raw bytes), try to open the image.
    Returns True if the image can be successfully loaded; otherwise, False.
    """
    image_bytes = image_dict.get("bytes")
    try:
        img = PILImage.open(BytesIO(image_bytes))
        img.load()
        return True
    except (UnidentifiedImageError, Exception) as e:
        print("Skipping image due to error:", e)
        return False


def format_data(sample):
    image_dict = sample["image"]
    pil_image = PILImage.open(BytesIO(image_dict["bytes"]))
    pil_image.load()
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "text",
                    "text": sample["problem"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{sample['thinking']}\n{sample['solution']}"}
            ],
        },
    ]


def get_data() -> Dataset:
    train_data, val_data = load_dataset(
        "MMInstruction/Clevr_CoGenT_TrainA_R1", split=["train[:90%]", "train[90%:]"]
    )
    train_data = train_data.cast_column("image", Image(decode=False))
    val_data = val_data.cast_column("image", Image(decode=False))

    return train_data, val_data


train_dataset, eval_dataset = get_data()
valid_train_dataset = [
    sample for sample in train_dataset if is_valid_image(sample["image"])
]
valid_eval_dataset = [
    sample for sample in eval_dataset if is_valid_image(sample["image"])
]
train_dataset = [format_data(example) for example in valid_train_dataset]
eval_dataset = [format_data(example) for example in valid_eval_dataset]


# Configure training arguments
training_args = SFTConfig(
    output_dir=output_dir,  # Directory to save the model
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="cosine",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=500,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=500,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,
    report_to="mlflow",
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    max_seq_length=2048,  # Maximum sequence length for input
)

training_args.remove_unused_columns = (
    False  # Keep unused columns in dataset since we might use it later
)


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    labels = batch["input_ids"].clone()
    labels[
        labels == processor.tokenizer.pad_token_id
    ] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(
        processor, Qwen2_5_VLProcessor
    ):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [
            151652,
            151653,
            151655,
        ]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    answer_delimiter = "assistant\n"

    # Process each example individually to mask the question part
    for i, text in enumerate(texts):
        # Find where the answer starts in the raw text
        delimiter_index = text.find(answer_delimiter)
        if delimiter_index == -1:
            continue

        # Tokenize the portion of the text that comes before the answer.
        question_text = text[: delimiter_index + len(answer_delimiter)]
        question_token_ids = processor.tokenizer.encode(
            question_text, add_special_tokens=False
        )
        question_length = len(question_token_ids)

        # Set the corresponding tokens in the labels to -100 so that the loss is not computed on them.
        labels[i, :question_length] = -100

    batch["labels"] = labels
    return batch


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)
# Check parameters that are not trainable
for name, params in model.named_parameters():
    if not params.requires_grad:
        print(name)
trainer.train()
