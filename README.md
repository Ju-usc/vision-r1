## Requirements
use `uv` for faster installation.
```
python -m venv myenv
source myenv/bin/activate
uv pip install -r requirements.txt
```

## Repository Structure

This repository is organized as follows:

```
├── data/                           # Directory for datasets (e.g., Food Ingredients and Recipe Dataset with Images)
├── outputs/                        # Directory for trained model checkpoints, logs, and generated outputs
├── stage2_grpo.py                  # Main Python script for Stage 2 GRPO training of the VLMs
├── grpo_stage2_notebook.ipynb      # Jupyter Notebook to facilitate training in environments like Google Colab
├── utils.py                        # Utility functions for data preprocessing, XML parsing, embedding generation, etc.
├── evals.py                        # Python script defining evaluation metrics (Cosine Sim, BLEU, ROUGE, LLM-as-Judge)
├── eval_pipeline.ipynb             # Jupyter Notebook for running the evaluation pipeline on model outputs
├── README.md                       
├── requirements.txt                
└── .gitignore                      
```

## Dataset

This project utilizes the **Food Ingredients and Recipes Dataset with Images** sourced from Kaggle.

- **Kaggle Link**: [Food Ingredients and Recipes Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images/data)

It contains food images along with their corresponding titles, ingredients, and instructions, which are essential for training and evaluating the inverse cooking models.

## Methodology

Our approach to inverse cooking involves fine-tuning a Vision-Language Model (VLM) using Group-Relative Policy Optimisation (GRPO).

### Model
We utilize the `Qwen/Qwen2.5-VL-3B-Instruct` model as our base. The fine-tuning process is primarily managed by the `stage2_grpo.py` script.

### Training
The GRPO training process aims to teach the model to generate accurate and well-formatted recipes directly from food images. A key aspect is the system prompt (defined in `stage2_grpo.py`), which instructs the model to output recipes in a specific XML format, including a `<think>...</think>` block where the model articulates its reasoning process.

### Reward Functions
The GRPO algorithm relies on reward signals to guide the learning process. Our reward mechanism, implemented in `stage2_grpo.py`, includes:
1.  **XML Format Correctness**: A binary reward (`strict_format_reward_func`) to ensure the model adheres to the specified XML structure.
2.  **Content Quality (`correctness_reward_func`)**: This composite reward assesses:
    *   **Ingredient Similarity**: To account for variations in order or phrasing, the similarity for each predicted ingredient is determined by finding its highest cosine similarity score against all ground-truth ingredients. These individual best scores are then averaged. Embeddings are generated using `sentence-transformers` (`all-MiniLM-L6-v2`), a process handled in `utils.py`.
    *   **Instruction Similarity**: A similar approach is used for instruction steps. For each predicted step, its highest cosine similarity score against all ground-truth steps is identified. These top scores are then averaged to produce the final instruction similarity reward.

### Data Processing
The `utils.py` script contains essential functions for:
*   Parsing recipes from the dataset and converting them into the XML format expected by the model during training.
*   Generating sentence embeddings for ingredients and instructions.
*   Encoding images to base64 for model input.
*   Preprocessing the dataset to include these embeddings and encoded images.

## Evaluation

We employ a multi-faceted evaluation strategy, with core logic defined in `evals.py` and execution managed via the `eval_pipeline.ipynb` notebook.

### Metrics
1.  **Cosine Similarity**: Measures the semantic similarity between the `sentence-transformer` embeddings of predicted vs. ground-truth ingredients and instruction steps. This leverages embedding generation from `utils.py` and similarity computation in `evals.py`.
2.  **BLEU (Bilingual Evaluation Understudy) Score**: Assesses n-gram overlap between predicted and reference recipes, indicating fluency and precision. Calculated in `evals.py`.
3.  **ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score**: Measures recall of n-grams, focusing on content overlap. We specifically use ROUGE-1 and ROUGE-L, computed in `evals.py`.
4.  **LLM-as-a-Judge**: We leverage Google's Gemini Pro model (interfaced in `evals.py`) as an impartial judge. The LLM evaluates:
    *   The overall feasibility of the predicted recipe.
    *   Its alignment with the input image and the ground-truth recipe.
    *   The logical consistency of the model's reasoning process if a `<think>` tag is present in the output.
    This provides a qualitative score and textual feedback.

The `eval_pipeline.ipynb` notebook is used to run these evaluations systematically across different model outputs (e.g., from various checkpoints or baseline models) and aggregate the average scores.

## Usage

### Dependencies
Install the required Python packages using:
```bash
pip3 install -r requirements.txt
```
You may also need to set up API keys for services like Weights & Biases or Google GenAI (for `evals.py`) if you intend to use those features for logging or LLM-as-a-Judge evaluations. Check `utils.py` for Together API key usage if running baseline models.

### Training
The primary method for training the model is by using the `grpo_stage2_notebook.ipynb`. This notebook is particularly well-suited for environments like Google Colab and automates:
1.  Cloning the repository.
2.  Installing dependencies from `requirements.txt`.
3.  Setting up authentications (e.g., Hugging Face token, Weights & Biases).
4.  Executing the `stage2_grpo.py` training script with appropriate configurations.

Alternatively, after setting up your local environment and installing dependencies, you can run the training script directly:
```bash
python3 stage2_grpo.py
```
Ensure your dataset is correctly placed and paths within the script (e.g., for `model_name`, `output_dir`) are configured as needed.

### Evaluation
To evaluate your trained models or baseline outputs:
1.  Ensure your model outputs are in the expected `.jsonl` format, where each line contains the model-generated XML recipe and corresponding ground truth data.
2.  Run the `eval_pipeline.ipynb` notebook. This notebook will:
    *   Load the model outputs and the evaluation dataset.
    *   Iterate through the data, calling the `compute_evals` function from `evals.py` for each sample.
    *   Print and aggregate the evaluation metrics.


## Stage GRPO:
```
# modify the model name with your own chekcpoint
python stage2_grpo.py
```
This will also generate a `response_log.txt` to observe the response generated by the policy model.

## Acknowledgements
This project utilizes and builds upon the GRPO implementation for VLMs, originally open-sourced by [JacksonCakes](https://github.com/JacksonCakes/vision-r1).