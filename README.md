# Fine Tuning Script - Using Unsloth

## Overview

This provides a pipeline for fine-tuning large language models (LLMs) on SysMLv2 datasets and exporting them to GGUF format for efficient inference.
The script allows training with LoRA adapters, dataset formatting (instruction or conversation), and supports quantized GGUF export.

## Features

- Fine-tune HuggingFace-compatible models on SysMLv2 datasets.
- Supports LoRA (Low-Rank Adaptation).
- Dataset formatting:
  - Instruction-based
  - Conversation-based with optional chat templates.
- Training with configurable:
  - Batch size
  - Gradient accumulation
  - Sequence length
  - Learning rate
  - Epochs
- Save trained model in HuggingFace format.
- Convert fine-tuned models to GGUF format (compatible with llama.cpp and other lightweight inference frameworks).
- Supports 4-bit quantization for memory-efficient training/inference.

## Repository Structure

```
LLM-SE_FAU/
├── data/                # Datasets and experimental data
├── models/              # Modelfiles configurations and fine-tuned models '.gguf'
├── notebook/            # Documentation and papers
├── scripts/             # Source code for tools and implementations
├── evaluate.py
├── train.py
└── requirements.txt     # Python dependencies

```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git for version control

### Installation

1. Clone the repository:

```bash
git clone https://github.com/FAPS-LLM-SE/LLM-SE-FineTuning.git
cd LLM-SE-FineTuning
```

2. Create and activate a virtual environment:

```bash
python -m venv <name of enivornment>
source  <name of enivornment>/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Format

The dataset should be in CSV format with fields depending on --format:

- Instruction mode:

`instruction,output`

- Conversation mode (with --chat-template):

`role,content`

## Training & Conversion

Run the train script:

```bash
python train.py \
  --dataset_path ./data/sysml_dataset.csv \
  --output_dir ./models \
  --save_gguf_path ./SysML-V2-Phi-4 \
  --model_name unsloth/Phi-4-unsloth-bnb-4bit \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
  --learning_rate 1e-4 \
  --max_seq_length 2048 \
  --format instruction \
  --lora_r 32 \
  --load_in_4bit False \
  --quantization_method q4_k_m
```

## Arguments

| Argument                        | Type  | Default                                  | Description                                                    |
| ------------------------------- | ----- | ---------------------------------------- | -------------------------------------------------------------- |
| `--model_name`                  | str   | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | Base model from HuggingFace                                    |
| `--dataset_path`                | str   | **Required**                             | Path to CSV dataset file                                       |
| `--output_dir`                  | str   | `./models`                               | Directory to save fine-tuned model                             |
| `--save_gguf_path`              | str   | `./models/Model_gguf`                    | Path to save GGUF model                                        |
| `--batch_size`                  | int   | 1                                        | Training batch size                                            |
| `--gradient_accumulation_steps` | int   | 4                                        | Number of gradient accumulation steps                          |
| `--num_epochs`                  | int   | 5                                        | Training epochs                                                |
| `--learning_rate`               | float | `1e-4`                                   | Learning rate                                                  |
| `--max_seq_length`              | int   | 2048                                     | Maximum sequence length                                        |
| `--format`                      | str   | `instruction`                            | Dataset format: `instruction` or `conversation`                |
| `--chat-template`               | str   | None                                     | Chat template (if using conversation mode)                     |
| `--lora_r`                      | int   | 32                                       | LoRA rank                                                      |
| `--load_in_4bit`                | bool  | False                                    | Load model in 4-bit quantization                               |
| `--quantization_method`         | str   | `q4_k_m`                                 | Quantization method for GGUF export `q8_0`, `q4_k_m`, `q5_k_m` |

### Note:

- `q8_0` - Fast conversion. High resource use, but generally acceptable.
- `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
- `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

### Example Workflow

- Prepare dataset in CSV format.
- Run train.py with your parameters.
- Choose an unsloth model of your choice, and load the appropriate chat-template.
- Export to GGUF at (specified model output path) for lightweight inference.

## GGUF Export

- After training, the model is automatically:
- Reloaded from the output directory.
- Merged with LoRA (if applicable).
- Converted to GGUF with chosen quantization method.
- Use with llama.cpp or any GGUF-compatible inference runtime.

### Note:

- Before Training make sure to setup up llama.cpp, according to this repository [llama.cpp](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html?utm_source=chatgpt.com) in the root folder.

- `llama.cpp` will automatically be cloned during the model saving process but it might cause errors if it's not build yet. So, it's better to set it up beforehand and build it according to the process in this link [here](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html?utm_source=chatgpt.com)

## Evaluation

It also includes a script for evaluating models on SysMLv2 code generation tasks.
The evaluation pipeline compares model predictions against ground truth using both text-based and structural code metrics.

### How It Works

- Dataset Loading

  - Loads a CSV dataset with instruction–output (or prompt–label) pairs.

  - Supports common column names such as Instruction, prompt, output, target, or label.

- Model Inference with Ollama

  - Uses ollama.chat API to query one or more Ollama models.

  - Enforces strict formatting rules so the model must respond with a single code block (no explanations, comments, or extra text).


### Running Evaluation

### Example usage:

```
python evaluate.py \
  --dataset ./data/sysml_eval.csv \
  --models phi4-sysml mistral-sysml \
  --output results/eval_sysml.json \
  --fraction 0.2 \
  --seed 123
```

### Arguments

| Argument     | Type       | Default           | Description                                                   |
| ------------ | ---------- | ----------------- | ------------------------------------------------------------- |
| `--dataset`  | Path       | **Required**      | Path to dataset file (CSV/JSONL with prompts and outputs).    |
| `--models`   | List\[str] | `llama3.1:latest` | One or more Ollama model names to evaluate.                   |
| `--output`   | Path       | `./results`       | Path to save predictions and metrics (`.json`).               |
| `--fraction` | float      | `1.0`             | Fraction of dataset to sample for evaluation (0.0 < f ≤ 1.0). |
| `--seed`     | int        | `42`              | Random seed for reproducible sampling.                        |
