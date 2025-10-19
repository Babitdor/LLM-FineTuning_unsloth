import argparse
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser(description="Generate text using FastLanguageModel")
parser.add_argument(
    "--model_path", type=str, default="./models/output", help="Path to the model"
)
parser.add_argument(
    "--prompt", type=str, required=True, help="Prompt text for generation"
)
parser.add_argument(
    "--max_seq_length", type=int, default=4096, help="Max sequence length"
)
parser.add_argument(
    "--max_new_tokens", type=int, default=1024, help="Max tokens to generate"
)
parser.add_argument(
    "--temperature", type=float, default=0.7, help="Sampling temperature"
)
parser.add_argument(
    "--top_p", type=float, default=0.9, help="Top-p sampling probability"
)
args = parser.parse_args()

# --- Load model and tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_path,
    max_seq_length=args.max_seq_length,
    dtype=None,
    load_in_4bit=False,
)

FastLanguageModel.for_inference(model)

chat_template = """
        Below is an instruction that describes a systems engineering task using SysMLv2. Write the appropriate SysMLv2 textual notation code.

        ### Task:
        {instruction}

        ### SysMLv2 Code:
        {output}"""

# --- Tokenize prompt ---
inputs = tokenizer(
    chat_template.format(instruction=args.prompt), return_tensors="pt"
).to(model.device)

# --- Generate output ---
outputs = model.generate(
    **inputs,
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    temperature=args.temperature,
    top_p=args.top_p
)

# --- Decode and print ---
print(tokenizer.decode(outputs, skip_special_tokens=True))
