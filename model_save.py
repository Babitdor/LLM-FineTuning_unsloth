import argparse
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser(description="Generate text using FastLanguageModel")
parser.add_argument(
    "--model_path", type=str, default="./models", help="Path to the model"
)
parser.add_argument(
    "--max_seq_length", type=int, default=2048, help="Max sequence length"
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./SysML-V2-llama3.1-Instruct/q8_0",
    help="Path to save the merged model",
)
args = parser.parse_args()

# --- Load model and tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_path, max_seq_length=args.max_seq_length, load_in_4bit=False
)
model = model.merge_and_unload()

# Save in Hugging Face format
# model.save_pretrained(args.save_path)
# tokenizer.save_pretrained(args.save_path)
model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method="q8_0")
print(f"✅ Model successfully saved at {args.save_path}")
