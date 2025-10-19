import argparse
from unsloth import FastLanguageModel
from scripts.dataloader import SysMLV2Dataset
from scripts.model import LLMModel
from scripts.trainer import Trainer
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on SysMLv2 data and convert to GGUF"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Phi-4-unsloth-bnb-4bit",
        help="Base model name from HuggingFace",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the CSV dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models_8B",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--save_gguf_path",
        type=str,
        default="./SysML-V2-Qwen3-8B",
        # unsloth/mistral-7b-v0.3
        help="Path to save the GGUF model",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=16000, help="Maximum sequence length"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="instruction",
        help="Format for dataset type : conversation/instruction",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="qwen-3",
        help="If using conversation mode specify chat-template.",
    )

    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Whether to load model in 4-bit quantization",
    )
    parser.add_argument(
        "--quantization_method",
        type=str,
        default="q4_k_m",
        help="Quantization method for GGUF conversion",
    )
    parser.add_argument(
        "--save_adapters_only",
        action="store_true",
        default=False,
        help="Save only adapters and not gguf models",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        logger.error("Dataset not found at %s", args.dataset_path)
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_gguf_path) or ".", exist_ok=True)

    logger.info("Loading Model and Tokenizer...")
    model_loader = LLMModel(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        r=args.lora_r,
    )

    model, tokenizer = model_loader.get_model_tokenizer()
    assert tokenizer is not None, "Tokenizer is None!"

    logger.info("Loading dataset...")
    data_loader = SysMLV2Dataset(
        csv_path=args.dataset_path,
        format=args.format,
        chat_template=args.chat_template,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )
    train_dataset = data_loader.load_data()

    logger.info("Setting up trainer...")
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )

    try:
        logger.info("Starting training...")
        metrics = trainer.train()
        logger.info("Training finished. Metrics: %s", metrics)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)

    try:
        logger.info("Saving model to %s...", args.output_dir)
        trainer.save_model(args.output_dir)
    except Exception:
        logger.exception("Saving model failed")
        sys.exit(1)

    # --- GGUF Conversion ---
    if not args.save_adapters_only:
        logger.info("Starting GGUF conversion...")
        try:
            logger.info(
                "Loading trained model from %s for GGUF conversion...", args.output_dir
            )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.output_dir,
                max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit,
            )
            if hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()

            logger.info("Saving GGUF model to %s...", args.save_gguf_path)
            model.save_pretrained_gguf(
                args.save_gguf_path,
                tokenizer,
                quantization_method=args.quantization_method,
            )
            logger.info("✅ GGUF model successfully saved at %s", args.save_gguf_path)
        except Exception:
            logger.exception("GGUF conversion failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
