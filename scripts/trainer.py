from trl import SFTTrainer, SFTConfig
import os
import logging
from typing import Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset: Optional[Any] = None,
        output_dir: str = "./models/output",
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 20,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        bf16: bool = True,
        fp16: bool = False,
        logging_steps: int = 10,
        optimizer: str = "adamw_8bit",
        save_strategy: str = "epoch",
        logging_dir: str = "./logs",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        if optimizer not in {"adamw_8bit", "adamw_hf", "adamw_torch"}:
            logger.warning(
                "Unknown optimizer '%s', proceeding but consider using a supported optimizer.",
                optimizer,
            )

        if bf16 and fp16:
            logger.warning("Both bf16 and fp16 requested; using fp16 by priority.")
            bf16 = False

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # type: ignore
            dataset_text_field="text",  # type: ignore
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                optim=optimizer,
                greater_is_better=True,
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=output_dir,
                logging_dir=logging_dir,
                save_strategy=save_strategy,
                bf16=bf16,
                fp16=fp16,
                report_to="none",
            ),
        )

    def train(self):
        """Run the fine-tuning process and return training metrics if available."""
        try:
            logger.info("Starting training")
            result = self.trainer.train()
            logger.info("Training finished")
            return result  # trainer.train() may return a TrainOutput or None depending on TRL version
        except Exception as e:
            logger.exception("Training failed: %s", e)
            raise

    def evaluate(self):
        """Run evaluation on the validation dataset and return metrics."""
        if self.val_dataset is None:
            logger.warning("No validation dataset provided.")
            return None
        logger.info("Starting evaluation")
        return self.trainer.evaluate()  # type: ignore

    def save_model(self, save_path: str):
        """Save the fine-tuned model and tokenizer."""
        os.makedirs(save_path, exist_ok=True)
        logger.info("Saving model to %s", save_path)
        self.trainer.model.save_pretrained(save_path)  # type: ignore
        self.trainer.tokenizer.save_pretrained(save_path)  # type: ignore
