from unsloth import FastLanguageModel


class LLMModel:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
        r: int = 16,
        target_modules=None,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bias: str = "none",
        use_gradient_checkpointing="unsloth",
        random_state: int = 42,
        dtype=None,
    ):
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]

        # Load base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
        )

    def get_model_tokenizer(self):
        """Return the model and tokenizer for training."""
        return self.model, self.tokenizer
