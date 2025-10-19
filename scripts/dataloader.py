from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from string import Template


class SysMLV2Dataset:
    def __init__(
        self,
        csv_path,
        tokenizer,
        format="instruction",
        chat_template="chatml",
        max_length=2048,
    ):  # instruction or conversation
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template
        self.format = format
        self.EOS_TOKEN = tokenizer.eos_token

        if format not in ["instruction", "conversation"]:
            raise ValueError("format must be either 'instruction' or 'conversation'")

    def instruction_formatting_func(self, samples):
        texts = []
        for instruction, output in zip(samples["Instruction"], samples["output"]):
            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            ### Instruction:
            {}
            ### Input:
            {}
            ### Response:
            {}"""
            text = alpaca_prompt.format(instruction, "", output) + self.EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    def conversation_formatting_func(self, samples):
        tokenizer = get_chat_template(
            self.tokenizer,
            # chat_template=self.chat_template,
            mapping={
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt",
            },
            map_eos_token=True,
        )

        # Create conversations directly
        conversations = []
        for i in range(len(samples["Instruction"])):
            messages = [
                {"from": "human", "value": samples["Instruction"][i]},
                {"from": "gpt", "value": samples["output"][i]},
            ]
            conversations.append(messages)

        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in conversations
        ]
        return {
            "text": texts,
        }

    def load_data(self):
        dataset = load_dataset("csv", data_files=self.csv_path, split="train")

        if self.format == "instruction":
            train = dataset.map(
                self.instruction_formatting_func,
                batched=True,
            )
            return train
        else:
            train = dataset.map(self.conversation_formatting_func, batched=True)
            return train
