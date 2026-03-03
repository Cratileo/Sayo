import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from datasets import load_dataset, Features, Value, Image
import datasets
import accelerate

from trainer import NeoGRPOTrainerInternVL, NeoGRPOTrainer
from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    GRPOTrainer
)
from reward import attn_reward, METER_reward, answer_format_reward
import base64
import io

class QwenVLGRPOTrainer():
    def __init__(self, praser:TrlParser):
        self.script_args, self.training_args, self.model_args = parser.parse_args_and_config()

    def train(self):
        trainer = NeoGRPOTrainer(
            model=self.model_args.model_name_or_path,
            args=self.training_args,
            reward_funcs=[attn_reward, answer_format_reward],
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=get_peft_config(self.model_args),
        )
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
            
        trainer.train(resume_from_checkpoint=checkpoint)


    def model_init(self):
        torch_dtype = (
            self.model_args.dtype if self.model_args.dtype in ["auto", None] else getattr(torch, self.model_args.dtype)
        )
        # torch_dtype = torch.bfloat16
        self.training_args.model_init_kwargs = dict(
            revision=self.model_args.model_revision,
            attn_implementation=self.model_args.attn_implementation,
            dtype=torch_dtype
        )


    def load_data(self):
        features = Features({
            "image":Image(),
            "question": Value("string"), 
            "answer": Value("string"),
            "short_answer": Value("string"),
            'focus_area': datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
            "type": Value('string'),
            "id": Value('string'),
        })
        dataset = load_dataset("parquet", data_files=self.script_args.dataset_name, split='train', features=features)
        dataset = dataset.train_test_split(test_size=0.01, seed=42)

        SYSTEM_PROMPT = (
            "A conversation between a user and an assistant. The user asks a question, and the assistant must answer it."
            "The assistant first conducts internal reasoning, followed immediately by the final answer enclosed within <answer>...</answer> tags."
            "Requirements:"
            "1. The reasoning outside <answer></answer> should be concise, minimal, and essential.\n"
            "2. The final answer must be correct, complete, and directly address the user’s question.\n"
            "3. When reasoning, focus on the areas of aim object in the image.\n"
            "4. You MUST HAVE thinking process before the final answer.\n"
        )

        def make_conversation(example):
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
            return {"prompt": prompt}

        dataset = dataset.map(make_conversation)

        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"] if self.training_args.eval_strategy != "no" else None


    def start(self):
        self.model_init()
        self.load_data()
        self.train()

if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    Trainer = QwenVLGRPOTrainer(parser)
    Trainer.start()