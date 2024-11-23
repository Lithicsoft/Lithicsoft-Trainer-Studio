# License: Apache-2.0
 #
 # llm_finetuning/trainer.py: Trainer for LLM Finetuning model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()

model_name = os.getenv("MODEL_NAME", "gpt2")
dataset_name = os.getenv("DATASET_NAME", "wikitext")
dataset_config = os.getenv("DATASET_CONFIG", "wikitext-2-raw-v1")
output_dir = os.getenv("OUTPUT_DIR", "./results")
logging_dir = os.getenv("LOGGING_DIR", "./logs")
batch_size = int(os.getenv("BATCH_SIZE", 8))
epochs = int(os.getenv("EPOCHS", 3))
save_steps = int(os.getenv("SAVE_STEPS", 10000))

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset(dataset_name, dataset_config)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_data = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    save_steps=save_steps,
    logging_dir=logging_dir,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
)

trainer.train()
