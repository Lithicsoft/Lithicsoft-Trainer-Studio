# License: Apache-2.0
 #
 # llm_finetuning/trainer.py: Trainer for LLM Finetuning model in Trainer Studio
 #
 # (C) Copyright 2024 Lithicsoft Organization
 # Author: Bui Nguyen Tan Sang <tansangbuinguyen52@gmail.com>
 #

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
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

lora_r = int(os.getenv("LORA_R", 8))
lora_alpha = int(os.getenv("LORA_ALPHA", 32))
lora_dropout = float(os.getenv("LORA_DROPOUT", 0.1))
lora_bias = os.getenv("LORA_BIAS", "none")

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=lora_bias,
)

model = get_peft_model(base_model, lora_config)

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

model.save_pretrained(output_dir)
