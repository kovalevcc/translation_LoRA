import re
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = False

# Initialize wandb
wandb.init()

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lightblue/suzume-llama-3-8B-multilingual",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Get PEFT model
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

# Prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

alma_prompt = """Translate this from Kazakh to Russian:\n Kazakh: {} \nRussian: {}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instruction = 'Translate given text from Kazakh to Russian language'
    inputs = examples["source_lang"]
    outputs = examples["target_lang"]
    texts = [alma_prompt.format(input_text, output_text) +
             EOS_TOKEN for input_text, output_text in zip(inputs, outputs)]
    return {"text": texts}


# Load and prepare dataset
dataset = load_dataset("issai/kazparc", 'kazparc', split="train")
dataset = dataset.filter(lambda example: example['pair'] == 'kk_ru')
dataset = dataset.remove_columns(['id', 'domain'])
dataset = dataset.map(formatting_prompts_func, batched=True)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=291,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_steps=250,
        report_to='wandb'
    ),
)

trainer_stats = trainer.train()

# Save the model
model.save_pretrained_merged("model_final", tokenizer, save_method="lora")
model.save_pretrained("lora_model_final")
