from datasets import load_dataset
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import wandb


wandb.init()

train_dataset = load_dataset("issai/kazparc", 'kazparc', split="train")
train_dataset = train_dataset.filter(filter_kk_ru)
train_dataset = train_dataset.remove_columns(['id', 'domain'])

validation_dataset = load_dataset(
    "issai/kazparc", 'kazparc', split="validation")
validation_dataset = validation_dataset.filter(filter_kk_ru)
validation_dataset = validation_dataset.select(
    range(len(validation_dataset) // 2))
validation_dataset = validation_dataset.remove_columns(['id', 'domain'])

# Load BLEU metric
bleu_metric = evaluate.load("bleu")

# Load model and tokenizer
model_name = "m2m100_finetuned_kk_ru/checkpoint-4500"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Preprocess datasets

def filter_kk_ru(example):
    return example['pair'] == 'kk_ru'


def preprocess(examples):
    source_lang = "kk"
    target_lang = "ru"
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang
    inputs = examples["source_lang"]
    targets = examples["target_lang"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


train_dataset = train_dataset.map(
    preprocess, batched=True, remove_columns=train_dataset.column_names)
validation_dataset = validation_dataset.map(
    preprocess, batched=True, remove_columns=validation_dataset.column_names)

training_args = Seq2SeqTrainingArguments(
    output_dir="./m2m100_finetuned_kk_ru_more_epochs",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=10,
    predict_with_generate=True,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[label] for label in tokenizer.batch_decode(
        labels, skip_special_tokens=True)]
    results = bleu_metric.compute(predictions=decoded_preds, references=labels)
    return {"bleu": results["bleu"]}


# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
