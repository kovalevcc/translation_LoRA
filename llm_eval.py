import re
import pandas as pd
import torch
from datasets import concatenate_datasets, Dataset, load_dataset
from evaluate import evaluate
from peft import PeftModel, get_peft_model, PeftConfig
from tqdm.auto import tqdm
from unsloth import FastLanguageModel


# Load evaluation metrics
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

# Define model parameters
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = False

# Load the base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="archive/BESTcheckpoint-7250",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Load the LoRA model configuration
lora_config_path = "outputs_suzume_32_full/checkpoint-600"
peft_config = PeftConfig.from_pretrained(lora_config_path)

# Apply the LoRA to the base model
model = get_peft_model(model, peft_config)

# Define prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instruction = 'Translate given text to Russian language'
    inputs = examples["source_lang"]
    outputs = ''
    texts = [
        alpaca_prompt.format(instruction, input_text, outputs) + EOS_TOKEN
        for input_text in inputs
    ]
    return {"text": texts}


# Load and prepare test dataset
test = load_dataset("issai/kazparc", 'kazparc', split="test")
test = test.filter(lambda x: x['pair'] == 'kk_ru')
test = test.remove_columns(['id', 'domain'])
test = test.select(range(100))

# Apply formatting to the test dataset
test = test.map(formatting_prompts_func, batched=True)

# Define dynamic batching function


def dynamic_batching_with_indices(texts, max_tokens=1024):
    batches = []
    batch_indices = []
    current_batch = []
    current_indices = []
    current_tokens_count = 0

    for index, text in enumerate(texts):
        num_tokens = len(tokenizer.encode(text))
        if current_tokens_count + num_tokens > max_tokens:
            batches.append(current_batch)
            batch_indices.append(current_indices)
            current_batch = [text]
            current_indices = [index]
            current_tokens_count = num_tokens
        else:
            current_batch.append(text)
            current_indices.append(index)
            current_tokens_count += num_tokens

    if current_batch:
        batches.append(current_batch)
        batch_indices.append(current_indices)

    return batches, batch_indices

# Define translation function


def translate_text(dataset):
    model.eval()
    texts = dataset['text']

    dynamic_batches, batch_indices = dynamic_batching_with_indices(texts)
    all_translations = []

    with torch.no_grad():
        for batch_texts, indices in tqdm(zip(dynamic_batches, batch_indices), desc="Translating batches"):
            inputs = tokenizer(batch_texts, return_tensors="pt",
                               padding=True, truncation=True).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=64)
            translations = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)

            clean_translations = [
                re.search(r"### Response:\n(.*)", text, re.DOTALL).group(1)
                if re.search(r"### Response:\n(.*)", text, re.DOTALL)
                else text for text in translations
            ]

            indexed_translations = list(zip(indices, clean_translations))
            all_translations.extend(indexed_translations)

    all_translations.sort(key=lambda x: x[0])
    sorted_translations = [translation for index,
                           translation in all_translations]

    return sorted_translations


# Translate and evaluate
translated_texts = translate_text(test)
test = test.to_pandas()
test['predictions'] = translated_texts

# Evaluation
scores_bleu = bleu.compute(
    predictions=test['predictions'], references=test['target_lang'])
scores_chrf = chrf.compute(
    predictions=test['predictions'], references=test['target_lang'])

# Print evaluation results
print("BLEU Score:", scores_bleu)
print("CHRF Score:", scores_chrf)

# Load Flores dataset and format
flores_test_kaz = load_dataset("facebook/flores", "kaz_Cyrl")


def flores_formatting_prompts_func(examples):
    instruction = 'Translate given text to Russian language'
    inputs = examples["sentence"]
    texts = [
        alpaca_prompt.format(instruction, input_text, "") + EOS_TOKEN
        for input_text in inputs
    ]
    return {"text": texts}


flores_test = flores_test_kaz.map(flores_formatting_prompts_func, batched=True)
flores_test_rus = load_dataset("facebook/flores", "rus_Cyrl")

translated_texts_flores = translate_text(flores_test['dev'][:10])

# Evaluate Flores translations
scores_bleu_flores = bleu.compute(
    predictions=translated_texts_flores, references=flores_test_rus['dev']['sentence'][:10])
scores_chrf_flores = chrf.compute(
    predictions=translated_texts_flores, references=flores_test_rus['dev']['sentence'][:10])

# Print Flores evaluation results
print("BLEU Score:", scores_bleu_flores)
print("CHRF Score:", scores_chrf_flores)
