import torch
import evaluate
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

# Load BLEU and CHRF metrics
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

# Define the model name and tokenizer
model_name = "m2m100_finetuned_kk_ru/checkpoint-4500"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to("cuda")

# Set the language pairs
tokenizer.src_lang = "kk"
tokenizer.tgt_lang = "ru"


test_dataset = load_dataset("issai/kazparc", 'kazparc', split="test")
test_dataset = test_dataset.filter(lambda x: x['pair'] == 'kk_ru')
test_dataset = test_dataset.remove_columns(['id', 'domain'])

# batch size and max length for translation
max_length = 128
batch_size = 32


# Implement dynamic batching
def dynamic_batching_with_indices(texts, max_tokens=960):
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


def translate_text(dataset):
    model.eval()
    texts = dataset['source_lang']
    dynamic_batches, batch_indices = dynamic_batching_with_indices(texts)

    all_translations = []
    with torch.no_grad():
        for batch_texts, indices in tqdm(zip(dynamic_batches, batch_indices), desc="Translating batches"):
            inputs = tokenizer(batch_texts, return_tensors="pt",
                               padding=True, truncation=True).to("cuda")
            outputs = model.generate(**inputs, max_length=max_length,
                                     num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
            translations = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)

            indexed_translations = list(zip(indices, translations))
            all_translations.extend(indexed_translations)

    all_translations.sort(key=lambda x: x[0])
    sorted_translations = [translation for index,
                           translation in all_translations]

    return sorted_translations


# Evaluate on the Kazprc
translated_texts = translate_text(test_dataset)
test_dataset = test_dataset.to_pandas()
test_dataset['predictions'] = translated_texts

scores_bleu = bleu.compute(
    predictions=test_dataset['predictions'], references=test_dataset['target_lang'])
scores_chrf = chrf.compute(
    predictions=test_dataset['predictions'], references=test_dataset['target_lang'])

print("BLEU Score:", scores_bleu)
print("CHRF Score:", scores_chrf)

# Evaluate on FLORES dataset
flores_test_kaz = load_dataset("facebook/flores", "kaz_Cyrl")['dev']
flores_test_rus = load_dataset("facebook/flores", "rus_Cyrl")['dev']

flores_test_kaz = flores_test_kaz.remove_columns(['id'])


# Translate the FLORES dataset
translated_texts_flores = translate_text(flores_test_kaz)

# Evaluate translations on FLORES dataset
scores_bleu_flores = bleu.compute(
    predictions=translated_texts_flores, references=flores_test_rus['sentence'])
scores_chrf_flores = chrf.compute(
    predictions=translated_texts_flores, references=flores_test_rus['sentence'])

print("FLORES BLEU Score:", scores_bleu_flores)
print("FLORES CHRF Score:", scores_chrf_flores)

# Verify translations
print("Example Translation:")
print("Kazakh:", flores_test_kaz['sentence'][51])
print("Russian (Target):", flores_test_rus['sentence'][51])
print("Russian (Predicted):", translated_texts_flores[51])
