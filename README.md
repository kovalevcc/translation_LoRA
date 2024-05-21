# Enhancing Kazakh-Russian Translation with LLM and LoRA

This repository contains the implementation and evaluation of a translation model based on the Large Language Model Suzume 8B, fine-tuned with rank-stabilized LoRA on the KazParC dataset to improve translation from Kazakh to Russian.

## Project Overview

Translation models for medium-resource languages like Kazakh currently underperform, making the exploration of new methodologies crucial. This project utilizes the Suzume 8B model, applying the rank-stabilized LoRA fine-tuning technique to enhance its performance in translation. The project uses the SFT method of training in the Kazakh-Russian language pair.

## Project Structure

The project consists of the following files:

```plaintext
├── llm_train.py       # Script for training the LLM model
├── llm_eval.py        # Script for evaluating the LLM model
├── m2m_train.py       # Script for training Facebook M2M model
├── m2m_eval.py        # Script for evaluating Facebook M2M model
