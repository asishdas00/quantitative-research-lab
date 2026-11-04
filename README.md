# Quantitative Research Lab

A research workspace for financial machine learning, natural language processing, and data-driven analysis of financial markets.

This repository currently contains an implementation and supporting materials for **FinBERT**, a BERT-based model for classifying sentiment in financial text. The upstream project and model are credited below. This repository is a fork/adaptation workspace; original implementation details remain attributable to their respective authors.

## Project focus

- Financial text preprocessing and sentiment classification
- Using pretrained transformer models for financial language
- Producing positive, negative, and neutral sentiment predictions
- Exploring how text-derived signals can support quantitative research

## FinBERT

FinBERT is a pretrained language model adapted to financial text. The upstream model is published by Prosus AI:

- **Model:** [ProsusAI/finbert on Hugging Face](https://huggingface.co/ProsusAI/finbert)
- **Original code repository:** [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT)
- **Research paper:** [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)

The model produces sentiment classes for financial text. Review the upstream model card for supported usage and model-specific terms before using its weights.

## Repository contents

| Path | Purpose |
| --- | --- |
| `finbert/` | FinBERT-related source package |
| `notebooks/` | Research and experimentation notebooks |
| `scripts/` | Utility and model workflow scripts |
| `environment.yml` | Conda environment specification |
| `config.json` | Project configuration |
| `examples.csv`, `test.txt` | Example input data |

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate finbert
```

Inspect the scripts and notebooks for the available workflows and required input formats. Model downloads and dataset preparation may require internet access and separate dataset permissions.

## Example: load the pretrained model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "The company reported stronger-than-expected quarterly earnings."
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
```

This example loads the upstream pretrained model. It is an inference example, not a claim that this repository independently trained or validated the model.

## Research notes and extensions

The repository can be extended with independently authored experiments, such as:

- Comparing FinBERT sentiment with simple lexical sentiment baselines
- Evaluating precision, recall, F1-score, and confusion matrices on a documented test set
- Aggregating text sentiment into daily asset-level signals
- Testing whether sentiment features add value in a properly time-ordered backtest
- Recording data sources, preprocessing choices, and experiment settings for reproducibility

Any results added here should include the dataset, evaluation method, baseline, time period, and limitations. No performance claims are made until the experiments have been run and documented.

## Attribution and licensing

This repository contains third-party work. The original FinBERT implementation and model are associated with Prosus AI and their respective contributors. See the original [ProsusAI/finBERT repository](https://github.com/ProsusAI/finBERT), its contributors, and the included `LICENSE` file for source-code terms.

The upstream repository's licensing notice distinguishes the Apache-2.0-licensed code from pretrained model weights and datasets. The model weights and datasets may have separate terms; consult the [Hugging Face model card](https://huggingface.co/ProsusAI/finbert) and dataset terms before use or redistribution.

Renaming this repository does not change ownership or authorship of upstream code, model weights, research, or datasets. Contributions made in this workspace should be described separately from inherited upstream material.

## Maintainer

Maintained as a personal financial machine-learning research workspace by [Asish Das](https://github.com/asishdas00).

