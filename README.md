# Fine-Tuning BERT Model Using HuggingFace IMDB Dataset and Trainer from HuggingFace

This repository contains a Jupyter Notebook demonstrating how to fine-tune a BERT model for sentiment analysis using the HuggingFace Trainer API and the IMDB dataset.

## Contents

**Files in the repository:**
- FIne_tuning_Using_Trainer_from_Transformers_library.ipynb

## Notebook Overview

The notebook guides you through the following steps:

### Environment Setup
- Installs required Python packages: torch, transformers, datasets, scikit-learn.
- Checks for GPU availability.

### Data Loading and Exploration
- Loads the IMDB sentiment analysis dataset using HuggingFace’s datasets library.
- Displays sample data for verification.

### Tokenization
- Uses BERT’s tokenizer (google-bert/bert-base-cased) to preprocess text data.
- Defines a function for batch tokenization and applies it to the dataset splits.

### Model Preparation
- Loads BertForSequenceClassification from HuggingFace Transformers with 2 output labels (positive/negative).
- Moves the model to GPU if available.

### Training Setup
- Defines accuracy metric using the evaluate library.
- Sets up training arguments such as output directory and evaluation strategy.
- Optionally integrates with Weights & Biases (wandb) for experiment tracking.

### Training and Evaluation
- Instantiates a Trainer object with the model, training arguments, datasets, and metrics.
- Trains the model and displays training progress, loss, and accuracy.
- Evaluates the model on the test set and prints out metrics.

### Saving and Inference
- Saves the trained model locally.
- Demonstrates inference on a custom text sample, printing its predicted sentiment.

## Requirements
- Python 3.10 or newer
- Jupyter Notebook
- GPU recommended (for efficient training)

**Python Packages:**
- torch
- transformers
- datasets
- scikit-learn
- evaluate
- wandb (optional, for logging)

## Usage
1. Clone the repository.
2. Open FIne_tuning_Using_Trainer_from_Transformers_library.ipynb in Jupyter or Google Colab.
3. Run the cells in order.
4. Modify or extend the notebook to suit your sentiment analysis needs or experiment with other datasets/models.

## Reference Links
- [HuggingFace Transformers Documentation](https://huggingface.co/transformers/)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [IMDB Dataset on HuggingFace Hub](https://huggingface.co/datasets/stanfordnlp/imdb)

For more details and updates, view the notebook and files directly in the [GitHub repository](https://github.com/ChapelFob80930/Fine-Tuning-Bert-Model-Using-HuggingFace-IMDB-Dataset-and-Trainer-from-HuggingFace).