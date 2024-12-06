# Google Taxonomy Classifier using DeBERTa

This repository contains a Python script (`train.py`) for fine-tuning a DeBERTa-v3-base model to classify text into the 21 top-level categories of the Google Taxonomy.  Future work will extend this to the full 5000+ category taxonomy.

## Overview

The script uses the Hugging Face Transformers library to train a DeBERTa model on a provided CSV dataset.  Key features include:

* **Class Weighting:** Addresses class imbalance in the training data.
* **Early Stopping:** Prevents overfitting by monitoring validation loss.
* **Weighted Loss Function:**  Uses a weighted cross-entropy loss to account for class imbalances.
* **Comprehensive Evaluation:** Reports precision, recall, F1-score, and accuracy.


## Requirements

Ensure you have the following Python packages installed:

```bash
pip install pandas scikit-learn transformers torch wandb
