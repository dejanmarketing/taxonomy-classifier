import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import EarlyStoppingCallback

# Step 1: Handle W&B to ignore errors and continue training
try:
    import wandb
except Exception as e:
    print(f"Warning: W&B encountered an issue: {e}. Training will continue without W&B.")
    os.environ["WANDB_MODE"] = "disabled"

# Step 2: Load the dataset
df = pd.read_csv("train.csv")

# Step 3: Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Step 4: Extract texts and labels from the dataframe
train_texts = train_df['text'].tolist()
train_labels = train_df['class'].tolist()
val_texts = val_df['text'].tolist()
val_labels = val_df['class'].tolist()

# Step 5: Load the DeBERTa tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

# Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Step 6: Create a custom Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Step 7: Calculate class weights to handle class imbalance
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_labels), y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Step 8: Create the datasets
train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# Step 9: Load the DeBERTa model for sequence classification
model = DebertaV2ForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base', 
    num_labels=len(set(train_labels))
)

# Modify the Trainer to include class weights in the loss function
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Step 10: Define custom compute_metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Step 11: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',                # output directory
    num_train_epochs=10,                   # number of training epochs
    per_device_train_batch_size=7,         # batch size for training
    per_device_eval_batch_size=14,         # batch size for evaluation
    warmup_steps=500,                      # number of warmup steps for learning rate scheduler
    weight_decay=0.1,                      # stronger regularization
    logging_dir='./logs',                  # directory for storing logs
    logging_steps=1,                       # log every step
    evaluation_strategy="epoch",           # evaluation strategy to adopt during training
    save_strategy="epoch",                 # save checkpoint after every epoch
    save_total_limit=None,                 # save all checkpoints
    load_best_model_at_end=True,           # load the best model at the end of training
    learning_rate=1e-5,                    # reduced learning rate
    max_grad_norm=1.0                      # clip gradients to stabilize updates
)

# Step 12: Initialize the WeightedTrainer with early stopping
trainer = WeightedTrainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # validation dataset
    compute_metrics=compute_metrics,     # metric calculation function
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # early stopping
)

# Step 13: Fine-tune the model
try:
    trainer.train()
except Exception as e:
    print(f"Warning: An error occurred during training: {e}. Continuing without W&B logging.")

# Step 14: Save the model, tokenizer, and config
model.save_pretrained("./fine_tuned_deberta_model")
tokenizer.save_pretrained("./fine_tuned_deberta_model")
model.config.to_json_file("./fine_tuned_deberta_model/config.json")
