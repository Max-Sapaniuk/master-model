import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from random import shuffle

# Load datasets
train_data_path = './dataset/train.csv'
test_data_path = './dataset/test.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Load additional dataset
ukr_data_path = './dataset/ukr_processed.csv'
ukr_df = pd.read_csv(ukr_data_path)

# Preprocessing: Fill missing values
for col in ['text', 'title', 'author']:
    train_df[col] = train_df[col].fillna('')
    test_df[col] = test_df[col].fillna('')

# Use title, author, and text together for input
train_df['content'] = train_df['title'] + ' ' + train_df['author'] + ' ' + train_df['text']
test_df['content'] = test_df['title'] + ' ' + test_df['author'] + ' ' + test_df['text']

train_df = train_df[['content', 'label']]  # Keep only necessary columns

# Preprocess additional dataset
ukr_df['content'] = str(ukr_df['Source'] + ' ' + ukr_df['Statement'])
ukr_df = ukr_df[['content', 'label']]  # Keep only necessary columns
for item in range(len(ukr_df['label'])):
    if ukr_df['label'][item] == 2:
        ukr_df['label'][item] = 0
    elif ukr_df['label'][item] == 1:
        ukr_df['label'][item] = 1
    elif ukr_df['label'][item] == 0:
        ukr_df['label'][item] = 1


ukr_df = ukr_df[['content', 'label']]  # Keep only necessary columns

# Combine both datasets
combined_df = pd.concat([train_df, ukr_df], ignore_index=True)

# Tokenizer and model initialization
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['content'], padding='max_length', truncation=True, max_length=512)

# Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(combined_df['content'], combined_df['label'], test_size=0.1, stratify=combined_df['label'])

# Calculate class weights to handle imbalance
class_counts = train_labels.value_counts()
class_weights = torch.tensor([class_counts[1] / len(train_labels), class_counts[0] / len(train_labels)], dtype=torch.float)
print("Class Weights:", class_weights)

# Convert to dataset format
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = FakeNewsDataset(train_texts, train_labels)
val_dataset = FakeNewsDataset(val_texts, val_labels)

# Custom Trainer to apply weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights.to(model.device))  # Ensure weights are on the same device as model
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./working/model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,  # Adjusted learning rate
    logging_dir='./working/logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,  # Keep the best model
    metric_for_best_model="f1",   # Select based on F1 score
    report_to="none",  # Disable reporting to outside frameworks like WandB
    do_eval=True
)

# Early stopping callback to prevent overfitting
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# Initialize the custom WeightedTrainer with weighted loss
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./working/fake-news-model')
