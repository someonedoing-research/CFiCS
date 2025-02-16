# -*- coding: utf-8 -*-
"""htc-bert-baseline-3fold.ipynb"""

# =============================================================================
# 1) Imports and Label Mappings
# =============================================================================
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertModel,
    Trainer,
    TrainingArguments
)

# ----- Mappings -----
CF_label2id = {
    "GA": 0,
    "TA": 1,
    "B": 2,
    "": 3,
}
CF_id2label = {v: k for k, v in CF_label2id.items()}

intervention_label2id = {
    "EAR": 0,
    "CP": 1,
    "": 2  # blank / none
}
intervention_id2label = {v: k for k, v in intervention_label2id.items()}

skill_label2id = {
   "RL": 0,  # Reflective Listening
   "G": 1,   # Genuineness
   "V": 2,   # Validation
   "A": 3,   # Affirmation
   "RA": 4,  # Respect for Autonomy
   "AP": 5,  # Asking for Permission
   "OQ": 6,  # Open-ended Question
   "": 7     # Neutral
}
skill_id2label = {v: k for k, v in skill_label2id.items()}
NUM_SKILLS = len(skill_label2id)  # 8

# =============================================================================
# 2) Load DataFrame and Encode Labels
# =============================================================================
df = pd.read_csv("data/htc_examples_ids_with_pure_skills.csv")

def encode_CF(cf_label):
    return CF_label2id[cf_label]

def encode_intervention(iv_label):
    return intervention_label2id[iv_label]

def encode_skill(skill_label):
    return skill_label2id[skill_label]

df["CF"].fillna("", inplace=True)
df["IC"].fillna("", inplace=True)
df["skill"].fillna("", inplace=True)

df["CF_id"] = df["CF"].apply(encode_CF)
df["intervention_id"] = df["IC"].apply(encode_intervention)
df["skill_ids"] = df["skill"].apply(encode_skill)

# =============================================================================
# 3) Create a Hugging Face Dataset from df
# =============================================================================
dataset = Dataset.from_pandas(df)  # => We will not do train_test_split here
print("Full dataset:", dataset)

# =============================================================================
# 4) Pretrained BERT Tokenizer
# =============================================================================
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Map the tokenize function
dataset = dataset.map(tokenize_function, batched=True)

# Columns we do NOT need to feed to the model:
REMOVE_COLUMNS = ["text", "CF", "IC", "skill"]

# =============================================================================
# 5) Define Model Class (MultiTaskBert) and DataCollator
# =============================================================================
class MultiTaskBert(nn.Module):
    def __init__(self, model_name, num_cf_labels=3, num_intervention_labels=3, num_skills=8):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.cf_classifier = nn.Linear(hidden_size, num_cf_labels)
        self.intervention_classifier = nn.Linear(hidden_size, num_intervention_labels)
        self.skill_classifier = nn.Linear(hidden_size, num_skills)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, cf_labels=None, intervention_labels=None, skill_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        cf_logits = self.cf_classifier(pooled_output)
        intervention_logits = self.intervention_classifier(pooled_output)
        skill_logits = self.skill_classifier(pooled_output)

        loss = None
        if cf_labels is not None and intervention_labels is not None and skill_labels is not None:
            ce_loss_fct = nn.CrossEntropyLoss()
            loss_cf = ce_loss_fct(cf_logits, cf_labels)
            loss_interv = ce_loss_fct(intervention_logits, intervention_labels)
            loss_skills = ce_loss_fct(skill_logits, skill_labels)
            loss = loss_cf + loss_interv + loss_skills

        return {
            "loss": loss,
            "cf_logits": cf_logits,
            "intervention_logits": intervention_logits,
            "skill_logits": skill_logits,
        }

class MultiTaskDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        # Single-label integer columns
        batch["cf_labels"] = torch.tensor([f["CF_id"] for f in features], dtype=torch.long)
        batch["intervention_labels"] = torch.tensor([f["intervention_id"] for f in features], dtype=torch.long)
        batch["skill_labels"] = torch.tensor([f["skill_ids"] for f in features], dtype=torch.long)
        return batch

# =============================================================================
# 6) Define Trainer Subclass for Multi-Task
# =============================================================================
from transformers import Trainer

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_cf = inputs["cf_labels"]
        labels_interv = inputs["intervention_labels"]
        labels_skill = inputs["skill_labels"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            cf_labels=labels_cf,
            intervention_labels=labels_interv,
            skill_labels=labels_skill
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not prediction_loss_only:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    cf_labels=inputs["cf_labels"],
                    intervention_labels=inputs["intervention_labels"],
                    skill_labels=inputs["skill_labels"]
                )
            loss = outputs["loss"]
            cf_logits = outputs["cf_logits"].detach()
            intervention_logits = outputs["intervention_logits"].detach()
            skill_logits = outputs["skill_logits"].detach()

            cf_labels = inputs["cf_labels"]
            interv_labels = inputs["intervention_labels"]
            skill_labels = inputs["skill_labels"]

            return (loss,
                    (cf_logits, intervention_logits, skill_logits),
                    (cf_labels, interv_labels, skill_labels))
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# Simple metric function for demonstration
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    cf_logits = torch.tensor(logits[0])
    interv_logits = torch.tensor(logits[1])
    skill_logits = torch.tensor(logits[2])

    cf_preds = cf_logits.argmax(dim=-1)
    interv_preds = interv_logits.argmax(dim=-1)
    skill_preds = skill_logits.argmax(dim=-1)

    cf_labels = torch.tensor(labels[0])
    interv_labels = torch.tensor(labels[1])
    skill_labels = torch.tensor(labels[2])

    cf_acc = (cf_preds == cf_labels).float().mean().item()
    interv_acc = (interv_preds == interv_labels).float().mean().item()
    skill_acc = (skill_preds == skill_labels).float().mean().item()
    return {
        "cf_accuracy": cf_acc,
        "intervention_accuracy": interv_acc,
        "skill_exact_match": skill_acc
    }

# =============================================================================
# 7) K-FOLD CROSS VALIDATION (k=3)
# =============================================================================
from transformers import TrainingArguments
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

k = 3
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

# We'll collect final F1 scores across folds
cf_micro_scores = []
cf_macro_scores = []
iv_micro_scores = []
iv_macro_scores = []
skill_micro_scores = []
skill_macro_scores = []

all_indices = np.arange(len(dataset))

fold_num = 1
for train_idx, val_idx in kfold.split(all_indices):
    print(f"\n===== Fold {fold_num}/{k} =====")

    # 1) Subset the dataset using .select()
    ds_train = dataset.select(train_idx)
    ds_val = dataset.select(val_idx)

    # 2) Remove columns not needed
    ds_train = ds_train.remove_columns(REMOVE_COLUMNS)
    ds_val = ds_val.remove_columns(REMOVE_COLUMNS)

    # 3) Initialize the model (fresh for each fold)
    model = MultiTaskBert(model_name=model_name,
                          num_cf_labels=len(CF_label2id),
                          num_intervention_labels=len(intervention_label2id),
                          num_skills=NUM_SKILLS)

    # 4) Training arguments
    output_dir_fold = f"bert_multitask_cv_fold{fold_num}"
    if not os.path.exists(output_dir_fold):
        os.makedirs(output_dir_fold)

    training_args = TrainingArguments(
        output_dir=output_dir_fold,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=10,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True if device.type == "cuda" else False,
        report_to="none"
    )

    data_collator = MultiTaskDataCollator(tokenizer)

    # 5) Create a Trainer for this fold
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6) Train
    trainer.train()

    # 7) Evaluate -> get predictions for this fold
    preds_output = trainer.predict(ds_val)
    (cf_logits, iv_logits, skill_logits) = preds_output.predictions
    (cf_labels, iv_labels, skill_labels) = preds_output.label_ids

    cf_preds = np.argmax(cf_logits, axis=-1)
    iv_preds = np.argmax(iv_logits, axis=-1)
    skill_preds = np.argmax(skill_logits, axis=-1)

    # 8) Compute F1 scores
    # CF
    cf_micro = f1_score(cf_labels, cf_preds, average="micro", zero_division=0)
    cf_macro = f1_score(cf_labels, cf_preds, average="macro", zero_division=0)
    cf_micro_scores.append(cf_micro)
    cf_macro_scores.append(cf_macro)

    # Intervention
    iv_micro = f1_score(iv_labels, iv_preds, average="micro", zero_division=0)
    iv_macro = f1_score(iv_labels, iv_preds, average="macro", zero_division=0)
    iv_micro_scores.append(iv_micro)
    iv_macro_scores.append(iv_macro)

    # Skill
    skill_micro = f1_score(skill_labels, skill_preds, average="micro", zero_division=0)
    skill_macro = f1_score(skill_labels, skill_preds, average="macro", zero_division=0)
    skill_micro_scores.append(skill_micro)
    skill_macro_scores.append(skill_macro)

    print(f"Fold {fold_num} CF F1 (micro/macro): {cf_micro:.4f} / {cf_macro:.4f}")
    print(f"Fold {fold_num} IV F1 (micro/macro): {iv_micro:.4f} / {iv_macro:.4f}")
    print(f"Fold {fold_num} SKILL F1 (micro/macro): {skill_micro:.4f} / {skill_macro:.4f}")

    fold_num += 1


# =============================================================================
# 8) Aggregate Final Metrics Across 3 Folds
# =============================================================================
print("\n===== CROSS-VALIDATION RESULTS (3 FOLDS) =====")
print(f"CF micro-F1 : {np.mean(cf_micro_scores):.4f} ± {np.std(cf_micro_scores):.4f}")
print(f"CF macro-F1 : {np.mean(cf_macro_scores):.4f} ± {np.std(cf_macro_scores):.4f}")

print(f"IV micro-F1 : {np.mean(iv_micro_scores):.4f} ± {np.std(iv_micro_scores):.4f}")
print(f"IV macro-F1 : {np.mean(iv_macro_scores):.4f} ± {np.std(iv_macro_scores):.4f}")

print(f"Skill micro-F1 : {np.mean(skill_micro_scores):.4f} ± {np.std(skill_micro_scores):.4f}")
print(f"Skill macro-F1 : {np.mean(skill_macro_scores):.4f} ± {np.std(skill_macro_scores):.4f}")

print("\n=== ALL DONE! ===")