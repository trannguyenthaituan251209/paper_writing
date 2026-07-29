import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load deduplicated dataset
df = pd.read_csv('dataset.csv')
ann_cols = [f'ann_{i}' for i in range(1, 11)]
df['label'] = df[ann_cols].sum(axis=1).apply(lambda x: 1 if x >= 5 else 0)
df['content'] = df['content'].fillna("").astype(str)

def get_char_3grams(text):
    text = str(text).lower().strip()
    if len(text) < 3:
        return set([text])
    return set([text[i:i+3] for i in range(len(text)-2)])

df['char_3grams'] = [get_char_3grams(t) for t in df['content']]

removed_indices = set()
items = [(i, df['char_3grams'].iloc[i], len(df['char_3grams'].iloc[i])) for i in range(len(df))]

for i in range(len(items)):
    if items[i][0] in removed_indices:
        continue
    idx_i, setA, lenA = items[i]
    if lenA == 0:
        continue
    for j in range(i + 1, len(items)):
        idx_j, setB, lenB = items[j]
        if idx_j in removed_indices:
            continue
        if min(lenA, lenB) / max(lenA, lenB) < 0.85:
            continue
        intersection = len(setA & setB)
        union = lenA + lenB - intersection
        if intersection / union >= 0.85:
            removed_indices.add(idx_j)

unique_df = df.drop(index=list(removed_indices)).reset_index(drop=True)
print(f"Unique benchmark dataset: {len(unique_df)} records")

model_name = "vinai/phobert-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("PhoBERT tokenizer loaded successfully!")
except Exception as e:
    print("Tokenizer load error:", e)
