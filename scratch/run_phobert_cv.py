import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_text = unique_df['content'].values
y_true = unique_df['label'].values

oof_predictions = np.zeros(len(unique_df))
accs, precs, recs, f1s = [], [], [], []

print("\n--- Starting PhoBERT 5-Fold Cross Validation ---")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y_true)):
    print(f"\nTraining Fold {fold + 1}/5...")
    train_texts, val_texts = X_text[train_idx], X_text[test_idx]
    train_labels, val_labels = y_true[train_idx], y_true[test_idx]

    train_ds = SMSDataset(train_texts, train_labels, tokenizer)
    val_ds = SMSDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 4 # 4 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_val_f1 = 0.0
    best_preds = None

    for epoch in range(4):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_preds = np.array(val_preds)
        val_f1 = f1_score(val_labels, val_preds, pos_label=1)
        if val_f1 > best_val_f1 or epoch == 3:
            best_val_f1 = val_f1
            best_preds = val_preds

    oof_predictions[test_idx] = best_preds
    fold_acc = accuracy_score(val_labels, best_preds)
    fold_prec = precision_score(val_labels, best_preds, pos_label=1)
    fold_rec = recall_score(val_labels, best_preds, pos_label=1)
    fold_f1 = f1_score(val_labels, best_preds, pos_label=1)

    accs.append(fold_acc)
    precs.append(fold_prec)
    recs.append(fold_rec)
    f1s.append(fold_f1)

    print(f"Fold {fold+1} Results -> Acc: {fold_acc*100:.2f}%, Prec: {fold_prec*100:.2f}%, Rec: {fold_rec*100:.2f}%, F1: {fold_f1*100:.2f}%")

phobert_metrics = {
    'accuracy_mean': float(np.mean(accs) * 100),
    'accuracy_std': float(np.std(accs) * 100),
    'precision_mean': float(np.mean(precs) * 100),
    'precision_std': float(np.std(precs) * 100),
    'recall_mean': float(np.mean(recs) * 100),
    'recall_std': float(np.std(recs) * 100),
    'f1_mean': float(np.mean(f1s) * 100),
    'f1_std': float(np.std(f1s) * 100)
}

# Bootstrap 95% Confidence Interval for PhoBERT
boot_accs, boot_precs, boot_recs, boot_f1s = [], [], [], []
n_samples = len(y_true)
np.random.seed(42)

for b in range(1000):
    boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    y_b = y_true[boot_idx]
    pred_b = oof_predictions[boot_idx]

    boot_accs.append(accuracy_score(y_b, pred_b) * 100)
    boot_precs.append(precision_score(y_b, pred_b, pos_label=1, zero_division=0) * 100)
    boot_recs.append(recall_score(y_b, pred_b, pos_label=1, zero_division=0) * 100)
    boot_f1s.append(f1_score(y_b, pred_b, pos_label=1, zero_division=0) * 100)

phobert_cis = {
    'acc_ci': [float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))],
    'prec_ci': [float(np.percentile(boot_precs, 2.5)), float(np.percentile(boot_precs, 97.5))],
    'rec_ci': [float(np.percentile(boot_recs, 2.5)), float(np.percentile(boot_recs, 97.5))],
    'f1_ci': [float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))]
}

print("\n==========================================================================")
print("PHOBERT-BASE FINAL CV RESULTS")
print("==========================================================================")
print(f"Accuracy : {phobert_metrics['accuracy_mean']:.2f}% ± {phobert_metrics['accuracy_std']:.2f}% | 95% CI: [{phobert_cis['acc_ci'][0]:.2f}%, {phobert_cis['acc_ci'][1]:.2f}%]")
print(f"Precision: {phobert_metrics['precision_mean']:.2f}% ± {phobert_metrics['precision_std']:.2f}% | 95% CI: [{phobert_cis['prec_ci'][0]:.2f}%, {phobert_cis['prec_ci'][1]:.2f}%]")
print(f"Recall   : {phobert_metrics['recall_mean']:.2f}% ± {phobert_metrics['recall_std']:.2f}% | 95% CI: [{phobert_cis['rec_ci'][0]:.2f}%, {phobert_cis['rec_ci'][1]:.2f}%]")
print(f"F1-Score : {phobert_metrics['f1_mean']:.2f}% ± {phobert_metrics['f1_std']:.2f}% | 95% CI: [{phobert_cis['f1_ci'][0]:.2f}%, {phobert_cis['f1_ci'][1]:.2f}%]")

# Save PhoBERT results
with open('scratch/phobert_results.json', 'w', encoding='utf-8') as f:
    json.dump({'metrics': phobert_metrics, 'cis': phobert_cis}, f, indent=2)

print("\nPhoBERT results saved to scratch/phobert_results.json")
