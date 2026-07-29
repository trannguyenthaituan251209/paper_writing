import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load dataset
df = pd.read_csv('dataset.csv')
ann_cols = [f'ann_{i}' for i in range(1, 11)]
df['spam_votes'] = df[ann_cols].sum(axis=1)
df['label'] = df['spam_votes'].apply(lambda x: 1 if x >= 5 else 0) # 1=SPAM, 0=HAM
df['content'] = df['content'].fillna("").astype(str)

# 2. Near-duplicate filtering (Character 3-gram Jaccard >= 0.85)
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
print(f"Dataset after deduplication: {len(unique_df)} unique records (HAM: {(unique_df['label']==0).sum()}, SPAM: {(unique_df['label']==1).sum()})")

# 3. Stratified 5-Fold Cross Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Multi-Layer Perceptron': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'Multinomial Naive Bayes': MultinomialNB(alpha=1.0)
}

results = {}
oof_predictions = {name: np.zeros(len(unique_df)) for name in models}

X_text = unique_df['content'].values
y_true = unique_df['label'].values

for name, model in models.items():
    accs, precs, recs, f1s = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y_true)):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        X_train = vectorizer.fit_transform(X_text[train_idx])
        X_test = vectorizer.transform(X_text[test_idx])
        
        y_train, y_test = y_true[train_idx], y_true[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        oof_predictions[name][test_idx] = preds
        
        accs.append(accuracy_score(y_test, preds))
        precs.append(precision_score(y_test, preds, pos_label=1))
        recs.append(recall_score(y_test, preds, pos_label=1))
        f1s.append(f1_score(y_test, preds, pos_label=1))
        
    results[name] = {
        'accuracy_mean': float(np.mean(accs) * 100),
        'accuracy_std': float(np.std(accs) * 100),
        'precision_mean': float(np.mean(precs) * 100),
        'precision_std': float(np.std(precs) * 100),
        'recall_mean': float(np.mean(recs) * 100),
        'recall_std': float(np.std(recs) * 100),
        'f1_mean': float(np.mean(f1s) * 100),
        'f1_std': float(np.std(f1s) * 100)
    }

print("\n==========================================================================")
print("TABLE 9: BASELINE CLASSIFICATION PERFORMANCE (Mean ± Std %)")
print("==========================================================================")
print(f"{'Model Architecture':<25} | {'Accuracy (%)':<15} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15}")
print("-" * 95)
for name, m in results.items():
    acc_str = f"{m['accuracy_mean']:.2f} ± {m['accuracy_std']:.2f}"
    prec_str = f"{m['precision_mean']:.2f} ± {m['precision_std']:.2f}"
    rec_str = f"{m['recall_mean']:.2f} ± {m['recall_std']:.2f}"
    f1_str = f"{m['f1_mean']:.2f} ± {m['f1_std']:.2f}"
    print(f"{name:<25} | {acc_str:<15} | {prec_str:<15} | {rec_str:<15} | {f1_str:<15}")

# 4. Bootstrap 95% Confidence Intervals (B = 1000)
bootstrap_cis = {}
np.random.seed(42)

for name in models:
    oof_pred = oof_predictions[name]
    boot_accs, boot_precs, boot_recs, boot_f1s = [], [], [], []
    n_samples = len(y_true)
    
    for b in range(1000):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_b = y_true[boot_idx]
        pred_b = oof_pred[boot_idx]
        
        boot_accs.append(accuracy_score(y_b, pred_b) * 100)
        boot_precs.append(precision_score(y_b, pred_b, pos_label=1, zero_division=0) * 100)
        boot_recs.append(recall_score(y_b, pred_b, pos_label=1, zero_division=0) * 100)
        boot_f1s.append(f1_score(y_b, pred_b, pos_label=1, zero_division=0) * 100)
        
    bootstrap_cis[name] = {
        'acc_ci': [float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))],
        'prec_ci': [float(np.percentile(boot_precs, 2.5)), float(np.percentile(boot_precs, 97.5))],
        'rec_ci': [float(np.percentile(boot_recs, 2.5)), float(np.percentile(boot_recs, 97.5))],
        'f1_ci': [float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))]
    }

print("\n==========================================================================")
print("TABLE 10: BOOTSTRAP 95% CONFIDENCE INTERVALS (B=1000)")
print("==========================================================================")
print(f"{'Model Architecture':<25} | {'Accuracy 95% CI':<20} | {'Precision 95% CI':<20} | {'Recall 95% CI':<20} | {'F1-Score 95% CI':<20}")
print("-" * 115)
for name, ci in bootstrap_cis.items():
    acc_ci_str = f"[{ci['acc_ci'][0]:.2f}%, {ci['acc_ci'][1]:.2f}%]"
    prec_ci_str = f"[{ci['prec_ci'][0]:.2f}%, {ci['prec_ci'][1]:.2f}%]"
    rec_ci_str = f"[{ci['rec_ci'][0]:.2f}%, {ci['rec_ci'][1]:.2f}%]"
    f1_ci_str = f"[{ci['f1_ci'][0]:.2f}%, {ci['f1_ci'][1]:.2f}%]"
    print(f"{name:<25} | {acc_ci_str:<20} | {prec_ci_str:<20} | {rec_ci_str:<20} | {f1_ci_str:<20}")

# Save full results to JSON
output_data = {
    'unique_records': len(unique_df),
    'unique_ham': int((unique_df['label']==0).sum()),
    'unique_spam': int((unique_df['label']==1).sum()),
    'baseline_metrics': results,
    'bootstrap_cis': bootstrap_cis
}

with open('scratch/experimental_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("\nExperimental results saved to scratch/experimental_results.json")
