import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')
ann_cols = [f'ann_{i}' for i in range(1, 11)]

# Matrix N x K where N = 2314, K = 2 (0=Ham, 1=Spam)
n_ij = np.zeros((len(df), 2))
n_ij[:, 1] = df[ann_cols].sum(axis=1).values # Spam count
n_ij[:, 0] = 10 - n_ij[:, 1] # Ham count

N, n = len(df), 10

# Fleiss' Kappa calculation
P_i = (np.sum(n_ij**2, axis=1) - n) / (n * (n - 1))
P_bar = np.mean(P_i)

p_j = np.sum(n_ij, axis=0) / (N * n)
P_e_bar = np.sum(p_j**2)

fleiss_kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

print(f"--- Fleiss' Kappa Calculation ---")
print(f"Total Annotated Messages (N): {N}")
print(f"P_bar (Observed Agreement): {P_bar:.6f}")
print(f"P_e_bar (Expected Agreement by chance): {P_e_bar:.6f}")
print(f"Fleiss' Kappa (kappa): {fleiss_kappa:.6f}")

# Near-duplicate character 3-gram Jaccard deduplication (Optimized)
def get_char_3grams(text):
    text = str(text).lower().strip()
    if len(text) < 3:
        return set([text])
    return set([text[i:i+3] for i in range(len(text)-2)])

df['char_3grams'] = [get_char_3grams(t) for t in df['content']]

# Create index mapping 3grams to message indices for fast candidate searching
from collections import defaultdict
ngram_index = defaultdict(list)
for idx, grams in enumerate(df['char_3grams']):
    for g in grams:
        ngram_index[g].append(idx)

removed_indices = set()
duplicates_info = []

for i in range(len(df)):
    if i in removed_indices:
        continue
    setA = df['char_3grams'].iloc[i]
    lenA = len(setA)
    if lenA == 0:
        continue
    
    # Candidate items that share at least one 3-gram with item i
    candidates = set()
    for g in setA:
        for c in ngram_index[g]:
            if c > i and c not in removed_indices:
                candidates.add(c)
                
    for c in candidates:
        setB = df['char_3grams'].iloc[c]
        intersection = len(setA & setB)
        union = lenA + len(setB) - intersection
        sim = intersection / union if union > 0 else 0
        if sim >= 0.85:
            removed_indices.add(c)
            duplicates_info.append((i, c, sim))

unique_df = df.drop(index=list(removed_indices)).copy()
print(f"\n--- Deduplication Results (Jaccard >= 0.85) ---")
print(f"Total raw dataset: {len(df)}")
print(f"Near-duplicates removed (J >= 0.85): {len(removed_indices)}")
print(f"Unique benchmark records remaining: {len(unique_df)}")

# Breakdown of duplicates by majority vote label
dup_df = df.iloc[list(removed_indices)].copy()
dup_df['majority_label'] = dup_df[ann_cols].sum(axis=1).apply(lambda x: 'SPAM' if x >= 5 else 'HAM')
print("\nBreakdown of removed duplicates (by majority vote):")
print(dup_df['majority_label'].value_counts())

# Unique benchmark breakdown by majority vote label
unique_df['majority_label'] = unique_df[ann_cols].sum(axis=1).apply(lambda x: 'SPAM' if x >= 5 else 'HAM')
print("\nUnique benchmark subset breakdown (by majority vote):")
print(unique_df['majority_label'].value_counts())

# Save metadata for next steps
summary_stats = {
    'total_raw': len(df),
    'fleiss_kappa': fleiss_kappa,
    'consensus_10_10': int((df[ann_cols].sum(axis=1).isin([0, 10])).sum()),
    'disagreement_count': int((~df[ann_cols].sum(axis=1).isin([0, 10])).sum()),
    'duplicates_removed': len(removed_indices),
    'unique_benchmark': len(unique_df),
    'unique_ham': int((unique_df['majority_label'] == 'HAM').sum()),
    'unique_spam': int((unique_df['majority_label'] == 'SPAM').sum())
}

import json
with open('scratch/dataset_summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\nSummary stats saved to scratch/dataset_summary_stats.json")
