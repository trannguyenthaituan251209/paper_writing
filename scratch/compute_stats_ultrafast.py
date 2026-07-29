import pandas as pd
import numpy as np
import json

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

# Multi-rater Agreement Breakdown
df['spam_votes'] = n_ij[:, 1].astype(int)
df['majority_label'] = df['spam_votes'].apply(lambda x: 'SPAM' if x >= 5 else 'HAM')

vote_counts = df['spam_votes'].value_counts().to_dict()
print("\nVote Distribution Breakdown:")
for v in range(11):
    print(f"  Votes {v}/10 SPAM ({10-v}/10 HAM): {vote_counts.get(v, 0)}")

# Near-duplicate character 3-gram Jaccard deduplication (Vectorized/Fast)
def get_char_3grams(text):
    text = str(text).lower().strip()
    if len(text) < 3:
        return set([text])
    return set([text[i:i+3] for i in range(len(text)-2)])

df['char_3grams'] = [get_char_3grams(t) for t in df['content']]

# Fast deduplication with length bound filter
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
        # Length bound filter: if min/max length < 0.85, Jaccard cannot be >= 0.85
        if min(lenA, lenB) / max(lenA, lenB) < 0.85:
            continue
        
        intersection = len(setA & setB)
        union = lenA + lenB - intersection
        if intersection / union >= 0.85:
            removed_indices.add(idx_j)

unique_df = df.drop(index=list(removed_indices)).copy()
print(f"\n--- Deduplication Results (Jaccard >= 0.85) ---")
print(f"Total raw dataset: {len(df)}")
print(f"Near-duplicates removed (J >= 0.85): {len(removed_indices)}")
print(f"Unique benchmark records remaining: {len(unique_df)}")

# Breakdown of duplicates by majority vote label
dup_df = df.iloc[list(removed_indices)].copy()
print("\nBreakdown of removed duplicates (by majority vote):")
print(dup_df['majority_label'].value_counts())

# Unique benchmark breakdown by majority vote label
print("\nUnique benchmark subset breakdown (by majority vote):")
print(unique_df['majority_label'].value_counts())

# Save metadata to JSON
summary_stats = {
    'total_raw': int(len(df)),
    'fleiss_kappa': float(fleiss_kappa),
    'P_bar': float(P_bar),
    'P_e_bar': float(P_e_bar),
    'consensus_10_10': int((df['spam_votes'].isin([0, 10])).sum()),
    'consensus_ham_10_10': int((df['spam_votes'] == 0).sum()),
    'consensus_spam_10_10': int((df['spam_votes'] == 10).sum()),
    'disagreement_count': int((~df['spam_votes'].isin([0, 10])).sum()),
    'disagreement_breakdown': {v: int(vote_counts.get(v, 0)) for v in range(1, 10)},
    'duplicates_removed': int(len(removed_indices)),
    'duplicate_ham': int((dup_df['majority_label'] == 'HAM').sum()),
    'duplicate_spam': int((dup_df['majority_label'] == 'SPAM').sum()),
    'unique_benchmark': int(len(unique_df)),
    'unique_ham': int((unique_df['majority_label'] == 'HAM').sum()),
    'unique_spam': int((unique_df['majority_label'] == 'SPAM').sum())
}

with open('scratch/dataset_summary_stats.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

print("\nSummary stats successfully saved to scratch/dataset_summary_stats.json")
