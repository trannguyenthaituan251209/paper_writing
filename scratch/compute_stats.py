import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')
ann_cols = [f'ann_{i}' for i in range(1, 11)]

# Matrix N x K where N = 2314, K = 2 (0=Ham, 1=Spam)
# n_i0 = count of 0s, n_i1 = count of 1s
n_ij = np.zeros((len(df), 2))
n_ij[:, 1] = df[ann_cols].sum(axis=1).values # Spam count
n_ij[:, 0] = 10 - n_ij[:, 1] # Ham count

N, n = len(df), 10

# Fleiss' Kappa calculation
# P_i = 1/(n(n-1)) * sum_j(n_ij^2 - n)
P_i = (np.sum(n_ij**2, axis=1) - n) / (n * (n - 1))
P_bar = np.mean(P_i)

# p_j = 1/(N*n) * sum_i(n_ij)
p_j = np.sum(n_ij, axis=0) / (N * n)
P_e_bar = np.sum(p_j**2)

fleiss_kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

print(f"--- Fleiss' Kappa Calculation ---")
print(f"P_bar (Observed Agreement): {P_bar:.6f}")
print(f"P_e_bar (Expected Agreement by chance): {P_e_bar:.6f}")
print(f"Fleiss' Kappa (kappa): {fleiss_kappa:.6f}")

# Near-duplicate character 3-gram Jaccard deduplication
def get_char_3grams(text):
    text = str(text).lower().strip()
    return set([text[i:i+3] for i in range(len(text)-2)])

def jaccard_sim(setA, setB):
    if not setA or not setB:
        return 0.0
    return len(setA & setB) / len(setA | setB)

print("\n--- Running Near-Duplicate Filtering (Character 3-gram Jaccard >= 0.85) ---")
grams = [get_char_3grams(t) for t in df['content']]
removed_indices = set()
duplicates_info = []

for i in range(len(df)):
    if i in removed_indices:
        continue
    for j in range(i + 1, len(df)):
        if j in removed_indices:
            continue
        sim = jaccard_sim(grams[i], grams[j])
        if sim >= 0.85:
            removed_indices.add(j)
            duplicates_info.append((i, j, sim))

unique_df = df.drop(index=list(removed_indices)).copy()
print(f"Total raw dataset: {len(df)}")
print(f"Near-duplicates removed (J >= 0.85): {len(removed_indices)}")
print(f"Unique benchmark records remaining: {len(unique_df)}")

# Breakdown of duplicates by majority vote label
dup_df = df.iloc[list(removed_indices)]
dup_df['majority_label'] = dup_df[ann_cols].sum(axis=1).apply(lambda x: 'SPAM' if x >= 5 else 'HAM')
print("Breakdown of removed duplicates (by majority vote):")
print(dup_df['majority_label'].value_counts())

# Unique benchmark breakdown by majority vote label
unique_df['majority_label'] = unique_df[ann_cols].sum(axis=1).apply(lambda x: 'SPAM' if x >= 5 else 'HAM')
print("\nUnique benchmark subset breakdown:")
print(unique_df['majority_label'].value_counts())

