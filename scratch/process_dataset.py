import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('dataset.csv')

print(f"Total records in dataset.csv: {len(df)}")
print("Columns:", df.columns.tolist())

# Inspect annotation columns
ann_cols = [f'ann_{i}' for i in range(1, 11)]

# Sum of positive votes (spam votes) for each message
df['spam_votes'] = df[ann_cols].sum(axis=1)

print("\n--- Voting Distribution (spam_votes out of 10) ---")
print(df['spam_votes'].value_counts().sort_index())

# Calculate consensus
# 10 votes for Spam (10) or 10 votes for Ham (0) -> Perfect Consensus (10/10)
df['is_consensus'] = df['spam_votes'].apply(lambda x: x == 10 or x == 0)
df['majority_label'] = df['spam_votes'].apply(lambda x: 'SPAM' if x >= 5 else 'HAM')

consensus_df = df[df['is_consensus']]
disagreement_df = df[~df['is_consensus']]

print(f"\nTotal Perfect Consensus (10/10): {len(consensus_df)} ({len(consensus_df)/len(df)*100:.2f}%)")
print(f"  - Consensual HAM (0/10 spam votes): {(consensus_df['spam_votes'] == 0).sum()}")
print(f"  - Consensual SPAM (10/10 spam votes): {(consensus_df['spam_votes'] == 10).sum()}")

print(f"\nTotal Active Disagreements (<10/10): {len(disagreement_df)} ({len(disagreement_df)/len(df)*100:.2f}%)")
print("\nDisagreement details by spam_votes:")
for v in range(1, 10):
    sub = df[df['spam_votes'] == v]
    print(f"  - Vote {v}/10 (Ham: {10-v}, Spam: {v}): {len(sub)} messages")

