import pandas as pd
import numpy as np
import json

def get_jaccard_sim(str1, str2):
    set1 = set([str1[i:i+3] for i in range(len(str1)-2)])
    set2 = set([str2[i:i+3] for i in range(len(str2)-2)])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def main():
    df = pd.read_csv("dataset.csv")
    ann_cols = [f'ann_{i}' for i in range(1, 11)]
    df['spam_count'] = (df[ann_cols] == 'SPAM').sum(axis=1)
    df['gold_label'] = np.where(df['spam_count'] >= 6, 1, 0)
    
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    texts_raw = df['content'].tolist()
    
    for thresh in thresholds:
        keep_indices = []
        for i in range(len(texts_raw)):
            t1 = str(texts_raw[i]).lower().strip()
            is_dup = False
            for idx in keep_indices:
                t2 = str(texts_raw[idx]).lower().strip()
                if get_jaccard_sim(t1, t2) >= thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
                
        sub_df = df.iloc[keep_indices]
        ham_c = (sub_df['gold_label'] == 0).sum()
        spam_c = (sub_df['gold_label'] == 1).sum()
        print(f"J >= {thresh:.2f}: Total = {len(sub_df)}, Ham = {ham_c}, Spam = {spam_c}, Duplicates Removed = {len(df) - len(sub_df)}")

if __name__ == '__main__':
    main()
