import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from pyvi import ViTokenizer
import time

def main():
    print("=== FAST JACCARD SENSITIVITY ANALYSIS ===")
    t0 = time.time()
    df = pd.read_csv("dataset.csv")
    ann_cols = [f'ann_{i}' for i in range(1, 11)]
    df['spam_count'] = df[ann_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['gold_label'] = np.where(df['spam_count'] >= 6, 1, 0)
    
    texts_raw = [str(t).lower().strip() for t in df['content']]
    # Pre-compute 3-gram sets for ultrafast comparison
    gram_sets = [set([t[i:i+3] for i in range(len(t)-2)]) if len(t) >= 3 else set([t]) for t in texts_raw]
    
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    results = {}
    
    for thresh in thresholds:
        print(f"Processing J >= {thresh:.2f}...")
        keep_indices = []
        for i in range(len(texts_raw)):
            s1 = gram_sets[i]
            if not s1:
                keep_indices.append(i)
                continue
            is_dup = False
            for idx in keep_indices:
                s2 = gram_sets[idx]
                if not s2:
                    continue
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                if union > 0 and (intersection / union) >= thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
                
        sub_df = df.iloc[keep_indices].sample(frac=1, random_state=42).reset_index(drop=True)
        ham_c = int((sub_df['gold_label'] == 0).sum())
        spam_c = int((sub_df['gold_label'] == 1).sum())
        dup_rem = int(len(df) - len(sub_df))
        
        clean_texts = np.array([ViTokenizer.tokenize(str(t).lower()) for t in sub_df['content']])
        y = sub_df['gold_label'].values
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores, acc_scores = [], []
        
        for train_idx, test_idx in skf.split(clean_texts, y):
            X_tr, X_te = clean_texts[train_idx], clean_texts[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            X_tr_v = vec.fit_transform(X_tr)
            X_te_v = vec.transform(X_te)
            
            clf = SVC(kernel='linear', C=1.0, random_state=42)
            clf.fit(X_tr_v, y_tr)
            preds = clf.predict(X_te_v)
            
            acc_scores.append(accuracy_score(y_te, preds))
            f1_scores.append(f1_score(y_te, preds, zero_division=0))
            
        results[f"J >= {thresh:.2f}"] = {
            'Threshold': thresh,
            'Duplicates_Removed': dup_rem,
            'Unique_Corpus_Size': len(sub_df),
            'Ham_Count': ham_c,
            'Spam_Count': spam_c,
            'Linear_SVM_Accuracy': f"{np.mean(acc_scores)*100:.2f}% ± {np.std(acc_scores)*100:.2f}%",
            'Linear_SVM_F1': f"{np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%"
        }
        
    print(f"\nCompleted in {time.time()-t0:.2f}s!")
    print(json.dumps(results, indent=2))
    
    with open("scratch/jaccard_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
