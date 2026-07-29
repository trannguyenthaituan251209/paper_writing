import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyvi import ViTokenizer

def get_jaccard_sim(str1, str2):
    set1 = set([str1[i:i+3] for i in range(len(str1)-2)])
    set2 = set([str2[i:i+3] for i in range(len(str2)-2)])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def main():
    print("=== RUNNING JACCARD SENSITIVITY ANALYSIS ===")
    df = pd.read_csv("dataset.csv")
    ann_cols = [f'ann_{i}' for i in range(1, 11)]
    
    # ann_cols are numeric 1 and 0 in dataset.csv
    df['spam_count'] = df[ann_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['gold_label'] = np.where(df['spam_count'] >= 6, 1, 0)
    
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    results = {}
    
    texts_raw = df['content'].tolist()
    
    for thresh in thresholds:
        print(f"\nProcessing Threshold J >= {thresh:.2f}...")
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
                
        sub_df = df.iloc[keep_indices].sample(frac=1, random_state=42).reset_index(drop=True)
        ham_count = int((sub_df['gold_label'] == 0).sum())
        spam_count = int((sub_df['gold_label'] == 1).sum())
        dup_removed = int(len(df) - len(sub_df))
        
        clean_texts = np.array([ViTokenizer.tokenize(str(t).lower()) for t in sub_df['content']])
        y = sub_df['gold_label'].values
        
        print(f"J >= {thresh:.2f}: Total = {len(sub_df)}, Ham = {ham_count}, Spam = {spam_count}, Duplicates Removed = {dup_removed}")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores, acc_scores = [], []
        
        for train_idx, test_idx in skf.split(clean_texts, y):
            X_train, X_test = clean_texts[train_idx], clean_texts[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            X_tr_vec = vec.fit_transform(X_train)
            X_te_vec = vec.transform(X_test)
            
            clf = SVC(kernel='linear', C=1.0, random_state=42)
            clf.fit(X_tr_vec, y_train)
            preds = clf.predict(X_te_vec)
            
            acc_scores.append(accuracy_score(y_test, preds))
            f1_scores.append(f1_score(y_test, preds, zero_division=0))
            
        results[f"J >= {thresh:.2f}"] = {
            'Threshold': thresh,
            'Duplicates_Removed': dup_removed,
            'Unique_Corpus_Size': len(sub_df),
            'Ham_Count': ham_count,
            'Spam_Count': spam_count,
            'Linear_SVM_Accuracy': f"{np.mean(acc_scores)*100:.2f}% ± {np.std(acc_scores)*100:.2f}%",
            'Linear_SVM_F1': f"{np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%"
        }
        
    print("\n=== SENSITIVITY RESULTS SUMMARY ===")
    print(json.dumps(results, indent=2))
    
    with open("scratch/jaccard_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
