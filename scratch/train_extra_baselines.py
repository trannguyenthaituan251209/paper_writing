import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_jaccard_sim(str1, str2):
    set1 = set([str1[i:i+3] for i in range(len(str1)-2)])
    set2 = set([str2[i:i+3] for i in range(len(str2)-2)])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def main():
    print("=== RUNNING CHAR N-GRAM BASELINE (3-5 CHARS) ===")
    df = pd.read_csv("dataset.csv")
    ann_cols = [f'ann_{i}' for i in range(1, 11)]
    df['spam_count'] = df[ann_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['gold_label'] = np.where(df['spam_count'] >= 6, 1, 0)
    
    texts_raw = df['content'].tolist()
    keep_indices = []
    for i in range(len(texts_raw)):
        t1 = str(texts_raw[i]).lower().strip()
        is_dup = False
        for idx in keep_indices:
            t2 = str(texts_raw[idx]).lower().strip()
            if get_jaccard_sim(t1, t2) >= 0.85:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(i)
            
    clean_df = df.iloc[keep_indices].sample(frac=1, random_state=42).reset_index(drop=True)
    raw_texts = np.array([str(t).lower() for t in clean_df['content']])
    y = clean_df['gold_label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    f1_scores, acc_scores, prec_scores, rec_scores = [], [], [], []
    
    for train_idx, test_idx in skf.split(raw_texts, y):
        X_tr, X_te = raw_texts[train_idx], raw_texts[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000)
        X_train = vec.fit_transform(X_tr)
        X_test = vec.transform(X_te)
        
        clf = SVC(kernel='linear', C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        acc_scores.append(accuracy_score(y_test, preds))
        prec_scores.append(precision_score(y_test, preds, zero_division=0))
        rec_scores.append(recall_score(y_test, preds, zero_division=0))
        f1_scores.append(f1_score(y_test, preds, zero_division=0))
        
    results = {
        'Char_Ngram_TFIDF_SVM': {
            'Accuracy': f"{np.mean(acc_scores)*100:.2f}% ± {np.std(acc_scores)*100:.2f}%",
            'Precision': f"{np.mean(prec_scores)*100:.2f}% ± {np.std(prec_scores)*100:.2f}%",
            'Recall': f"{np.mean(rec_scores)*100:.2f}% ± {np.std(rec_scores)*100:.2f}%",
            'F1_Score': f"{np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%"
        }
    }
    
    print("\n=== CHAR N-GRAM RESULTS SUMMARY ===")
    print(json.dumps(results, indent=2))
    
    with open("scratch/char_ngram_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
