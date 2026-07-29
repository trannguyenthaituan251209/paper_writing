import pandas as pd
import numpy as np
import json
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from pyvi import ViTokenizer

def get_jaccard_sim(str1, str2):
    set1 = set([str1[i:i+3] for i in range(len(str1)-2)])
    set2 = set([str2[i:i+3] for i in range(len(str2)-2)])
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def main():
    print("=== RUNNING STATISTICAL SIGNIFICANCE TESTS (MCNEMAR & WILCOXON) ===")
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
    clean_texts = np.array([ViTokenizer.tokenize(str(t).lower()) for t in clean_df['content']])
    y_gold = clean_df['gold_label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Linear_SVM': lambda: SVC(kernel='linear', C=1.0, random_state=42),
        'RBF_SVM': lambda: SVC(kernel='rbf', C=1.0, random_state=42),
        'Logistic_Regression': lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'MLP': lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'Naive_Bayes': lambda: MultinomialNB(alpha=1.0)
    }
    
    oof_preds = {}
    fold_f1s = {}
    
    for name, model_fn in models.items():
        oof = np.zeros(len(clean_df), dtype=int)
        f1s = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(clean_texts, y_gold)):
            X_tr, X_te = clean_texts[train_idx], clean_texts[test_idx]
            y_train, y_test = y_gold[train_idx], y_gold[test_idx]
            
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            X_train = vec.fit_transform(X_tr)
            X_test = vec.transform(X_te)
            
            clf = model_fn()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            oof[test_idx] = preds
            from sklearn.metrics import f1_score
            f1s.append(f1_score(y_test, preds, zero_division=0))
            
        oof_preds[name] = oof
        fold_f1s[name] = f1s
        
    pairs = [
        ('Linear_SVM', 'Naive_Bayes'),
        ('RBF_SVM', 'Naive_Bayes'),
        ('Linear_SVM', 'RBF_SVM'),
        ('Linear_SVM', 'Logistic_Regression')
    ]
    
    sig_results = {}
    
    for m1, m2 in pairs:
        p1 = oof_preds[m1]
        p2 = oof_preds[m2]
        
        c1 = (p1 == y_gold)
        c2 = (p2 == y_gold)
        
        n10 = np.sum(c1 & ~c2)
        n01 = np.sum(~c1 & c2)
        
        table = [[np.sum(c1 & c2), int(n10)],
                 [int(n01), np.sum(~c1 & ~c2)]]
        
        mcnemar_res = mcnemar(table, exact=True)
        w_stat, w_p = wilcoxon(fold_f1s[m1], fold_f1s[m2])
        
        sig_results[f"{m1} vs {m2}"] = {
            'McNemar_n10 (Model1 right, Model2 wrong)': int(n10),
            'McNemar_n01 (Model1 wrong, Model2 right)': int(n01),
            'McNemar_p_value': float(mcnemar_res.pvalue),
            'McNemar_Significant (alpha=0.05)': bool(mcnemar_res.pvalue < 0.05),
            'Wilcoxon_p_value': float(w_p),
            'Wilcoxon_Significant (alpha=0.05)': bool(w_p < 0.05)
        }
        
    print("\n=== SIGNIFICANCE TEST RESULTS SUMMARY ===")
    print(json.dumps(sig_results, indent=2))
    
    with open("scratch/significance_test_results.json", "w", encoding="utf-8") as f:
        json.dump(sig_results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
