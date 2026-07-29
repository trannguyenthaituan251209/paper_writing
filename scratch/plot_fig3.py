import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== GENERATING FIGURE 3 FROM DATASET.CSV (MATPLOTLIB) ===")
    df = pd.read_csv("dataset.csv")
    ann_cols = [f'ann_{i}' for i in range(1, 11)]
    
    # Count spam votes per message
    df['spam_count'] = (df[ann_cols] == 'SPAM').sum(axis=1)
    
    # Map consensus levels (10 out of 10 down to 6 out of 10)
    levels = ['10 out of 10', '9 out of 10', '8 out of 10', '7 out of 10', '6 out of 10']
    
    ham_counts = []
    spam_counts = []
    
    # Level 10/10: spam_count == 0 (Ham) or 10 (Spam)
    ham_counts.append((df['spam_count'] == 0).sum())
    spam_counts.append((df['spam_count'] == 10).sum())
    
    # Level 9/10: spam_count == 1 (Ham) or 9 (Spam)
    ham_counts.append((df['spam_count'] == 1).sum())
    spam_counts.append((df['spam_count'] == 9).sum())
    
    # Level 8/10: spam_count == 2 (Ham) or 8 (Spam)
    ham_counts.append((df['spam_count'] == 2).sum())
    spam_counts.append((df['spam_count'] == 8).sum())
    
    # Level 7/10: spam_count == 3 (Ham) or 7 (Spam)
    ham_counts.append((df['spam_count'] == 3).sum())
    spam_counts.append((df['spam_count'] == 7).sum())
    
    # Level 6/10: spam_count == 4 (Ham) or 6 (Spam)
    ham_counts.append((df['spam_count'] == 4).sum())
    spam_counts.append((df['spam_count'] == 6).sum())
    
    # Plotting using clean matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(levels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, ham_counts, width, label='HAM', color='#ed7d31')
    rects2 = ax.bar(x + width/2, spam_counts, width, label='SPAM', color='#4472c4')
    
    ax.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Inter-Annotator Agreement Levels before Disagreement Resolution', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig("Fig3.png", dpi=300)
    print("Figure 3 successfully generated and saved to Fig3.png!")

if __name__ == '__main__':
    main()
