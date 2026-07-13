import pandas as pd
import io
import ast
from itertools import combinations
from scipy.stats import wilcoxon
import numpy as np

csv_data = """put here the failure results data"""

df = pd.read_csv(io.StringIO(csv_data))
df['f1_0.5 values'] = df['f1_0.5 values'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Vargha-Delaney effect size
def vargha_delaney(A, B):
    n1, n2 = len(A), len(B)
    ranks = pd.Series(A + B).rank()
    R1 = sum(ranks[:n1])
    A_stat = (R1/n1 - (n1+1)/2) / n2
    return A_stat

def pairwise_stats(df_subset, filename):
    results = []
    for m1, m2 in combinations(df_subset['model'].unique(), 2):
        f1_1 = df_subset[df_subset['model']==m1]['f1_0.5 values'].values[0]
        f1_2 = df_subset[df_subset['model']==m2]['f1_0.5 values'].values[0]
        try:
            _, p = wilcoxon(f1_1, f1_2)
        except ValueError:
            p = np.nan
        A = vargha_delaney(f1_1, f1_2)
        results.append([m1, m2, p, A])
    results_df = pd.DataFrame(results, columns=['Model1', 'Model2', 'Wilcoxon_p', 'Vargha_A'])
    results_df.to_csv(filename, index=False)

binary_df = df[df['technique']=='binary']
pairwise_stats(binary_df, "f1_binary_pairwise.csv")

continuous_df = df[df['technique']=='continuous']
pairwise_stats(continuous_df, "f1_continuous_pairwise.csv")

binary_df.to_csv("./binary_judge_stat.csv")
continuous_df.to_csv("./cont_judge_stat.csv")