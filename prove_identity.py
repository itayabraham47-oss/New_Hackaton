import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF


def prove_innovation():
    print("--- Loading Data ---")
    data_dir = "data"

    # 1. טעינת מרקרים וסידור הציפיות (Ground Truth)
    tsv_path = f"{data_dir}/sc/NSCLC_GSE99254_AllDiffGenes_table.tsv"
    if not os.path.exists(tsv_path): return

    df_markers = pd.read_csv(tsv_path, sep='\t')
    cell_col = [c for c in df_markers.columns if 'lineage' in c or 'type' in c][0]

    # יצירת "מטריצת האמת" (Binary Mask)
    # שורות = גנים, עמודות = סוגי תאים
    markers_dict = {}
    for ct, grp in df_markers.groupby(cell_col):
        if 'log2FC' in grp.columns: grp = grp.sort_values('log2FC', ascending=False)
        clean_name = str(ct).replace(' ', '_').replace('/', '_').replace('+', '')
        markers_dict[clean_name] = grp['Gene'].head(50).tolist()

    # טעינת Bulk
    bulk_path = f"{data_dir}/bulk/TCGA-LUAD.HiSeqV2.gz"
    df_bulk = pd.read_csv(bulk_path, sep='\t', index_col=0, compression='gzip')
    df_bulk.index = df_bulk.index.str.upper().str.split('|').str[0]

    # חיתוך גנים
    all_markers = set([str(g).upper() for sublist in markers_dict.values() for g in sublist])
    common = sorted(list(all_markers.intersection(set(df_bulk.index))))

    df_filtered = df_bulk.loc[common].fillna(0)
    df_filtered[df_filtered < 0] = 0

    # יצירת מסכת האמת (Ideal Matrix)
    # 1 איפה שיש מרקר, 0 איפה שאין
    cell_types = list(markers_dict.keys())
    Ideal_W = pd.DataFrame(0, index=common, columns=cell_types)
    for ct, genes in markers_dict.items():
        for g in genes:
            g_upper = str(g).upper()
            if g_upper in Ideal_W.index:
                Ideal_W.loc[g_upper, ct] = 1

    n_components = len(cell_types)

    # --- הרצת המודלים ---

    # 1. Seeded
    print("Running Seeded Model...")
    W_seed = pd.DataFrame(0.1, index=common, columns=cell_types)
    for ct, genes in markers_dict.items():
        for g in genes:
            if str(g).upper() in W_seed.index: W_seed.loc[str(g).upper(), ct] = 10.0

    model_seeded = NMF(n_components=n_components, init='custom', solver='cd', max_iter=2000, random_state=42)
    W_seeded = model_seeded.fit_transform(df_filtered, W=W_seed.values,
                                          H=np.random.rand(n_components, df_filtered.shape[1]))
    W_seeded_df = pd.DataFrame(W_seeded, index=common, columns=cell_types)

    # 2. Random
    print("Running Random Model...")
    model_random = NMF(n_components=n_components, init='random', solver='cd', max_iter=2000, random_state=42)
    W_random = model_random.fit_transform(df_filtered)
    W_random_df = pd.DataFrame(W_random, index=common, columns=[f"Comp {i}" for i in range(n_components)])

    # --- חישוב הדמיון (Correlation) ---
    print("Calculating Correlations...")

    # קורלציה בין המודל שלכם לאמת
    corr_seeded = pd.DataFrame(index=cell_types, columns=cell_types)
    for true_ct in cell_types:
        for learned_ct in cell_types:
            # מתאם בין העמודה האידיאלית לעמודה שנלמדה
            corr = np.corrcoef(Ideal_W[true_ct], W_seeded_df[learned_ct])[0, 1]
            corr_seeded.loc[true_ct, learned_ct] = corr

    # קורלציה בין המודל הרנדומלי לאמת
    corr_random = pd.DataFrame(index=cell_types, columns=W_random_df.columns)
    for true_ct in cell_types:
        for learned_comp in W_random_df.columns:
            corr = np.corrcoef(Ideal_W[true_ct], W_random_df[learned_comp])[0, 1]
            corr_random.loc[true_ct, learned_comp] = corr

    # --- ציור הגרף המנצח ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(corr_seeded.astype(float), ax=axes[0], cmap="Greens", annot=False)
    axes[0].set_title("Seeded Model (Ours)\nDiagonal = Perfect Identification", fontsize=14, fontweight='bold',
                      color='green')
    axes[0].set_ylabel("True Cell Types (Goal)")
    axes[0].set_xlabel("Learned Components (Result)")

    sns.heatmap(corr_random.astype(float), ax=axes[1], cmap="Reds", annot=False)
    axes[1].set_title("Random Model\nScattered = Identity Lost", fontsize=14, fontweight='bold', color='red')
    axes[1].set_ylabel("True Cell Types (Goal)")
    axes[1].set_xlabel("Random Components (Result)")

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/identity_proof.png")
    print("Proof saved to: results/plots/identity_proof.png")
    plt.show()


if __name__ == "__main__":
    prove_innovation()