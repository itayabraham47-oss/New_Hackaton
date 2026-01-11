import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF


# ==========================================
# 1. פונקציות עזר (מהקוד שלכם)
# ==========================================
def load_data_and_markers():
    data_dir = "data"
    tsv_path = f"{data_dir}/sc/NSCLC_GSE99254_AllDiffGenes_table.tsv"
    bulk_path = f"{data_dir}/bulk/TCGA-LUAD.HiSeqV2.gz"

    # טעינת מרקרים
    if not os.path.exists(tsv_path): return None, None
    df_markers = pd.read_csv(tsv_path, sep='\t')
    cell_col = [c for c in df_markers.columns if 'lineage' in c or 'type' in c][0]

    markers_dict = {}
    for ct, grp in df_markers.groupby(cell_col):
        if 'log2FC' in grp.columns: grp = grp.sort_values('log2FC', ascending=False)
        clean_name = str(ct).replace(' ', '_').replace('/', '_').replace('+', '')
        markers_dict[clean_name] = grp['Gene'].head(50).tolist()

    # טעינת Bulk
    if not os.path.exists(bulk_path): return None, None
    df_bulk = pd.read_csv(bulk_path, sep='\t', index_col=0, compression='gzip')
    df_bulk.index = df_bulk.index.str.upper().str.split('|').str[0]

    # חיתוך גנים
    all_markers = set([str(g).upper() for sublist in markers_dict.values() for g in sublist])
    # חשוב: אנו ממיינים את הגנים לפי הסדר של המרקרים כדי שהגרף יצא יפה (באלכסון)
    ordered_genes = []
    for ct in markers_dict.keys():
        ordered_genes.extend([g.upper() for g in markers_dict[ct]])

    # מסננים רק מה שקיים ב-Bulk
    common = [g for g in ordered_genes if g in df_bulk.index]
    # הסרת כפילויות תוך שמירה על סדר
    seen = set()
    common = [x for x in common if not (x in seen or seen.add(x))]

    df_filtered = df_bulk.loc[common].fillna(0)
    df_filtered[df_filtered < 0] = 0

    return df_filtered, markers_dict


# ==========================================
# 2. השוואה ויזואלית ומתמטית
# ==========================================
def compare_W_matrices():
    print("--- Loading Data... ---")
    V, markers_dict = load_data_and_markers()
    if V is None:
        print("Error: Files not found.")
        return

    cell_types = list(markers_dict.keys())
    n_components = len(cell_types)
    n_genes, n_samples = V.shape

    print(f"Analyzing {n_genes} genes across {n_samples} samples.")

    # --- מודל 1: Seeded (שלנו) ---
    print("Running Seeded NMF...")
    # בניית W התחלתי
    W_init = pd.DataFrame(0.1, index=V.index, columns=cell_types)
    for ct, genes in markers_dict.items():
        for g in genes:
            g_upper = str(g).upper()
            if g_upper in W_init.index: W_init.loc[g_upper, ct] = 10.0

    model_seeded = NMF(n_components=n_components, init='custom', solver='cd', max_iter=4000, random_state=42)
    W_seeded = model_seeded.fit_transform(V, W=W_init.values, H=np.random.rand(n_components, n_samples))
    H_seeded = model_seeded.components_

    # חישוב שגיאת השחזור (Reconstruction Error): ||V - WH||
    WH_seeded = np.dot(W_seeded, H_seeded)
    error_seeded = np.linalg.norm(V - WH_seeded)

    # --- מודל 2: Random (רגיל) ---
    print("Running Random NMF...")
    model_random = NMF(n_components=n_components, init='random', solver='cd', max_iter=4000, random_state=42)
    W_random = model_random.fit_transform(V)
    H_random = model_random.components_

    WH_random = np.dot(W_random, H_random)
    error_random = np.linalg.norm(V - WH_random)

    # --- ויזואליזציה ---
    print("Generating Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # פונקציית נרמול להצגה יפה
    def normalize(mat):
        return mat / mat.max(axis=0)

    # גרף 1: Seeded W
    sns.heatmap(normalize(W_seeded), ax=axes[0], cmap="viridis", yticklabels=False)
    axes[0].set_title(f"Seeded Model (W)\nBiologically Structured\nError: {error_seeded:.0f}", fontsize=14,
                      fontweight='bold', color='green')
    axes[0].set_xlabel("Cell Types")
    axes[0].set_ylabel("Genes (Sorted by Cell Type)")

    # גרף 2: Random W
    # ננסה לסדר את העמודות של הרנדומלי שייראו הכי דומות לסידד (כדי להיות הוגנים)
    # אבל זה עדיין ייראה מבולגן
    sns.heatmap(normalize(W_random), ax=axes[1], cmap="viridis", yticklabels=False)
    axes[1].set_title(f"Random Model (W)\nMathematically Optimized but Messy\nError: {error_random:.0f}", fontsize=14,
                      color='green')
    axes[1].set_xlabel("Latent Components")
    axes[1].set_ylabel("Genes")

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/matrix_comparison.png")
    print("Saved comparison to: results/plots/matrix_comparison.png")

    print("\n--- Summary ---")
    print(f"Seeded Error: {error_seeded:.2f}")
    print(f"Random Error: {error_random:.2f}")
    print("Note: Random NMF often has LOWER error because it's purely mathematical.")
    print("But Seeded NMF has STRUCTURE, which allows us to identify cell types correctly.")

    plt.show()


if __name__ == "__main__":
    compare_W_matrices()