import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF


# ==========================================
# 1. טעינת מרקרים
# ==========================================
def get_markers_from_tsv(tsv_path, n_genes=50):
    print(f"--- Loading Markers from: {os.path.basename(tsv_path)} ---")
    if not os.path.exists(tsv_path):
        print(f"ERROR: File not found: {tsv_path}")
        return None

    df = pd.read_csv(tsv_path, sep='\t')

    cell_type_col = 'Celltype (major-lineage)'
    if cell_type_col not in df.columns:
        possible = [c for c in df.columns if 'lineage' in c or 'type' in c]
        if possible:
            cell_type_col = possible[0]
        else:
            print("ERROR: Could not find cell type column in TSV.")
            return None

    markers_dict = {}
    for cell_type, group_df in df.groupby(cell_type_col):
        if 'log2FC' in group_df.columns:
            group_df = group_df.sort_values(by='log2FC', ascending=False)

        top_genes = group_df['Gene'].head(n_genes).tolist()
        clean_name = str(cell_type).replace(' ', '_').replace('/', '_').replace('+', '')
        markers_dict[clean_name] = top_genes

    print(f"   Loaded {len(markers_dict)} cell types.")
    return markers_dict


# ==========================================
# 2. יצירת מטריצת W
# ==========================================
def create_W_matrix(bulk_genes, marker_dict):
    cell_types = list(marker_dict.keys())
    W = pd.DataFrame(0.1, index=bulk_genes, columns=cell_types)

    hits = 0
    for ct, genes in marker_dict.items():
        for gene in genes:
            g_upper = str(gene).upper()
            if g_upper in W.index:
                W.loc[g_upper, ct] = 10.0
                hits += 1
    print(f"   Seeding matched {hits} marker genes in the bulk data.")
    return W, cell_types


# ==========================================
# Main Pipeline
# ==========================================
def main():
    data_dir = "data"
    tsv_path = f"{data_dir}/sc/NSCLC_GSE99254_AllDiffGenes_table.tsv"
    bulk_path = f"{data_dir}/bulk/TCGA-LUAD.HiSeqV2.gz"
    surv_path = f"{data_dir}/bulk/TCGA-LUAD.survival.tsv.gz"

    os.makedirs("results/plots", exist_ok=True)

    marker_dict = get_markers_from_tsv(tsv_path, n_genes=50)
    if not marker_dict: return

    print("--- Loading Bulk Data ---")
    if not os.path.exists(bulk_path):
        print("Bulk file not found!")
        return

    df_bulk = pd.read_csv(bulk_path, sep='\t', index_col=0, compression='gzip')
    df_bulk.index = df_bulk.index.str.upper().str.split('|').str[0]

    all_markers = set([str(g).upper() for sublist in marker_dict.values() for g in sublist])
    common = list(all_markers.intersection(set(df_bulk.index)))
    print(f"Using {len(common)} common genes for analysis.")

    if len(common) < 10:
        print("Too few genes found.")
        return

    df_filtered = df_bulk.loc[common].fillna(0)
    df_filtered[df_filtered < 0] = 0

    print("--- Running NMF ---")
    W_init_df, cell_types = create_W_matrix(df_filtered.index, marker_dict)

    n_samples = df_filtered.shape[1]
    n_components = len(cell_types)
    H_init = np.random.rand(n_components, n_samples)

    model = NMF(n_components=n_components, init='custom', solver='cd', max_iter=4000, random_state=42)
    W_fitted = model.fit_transform(df_filtered, W=W_init_df.values, H=H_init)
    H_matrix = model.components_

    df_H = pd.DataFrame(H_matrix, index=cell_types, columns=df_filtered.columns)
    df_proportions = df_H.div(df_H.sum(axis=0), axis=1)

    df_proportions.to_csv("results/cell_proportions.csv")
    print("Results saved to results/cell_proportions.csv")

    print("--- Generating Plots ---")
    plt.figure(figsize=(12, 6))
    df_proportions.iloc[:, :30].T.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.title("Cell Type Proportions (First 30 Patients)")
    plt.xlabel("Patient Sample")
    plt.ylabel("Proportion")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/plots/proportions_barplot.png")
    print("Barplot saved.")
    plt.close()


if __name__ == "__main__":
    main()