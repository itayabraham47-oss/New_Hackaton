import pandas as pd
import numpy as np
import os
import scanpy as sc
from sklearn.decomposition import NMF


# --- 1. Logic: Get Markers from Single Cell Data ---

def get_markers_from_sc(sc_file_path, cell_type_column='cell_type', n_genes=50):
    """
    Step 1 (Team A): Load Single Cell data and find marker genes dynamically.
    """
    print(f"\n--- Loading Single Cell Data from: {sc_file_path} ---")

    if not os.path.exists(sc_file_path):
        print(f"WARNING: Single Cell file not found at {sc_file_path}")
        return None

    try:
        adata = sc.read_h5ad(sc_file_path)
        print(f"SC Data Loaded. Shape: {adata.shape}")
    except Exception as e:
        print(f"Error loading h5ad file: {e}")
        return None

    # Normalization Check
    if np.max(adata.X) > 100:
        print("Normalizing Single Cell data (Log1p)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    print(f"Identifying marker genes using Wilcoxon Rank-Sum test...")

    if cell_type_column not in adata.obs.columns:
        print(f"ERROR: Column '{cell_type_column}' not found in Single Cell data.")
        return None

    sc.tl.rank_genes_groups(adata, groupby=cell_type_column, method='wilcoxon')

    markers_dict = {}
    groups = adata.obs[cell_type_column].unique()

    for group in groups:
        gene_names = adata.uns['rank_genes_groups']['names'][group]
        top_genes = gene_names[:n_genes]
        markers_dict[group] = list(top_genes)

    print(f"Successfully generated markers for {len(markers_dict)} cell types.")
    return markers_dict


# --- 2. Fallback Logic ---

def get_hardcoded_signatures_fallback():
    return {
        'T_cells': ['CD3D', 'CD3E', 'CD2', 'CD8A', 'CD4', 'TRBC2'],
        'B_cells': ['CD79A', 'MS4A1', 'CD19', 'CD79B'],
        'Macrophages': ['CD68', 'CD163', 'AIF1', 'MARCO'],
        'Endothelial': ['PECAM1', 'VWF', 'CDH5'],
        'Fibroblasts': ['COL1A1', 'COL1A2', 'DCN'],
        'Epithelial_Tumor': ['EPCAM', 'KRT8', 'KRT18', 'KRT19']
    }


# --- 3. Matrix Construction ---

def create_W_matrix(bulk_genes, marker_dict):
    """
    Builds the seeded W matrix as a DataFrame first for alignment.
    """
    cell_types = list(marker_dict.keys())

    # Initialize with small value
    W = pd.DataFrame(0.1, index=bulk_genes, columns=cell_types)

    # "Seed" the matrix
    count_hits = 0
    for ct, genes in marker_dict.items():
        for gene in genes:
            if gene in W.index:
                W.loc[gene, ct] = 10.0
                count_hits += 1

    print(f"   (Matrix seeding: Matched {count_hits} marker genes in the Bulk data)")
    return W, cell_types


# --- 4. Main Execution ---

def main():
    # --- Step A: Load Bulk Data ---
    print("--- Step 1: Loading Bulk Data ---")

    data_dir = "data/bulk"
    possible_files = [
        "TCGA-LUAD.HiSeqV2.gz",
        "TCGA-LUAD.htseq_counts.tsv.gz",
        "TCGA-LUAD.star_counts.tsv.gz"
    ]

    file_path = None
    for f in possible_files:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            file_path = p
            break

    if not file_path:
        print(f"ERROR: No Bulk data file found in {data_dir}.")
        return

    print(f"Loading Bulk file: {file_path} ...")
    try:
        df_bulk = pd.read_csv(file_path, sep='\t', index_col=0, compression='gzip')
        df_bulk.index = df_bulk.index.str.upper()
        print(f"Bulk Data loaded. Shape: {df_bulk.shape}")
    except Exception as e:
        print(f"CRITICAL ERROR loading bulk file: {e}")
        return

    # --- Step B: Get Markers ---
    print("\n--- Step 2: Getting Markers ---")

    sc_path = "data/sc/lung_sc.h5ad"
    marker_dict = get_markers_from_sc(sc_path, cell_type_column='cell_type', n_genes=50)

    if marker_dict is None:
        print("!!! SC Data missing or failed. Using Fallback Dictionary. !!!")
        marker_dict = get_hardcoded_signatures_fallback()

    # --- Step C: Intersect and Filter ---
    print("\n--- Step 3: Intersecting Genes ---")

    all_markers = [g for sublist in marker_dict.values() for g in sublist]
    genes_to_keep = list(set(all_markers).intersection(set(df_bulk.index)))

    print(f"Found {len(genes_to_keep)} genes shared between SC markers and Bulk data.")

    if len(genes_to_keep) < 5:
        print("ERROR: Too few matching genes.")
        return

    df_filtered = df_bulk.loc[genes_to_keep]
    df_filtered = df_filtered.fillna(0)
    df_filtered[df_filtered < 0] = 0

    # --- Step D: Initialize Seeded NMF ---
    print("\n--- Step 4: Initializing Seeded NMF ---")

    W_init_df, cell_types = create_W_matrix(df_filtered.index, marker_dict)

    # Initialize H matrix randomly
    n_samples = df_filtered.shape[1]
    n_components = len(cell_types)
    H_init = np.random.rand(n_components, n_samples)

    # --- Step E: Run NMF ---
    print(f"Running NMF for {n_components} cell types...")

    model = NMF(n_components=n_components,
                init='custom',
                solver='cd',
                beta_loss='frobenius',
                max_iter=5000,
                random_state=42)

    # FIX: Convert W DataFrame to numpy array using .values
    # Scikit-learn requires numpy arrays for initialization
    W_fitted = model.fit_transform(df_filtered, W=W_init_df.values, H=H_init)
    H_matrix = model.components_

    # --- Step F: Save Results ---
    print("\n--- Step 5: Saving Results ---")

    df_H = pd.DataFrame(H_matrix, index=cell_types, columns=df_filtered.columns)
    # Normalize to percentages
    df_proportions = df_H.div(df_H.sum(axis=0), axis=1).T

    print("Top 5 rows of proportions:")
    print(df_proportions.head())

    os.makedirs("results", exist_ok=True)
    df_proportions.to_csv("results/cell_proportions.csv")
    print("\nSUCCESS! Saved to 'results/cell_proportions.csv'")


if __name__ == "__main__":
    main()