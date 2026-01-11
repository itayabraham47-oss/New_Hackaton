import pandas as pd
import numpy as np
import os
from sklearn.decomposition import NMF


# --- 1. Marker Loader (Strictly from Table) ---

def get_markers_from_tsv(tsv_path, n_genes=50):
    """
    Loads markers STRICTLY from the TISCH TSV table.
    No hardcoded fallbacks.
    """
    print(f"\n--- Loading Markers from Table: {os.path.basename(tsv_path)} ---")

    if not os.path.exists(tsv_path):
        print(f"CRITICAL ERROR: File not found at {tsv_path}")
        print("Please make sure the file 'NSCLC_GSE99254_AllDiffGenes_table.tsv' is in 'data/sc/' folder.")
        return None

    try:
        # Load the TSV file
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"   Table loaded successfully. Total rows: {len(df)}")
    except Exception as e:
        print(f"   Error reading TSV file: {e}")
        return None

    # --- Column Identification ---
    # Based on the file you uploaded, these are the exact columns:
    # 'Celltype (major-lineage)', 'Gene', 'log2FC'

    cell_type_col = 'Celltype (major-lineage)'
    gene_col = 'Gene'
    fc_col = 'log2FC'

    # Validation
    if cell_type_col not in df.columns:
        print(f"   ERROR: Column '{cell_type_col}' not found.")
        print(f"   Available columns: {df.columns.tolist()}")
        # Try to guess if the name is slightly different
        possible = [c for c in df.columns if 'major' in c or 'lineage' in c]
        if possible:
            cell_type_col = possible[0]
            print(f"   -> Found alternative column: '{cell_type_col}'")
        else:
            return None

    print(f"   Extracting markers based on: '{cell_type_col}' sorted by '{fc_col}'")

    # --- Extraction Logic ---
    markers_dict = {}
    grouped = df.groupby(cell_type_col)

    for cell_type, group_df in grouped:
        # 1. Sort by log2FC descending (strongest markers first)
        group_df = group_df.sort_values(by=fc_col, ascending=False)

        # 2. Select top N genes
        top_genes = group_df[gene_col].head(n_genes).tolist()

        # 3. Clean the cell type name (e.g., "CD4Tconv" -> "CD4Tconv")
        # Removing special chars usually helps plotting later
        clean_name = str(cell_type).replace(' ', '_').replace('/', '_')

        markers_dict[clean_name] = top_genes

    print(f"   Success! Learned {len(markers_dict)} cell types from table:")
    print(f"   {list(markers_dict.keys())}")
    return markers_dict


# --- 2. Matrix Construction ---

def create_W_matrix(bulk_genes, marker_dict):
    cell_types = list(marker_dict.keys())
    # Initialize W with a small base value (0.1)
    W = pd.DataFrame(0.1, index=bulk_genes, columns=cell_types)

    count_hits = 0
    total_markers = sum(len(genes) for genes in marker_dict.values())

    for ct, genes in marker_dict.items():
        for gene in genes:
            # We map the gene name to Upper Case to match TCGA format
            gene_upper = str(gene).upper()
            if gene_upper in W.index:
                W.loc[gene_upper, ct] = 10.0  # High weight (Seeding)
                count_hits += 1

    print(f"   (Seeding Stats: Used {count_hits} out of {total_markers} markers found in Bulk data)")
    return W, cell_types


# --- 3. Main Execution ---

def main():
    # --- Step A: Load Bulk Data ---
    print("--- Step 1: Loading Bulk Data ---")
    data_dir = "data/bulk"

    # Search for the bulk file
    files = [f for f in os.listdir(data_dir) if 'HiSeqV2' in f or 'counts' in f]
    if not files:
        print("ERROR: No bulk data file found in data/bulk. Please download it first.")
        return

    bulk_file = files[0]
    bulk_path = os.path.join(data_dir, bulk_file)
    print(f"Loading Bulk file: {bulk_file}")

    try:
        # Load TCGA data
        df_bulk = pd.read_csv(bulk_path, sep='\t', index_col=0,
                              compression='gzip' if bulk_file.endswith('gz') else None)
        # Standardize Gene Names to Upper Case
        df_bulk.index = df_bulk.index.str.upper().str.split('|').str[0]
        print(f"Bulk Data loaded. Shape: {df_bulk.shape}")
    except Exception as e:
        print(f"Critical Error loading bulk file: {e}")
        return

    # --- Step B: Load Markers (STRICT) ---
    print("\n--- Step 2: Learning Markers from Table ---")

    # Path to your uploaded table
    tsv_path = "data/sc/NSCLC_GSE99254_AllDiffGenes_table.tsv"

    marker_dict = get_markers_from_tsv(tsv_path, n_genes=50)

    if marker_dict is None:
        print("\n!!! STOPPING: Could not load markers from the table. !!!")
        print("Please fix the file path or download the correct TSV.")
        return  # We stop here. No fallbacks.

    # --- Step C: Intersect Genes ---
    print("\n--- Step 3: Intersecting Data ---")

    # Flatten the marker list
    all_markers = set([str(g).upper() for sublist in marker_dict.values() for g in sublist])
    bulk_genes = set(df_bulk.index)

    # Find common genes
    common_genes = list(all_markers.intersection(bulk_genes))

    print(f"Found {len(common_genes)} unique marker genes present in the Bulk dataset.")

    if len(common_genes) < 10:
        print("ERROR: Intersection is too low (<10 genes). Something is wrong with gene names.")
        print(f"Example Marker: {list(all_markers)[0]}")
        print(f"Example Bulk Gene: {list(bulk_genes)[0]}")
        return

    # Filter Bulk Matrix
    df_filtered = df_bulk.loc[common_genes].fillna(0)
    df_filtered[df_filtered < 0] = 0  # NMF requires non-negative

    # --- Step D: Run Seeded NMF ---
    print("\n--- Step 4: Running Seeded NMF ---")

    W_init_df, cell_types = create_W_matrix(df_filtered.index, marker_dict)

    n_samples = df_filtered.shape[1]
    n_components = len(cell_types)

    # Random H initialization
    H_init = np.random.rand(n_components, n_samples)

    print(f"Deconvoluting into {n_components} cell types: {cell_types}")

    model = NMF(n_components=n_components,
                init='custom',
                solver='cd',
                beta_loss='frobenius',
                max_iter=5000,
                random_state=42)

    # Convert W to numpy array (.values) to avoid sklearn errors
    W_fitted = model.fit_transform(df_filtered, W=W_init_df.values, H=H_init)
    H_matrix = model.components_

    # --- Step E: Save Results ---
    print("\n--- Step 5: Saving Results ---")

    df_H = pd.DataFrame(H_matrix, index=cell_types, columns=df_filtered.columns)

    # Normalize (Proportions sum to 1)
    df_proportions = df_H.div(df_H.sum(axis=0), axis=1).T

    os.makedirs("results", exist_ok=True)
    out_path = "results/cell_proportions.csv"
    df_proportions.to_csv(out_path)

    print(f"SUCCESS! Proportions saved to: {out_path}")
    print("\nTop 5 rows:")
    print(df_proportions.head())


if __name__ == "__main__":
    main()