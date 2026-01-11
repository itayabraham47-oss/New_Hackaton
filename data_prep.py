import pandas as pd
import scanpy as sc
import numpy as np
import os


def get_markers_from_sc(sc_file_path, cell_type_column='cell_type', n_genes=100):
    """
    שלב 1 (צוות א'): טעינת נתוני Single Cell ומציאת גנים מרקרים.
    """
    print(f"Loading Single Cell data from {sc_file_path}...")
    try:
        adata = sc.read_h5ad(sc_file_path)
    except Exception as e:
        print(f"Error loading h5ad file: {e}")
        return {}

    # וידוא נרמול (חשוב לחישוב סטטיסטי תקין)
    # אם הדאטה מגיע כ-Raw Counts, צריך לנרמל. אם הוא כבר מעובד, אפשר לדלג.
    # כאן נניח שצריך לנרמל בסיסי:
    if np.max(adata.X) > 100:  # אינדיקציה גסה שזה Counts ולא Log-transformed
        print("Normalizing Single Cell data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    print(f"Identifying marker genes using Wilcoxon Rank-Sum test...")
    # חישוב הגנים המבדילים ביותר עבור כל קבוצה בעמודת cell_type_column
    sc.tl.rank_genes_groups(adata, groupby=cell_type_column, method='wilcoxon')

    # שליפת התוצאות למילון פייתון נקי
    markers_dict = {}
    groups = adata.obs[cell_type_column].unique()

    for group in groups:
        # שליפת שמות הגנים עם הציון הגבוה ביותר לקבוצה זו
        # הגישה למבנה הנתונים של scanpy מעט מורכבת:
        gene_names = adata.uns['rank_genes_groups']['names'][group]

        # לוקחים רק את ה-N הראשונים
        top_genes = gene_names[:n_genes]
        markers_dict[group] = list(top_genes)

    print(f"Found markers for {len(markers_dict)} cell types.")
    return markers_dict


def process_bulk_data(bulk_counts_path, phenotype_path, markers_dict):
    """
    שלב 2 (צוות ב'): טעינת TCGA, סינון דגימות וסינון גנים.
    """
    print("Loading TCGA Bulk Data...")

    # 1. טעינת הנתונים
    # ה-Index צריך להיות שמות גנים (Gene Symbols) כדי להתאים למילון המרקרים
    df_counts = pd.read_csv(bulk_counts_path, index_col=0)
    df_pheno = pd.read_csv(phenotype_path, index_col=0)  # מניחים שאינדקס הוא שם הדגימה

    # 2. סינון דגימות: שמירה על Primary Tumor בלבד
    # ב-UCSC Xena, העמודה לרוב נקראת 'sample_type' או 'sample_type_id'
    # Primary Tumor מסומן לעיתים כ- 'Primary Tumor' או קוד 01

    print("Filtering samples (Keeping only Primary Tumor)...")

    # בדיקה מה שם העמודה המדויק בקובץ שהורדתם (יתכן שינוי קל בשם)
    target_col = 'sample_type'
    if target_col not in df_pheno.columns:
        # ניסיון ניחוש אם השם שונה
        possible_cols = [c for c in df_pheno.columns if 'type' in c.lower()]
        if possible_cols:
            target_col = possible_cols[0]
            print(f"Warning: Assuming '{target_col}' contains sample type info.")

    try:
        # הסינון עצמו
        tumor_samples = df_pheno[df_pheno[target_col] == 'Primary Tumor'].index

        # חיתוך: משאירים ב-Counts רק את העמודות שמופיעות ברשימת הגידולים
        # וגם קיימות בפועל בקובץ ה-Counts
        valid_samples = [s for s in tumor_samples if s in df_counts.columns]
        df_tumor = df_counts[valid_samples]

        print(f"Filtered samples: {df_counts.shape[1]} -> {df_tumor.shape[1]}")

    except KeyError:
        print("Could not filter by phenotype. Using all samples (Check column names!).")
        df_tumor = df_counts

    # 3. התאמת מרחב הגנים (Gene Filtering)
    # איחוד כל המרקרים לרשימה אחת שטוחה
    all_markers = set()
    for genes in markers_dict.values():
        all_markers.update(genes)

    print(f"Filtering genes based on {len(all_markers)} unique markers...")

    # בדיקת חפיפה: אילו מרקרים באמת קיימים בדאטה של ה-Bulk?
    # הערה: זה המקום בו דברים נופלים אם ב-Bulk יש Ensembl ID וב-SC יש שמות גנים
    common_genes = [g for g in all_markers if g in df_tumor.index]

    if len(common_genes) == 0:
        print("CRITICAL ERROR: No intersection between Bulk gene IDs and Marker gene names.")
        print("Check if one file uses Ensembl IDs (ENSG...) and the other uses Gene Symbols (TP53...)")
        return None

    # יצירת הדאטה הסופי
    df_final = df_tumor.loc[common_genes]

    print(f"Final Data Shape: {df_final.shape}")
    return df_final


# ==========================================
# Main Execution Block (ככה מריצים את זה)
# ==========================================
if __name__ == "__main__":
    # נתיבים לקבצים (הניחו שהם בתיקיית data)
    sc_path = "data/sc/lung_sc.h5ad"
    bulk_path = "data/bulk/tcga_luad_counts.csv"
    pheno_path = "data/bulk/tcga_luad_survival.csv"  # לעיתים המידע נמצא כאן

    # בדיקה שהקבצים קיימים לפני שמתחילים
    if os.path.exists(sc_path) and os.path.exists(bulk_path):

        # 1. הפקת המרקרים (צוות א')
        markers = get_markers_from_sc(sc_path, cell_type_column='cell_type', n_genes=50)

        # הדפסת דוגמה למילון
        print("\nExample Markers (T-cells):", markers.get('T-cells', [])[:5])

        # 2. עיבוד ה-Bulk (צוות ב')
        final_bulk_matrix = process_bulk_data(bulk_counts_path=bulk_path,
                                              phenotype_path=pheno_path,
                                              markers_dict=markers)

        if final_bulk_matrix is not None:
            # שמירת התוצאות המוכנות לאלגוריתם ה-NMF
            final_bulk_matrix.to_csv("data/processed_bulk_for_nmf.csv")
            print("\nDone! Processed data saved to 'data/processed_bulk_for_nmf.csv'")

            # כאן אפשר לקרוא לפונקציה מהשלב הקודם:
            # H, W = run_seeded_nmf(final_bulk_matrix, markers)

    else:
        print("Please make sure input files exist in the 'data' folder.")