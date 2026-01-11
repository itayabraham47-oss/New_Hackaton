import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.stats import pearsonr


# ==========================================
# 1. יצירת דאטה סינתטי (In-Silico Generator)
# ==========================================
def create_synthetic_data(markers_dict, all_genes, n_samples=100):
    """
    יוצר חולים מדומים (Synthetic Bulk) עם תשובות ידועות מראש.
    """
    print(f"Generating {n_samples} synthetic patients using {len(all_genes)} genes...")
    cell_types = list(markers_dict.keys())
    n_types = len(cell_types)

    # 1. יצירת פרופורציות אמת (Ground Truth) - אקראי אבל ידוע לנו
    # שימוש בהתפלגות דיריכלה (Dirichlet) שמדמה הרכב תאים אמיתי (סכום=1)
    true_H = np.random.dirichlet(np.ones(n_types), size=n_samples).T
    true_H_df = pd.DataFrame(true_H, index=cell_types, columns=[f"Syn_{i}" for i in range(n_samples)])

    # 2. יצירת מטריצת חתימות (Signature Matrix W)
    # אנחנו מגדירים: כל מרקר מקבל ביטוי גבוה (10) בתא שלו, ונמוך (0.5) באחרים
    W_simulated = pd.DataFrame(0.5, index=all_genes, columns=cell_types)

    for ct, genes in markers_dict.items():
        for gene in genes:
            g_upper = str(gene).upper()
            if g_upper in W_simulated.index:
                # הוספת רעש ביולוגי (כדי שזה לא יהיה קל מדי)
                noise = np.abs(np.random.normal(0, 1.0, 1)[0])
                W_simulated.loc[g_upper, ct] = 10.0 + noise

    # 3. יצירת ה-Bulk (הכפלת מטריצות + רעש מכשיר)
    V_synthetic = W_simulated.dot(true_H_df)

    # הוספת רעש טכני (Measurement Noise)
    noise_matrix = np.random.normal(0, 0.5, V_synthetic.shape)
    V_synthetic = V_synthetic + noise_matrix
    V_synthetic[V_synthetic < 0] = 0  # תיקון ערכים שליליים

    return V_synthetic, true_H_df


# ==========================================
# 2. הרצת המודל לבדיקה
# ==========================================
def run_validation():
    # נתיבים
    data_dir = "data"
    tsv_path = f"{data_dir}/sc/NSCLC_GSE99254_AllDiffGenes_table.tsv"

    if not os.path.exists(tsv_path):
        print(f"Error: Marker file not found at {tsv_path}")
        return

    # --- שלב 1: טעינת המרקרים ---
    print("--- Loading Biological Markers ---")
    df_markers = pd.read_csv(tsv_path, sep='\t')

    # מציאת עמודת סוג התא
    possible_cols = [c for c in df_markers.columns if 'lineage' in c or 'type' in c]
    if not possible_cols:
        print("Error: Could not find cell type column.")
        return
    cell_col = possible_cols[0]

    markers_dict = {}
    for ct, grp in df_markers.groupby(cell_col):
        if 'log2FC' in grp.columns: grp = grp.sort_values('log2FC', ascending=False)
        clean_name = str(ct).replace(' ', '_').replace('/', '_').replace('+', '')
        # לוקחים את ה-50 החזקים
        genes = grp['Gene'].head(50).tolist()
        markers_dict[clean_name] = genes

    # יצירת רשימת כל הגנים (היקום שלנו לסימולציה)
    # תיקון: משתמשים בכל המרקרים כבסיס, בלי תלות בקובץ ה-Bulk החיצוני
    all_genes = set([str(g).upper() for sublist in markers_dict.values() for g in sublist])
    all_genes_list = sorted(list(all_genes))

    if len(all_genes_list) == 0:
        print("Error: No marker genes found.")
        return

    # --- שלב 2: יצירת הולידציה (In-Silico) ---
    print("--- Creating Virtual Patients (Ground Truth) ---")
    V_syn, True_Proportions = create_synthetic_data(markers_dict, all_genes_list, n_samples=100)

    print(f"   Created matrix of shape: {V_syn.shape}")

    # --- שלב 3: הרצת המודל שלך (Seeded NMF) ---
    print("--- Running Your Model on Virtual Patients ---")
    cell_types = list(markers_dict.keys())
    n_components = len(cell_types)

    # בניית מטריצת האתחול (Seeding)
    W_seed = pd.DataFrame(0.1, index=all_genes_list, columns=cell_types)
    for ct, genes in markers_dict.items():
        for g in genes:
            g_upper = str(g).upper()
            if g_upper in W_seed.index: W_seed.loc[g_upper, ct] = 10.0

    # הרצת האלגוריתם
    model = NMF(n_components=n_components, init='custom', solver='cd', max_iter=4000, random_state=42)

    # המודל מקבל את הנתונים המדומים (V_syn) ואת הזרעים (W_seed)
    # ומנסה לגלות לבד את הפרופורציות
    W_pred = model.fit_transform(V_syn, W=W_seed.values, H=np.random.rand(n_components, 100))
    H_pred = model.components_

    # נרמול התוצאות
    Pred_Proportions = pd.DataFrame(H_pred, index=cell_types, columns=V_syn.columns)
    Pred_Proportions = Pred_Proportions.div(Pred_Proportions.sum(axis=0), axis=1)

    # --- שלב 4: השוואה וגרפים ---
    print("--- Generating Accuracy Plots ---")

    # בוחרים עד 4 תאים להצגה
    types_to_plot = cell_types[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    total_corr = []

    for i, ct in enumerate(types_to_plot):
        if i >= len(axes): break  # הגנה

        true_vals = True_Proportions.loc[ct]
        pred_vals = Pred_Proportions.loc[ct]

        # חישוב מתאם פירסון
        corr, _ = pearsonr(true_vals, pred_vals)
        total_corr.append(corr)

        ax = axes[i]
        ax.scatter(true_vals, pred_vals, alpha=0.6, color='dodgerblue')

        # קו האלכסון האדום (השלמות)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_title(f"Cell Type: {ct}\nAccuracy (R) = {corr:.3f}")
        ax.set_xlabel("True Proportion (Ground Truth)")
        ax.set_ylabel("Model Prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("In-Silico Validation: Model Accuracy Check", fontsize=16)
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    out_path = "results/plots/accuracy_validation.png"
    plt.savefig(out_path)

    print(f"\nSUCCESS! Validation plot saved to: {out_path}")
    print(f"Average Model Accuracy: {np.mean(total_corr):.3f} (max is 1.0)")


if __name__ == "__main__":
    run_validation()