import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# ==========================================
# 1. טעינת נתונים (משותפת לשני המודלים)
# ==========================================
def load_data():
    data_dir = "data"
    print("Loading Data...")

    # טעינת מרקרים
    tsv_path = f"{data_dir}/sc/NSCLC_GSE99254_AllDiffGenes_table.tsv"
    if not os.path.exists(tsv_path):
        print(f"Error: Marker file not found at {tsv_path}")
        return None, None, None

    df_markers = pd.read_csv(tsv_path, sep='\t')
    # זיהוי עמודת סוג תא
    possible_cols = [c for c in df_markers.columns if 'lineage' in c or 'type' in c]
    cell_col = possible_cols[0] if possible_cols else df_markers.columns[0]

    markers_dict = {}
    for ct, grp in df_markers.groupby(cell_col):
        if 'log2FC' in grp.columns: grp = grp.sort_values('log2FC', ascending=False)
        # ניקוי שמות
        clean_name = str(ct).replace(' ', '_').replace('/', '_').replace('+', '')
        markers_dict[clean_name] = grp['Gene'].head(50).tolist()

    # טעינת Bulk
    bulk_path = f"{data_dir}/bulk/TCGA-LUAD.HiSeqV2.gz"
    if not os.path.exists(bulk_path):
        print("Error: Bulk file not found")
        return None, None, None

    df_bulk = pd.read_csv(bulk_path, sep='\t', index_col=0, compression='gzip')
    df_bulk.index = df_bulk.index.str.upper().str.split('|').str[0]

    # חיתוך גנים
    all_markers = set([str(g).upper() for sublist in markers_dict.values() for g in sublist])
    common = list(all_markers.intersection(set(df_bulk.index)))
    df_filtered = df_bulk.loc[common].fillna(0)
    df_filtered[df_filtered < 0] = 0

    # טעינת הישרדות
    surv_path = f"{data_dir}/bulk/TCGA-LUAD.survival.tsv.gz"
    if not os.path.exists(surv_path):
        print("Error: Survival file not found")
        return None, None, None

    surv_df = pd.read_csv(surv_path, sep='\t', index_col=0, compression='gzip')
    surv_df.index = surv_df.index.str[:12]
    surv_df = surv_df[~surv_df.index.duplicated(keep='first')]

    return df_filtered, surv_df, markers_dict


# ==========================================
# 2. פונקציית הרצה (מודולרית) עם תיקונים
# ==========================================
def run_model(df, surv_df, markers_dict, mode='seeded', ax=None):
    if df is None: return 1.0

    cell_types = list(markers_dict.keys())
    n_components = len(cell_types)

    # --- שלב האלגוריתם ---
    if mode == 'seeded':
        print("Running Seeded NMF (Our Model)...")
        W = pd.DataFrame(0.1, index=df.index, columns=cell_types)
        for ct, genes in markers_dict.items():
            for g in genes:
                if str(g).upper() in W.index: W.loc[str(g).upper(), ct] = 10.0

        model = NMF(n_components=n_components, init='custom', solver='cd', max_iter=4000, random_state=42)
        W_out = model.fit_transform(df, W=W.values, H=np.random.rand(n_components, df.shape[1]))

    else:
        print("Running Random NMF (Standard)...")
        # במודל רגיל אין ידע מוקדם
        model = NMF(n_components=n_components, init='random', solver='cd', max_iter=4000, random_state=42)
        W_out = model.fit_transform(df)

    # --- עיבוד תוצאות ---
    H = model.components_
    df_H = pd.DataFrame(H, columns=df.columns)

    # בחירת עמודת המטרה לגרף
    if mode == 'random':
        # נותנים שמות גנריים
        df_H.index = [f"Component_{i}" for i in range(n_components)]
        # במודל רנדומלי, חלק מהרכיבים יכולים להיות "מתים" (אפסים).
        # נבחר את הרכיב עם השונות (Variance) הכי גבוהה כדי לתת לו סיכוי הוגן
        variances = df_H.var(axis=1)
        target_col = variances.idxmax()
        print(f"   Random mode selected strongest component: {target_col}")
    else:
        df_H.index = cell_types
        # מחפשים CD8
        target_col = next((c for c in cell_types if 'CD8' in c), cell_types[0])
        print(f"   Seeded mode selected target: {target_col}")

    # חישוב פרופורציות
    df_prop = df_H.div(df_H.sum(axis=0), axis=1)

    # חיתוך שמות ומיזוג (כולל התיקון ל-GroupBy)
    df_prop.columns = [c[:12] for c in df_prop.columns]
    # התיקון: שימוש ב-T.groupby.T
    df_prop = df_prop.T.groupby(level=0).mean().T

    merged = df_prop.T.join(surv_df, how='inner')

    # בדיקת תקינות המיזוג
    if merged.empty:
        print("   Warning: No patients matched after merge.")
        return 1.0

    # חלוקה לקבוצות (עם הגנה מקריסה)
    if target_col in merged.columns:
        median = merged[target_col].median()
        high = merged[merged[target_col] >= median]
        low = merged[merged[target_col] < median]

        # הגנה: אם אחת הקבוצות ריקה (קורה הרבה ברנדומלי), אי אפשר לחשב
        if len(high) < 5 or len(low) < 5:
            print(f"   Error: Groups too small for analysis (High: {len(high)}, Low: {len(low)}).")
            ax.text(0.5, 0.5, "Model Failed to Separate\nPatients (One group empty)",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{mode.capitalize()} Model (Failed)")
            return 1.0

        # חישוב KM
        kmf = KaplanMeierFitter()
        kmf.fit(high['OS.time'], high['OS'], label='High')
        kmf.plot_survival_function(ax=ax, ci_show=False)
        kmf.fit(low['OS.time'], low['OS'], label='Low')
        kmf.plot_survival_function(ax=ax, ci_show=False)

        # חישוב P-value
        pval = logrank_test(high['OS.time'], low['OS.time'], event_observed_A=high['OS'],
                            event_observed_B=low['OS']).p_value

        title = "Our Model (Seeded)" if mode == 'seeded' else "Random Initialization"
        ax.set_title(f"{title}\nP-value = {pval:.4f}")
        return pval

    return 1.0


# ==========================================
# 3. Main
# ==========================================
if __name__ == "__main__":
    df, surv, markers = load_data()

    if df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # הרצת המודל שלנו
        p_seeded = run_model(df, surv, markers, mode='seeded', ax=axes[0])

        # הרצת מודל אקראי
        p_random = run_model(df, surv, markers, mode='random', ax=axes[1])

        plt.suptitle("Impact of Biological Knowledge on Survival Prediction", fontsize=16)
        plt.tight_layout()

        os.makedirs("results/plots", exist_ok=True)
        plt.savefig("results/plots/innovation_comparison.png")
        print("\nComparison saved to results/plots/innovation_comparison.png")
        plt.show()

        print(f"\n--- Final Results ---")
        print(f"Seeded P-value: {p_seeded:.5f}")
        print(f"Random P-value: {p_random:.5f}")