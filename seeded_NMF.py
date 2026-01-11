import pandas as pd
import numpy as np
from sklearn.decomposition import NMF


def run_seeded_nmf(bulk_df, markers_dict, epsilon=1e-6):
    """
    מבצע פירוק Seeded NMF כדי למצוא פרופורציות של סוגי תאים.

    Parameters:
    -----------
    bulk_df : pd.DataFrame
        מטריצת הביטוי (Bulk Data).
        שורות = גנים (Gene Symbols), עמודות = חולים/דגימות.
        הערכים צריכים להיות מנורמלים (למשל CPM או TPM).

    markers_dict : dict
        מילון שבו המפתח הוא שם סוג התא (למשל 'T-cells')
        והערך הוא רשימה של שמות גנים (מרקרים) המאפיינים אותו.

    epsilon : float
        ערך קטן למילוי מקומות שאינם מרקרים במטריצת ה-Seed
        (כדי למנוע אפסים מוחלטים שעלולים לתקוע את האלגוריתם).

    Returns:
    --------
    H_normalized : pd.DataFrame
        מטריצת הפרופורציות החזויה (סוגי תאים X חולים).
        כל עמודה מסתכמת ל-1.
    W_learned : pd.DataFrame
        מטריצת החתימות שנלמדה סופית (גנים X סוגי תאים).
    """

    # 1. יצירת רשימת כל הגנים המרקרים (האיחוד של כל הרשימות)
    all_marker_genes = set()
    for genes in markers_dict.values():
        all_marker_genes.update(genes)

    # 2. חיתוך הדאטה: נשארים רק עם גנים שקיימים גם ב-Bulk וגם ברשימת המרקרים
    # זהו שלב קריטי להפחתת רעש ולשיפור הביצועים
    common_genes = list(all_marker_genes.intersection(bulk_df.index))

    if len(common_genes) < 1:
        raise ValueError("Error: Not enough common genes found between Bulk data and Markers list.")

    print(f"Running Seeded NMF on {len(common_genes)} common genes...")

    # סינון ה-Bulk לפי הגנים המשותפים
    V = bulk_df.loc[common_genes].copy()

    # הוספת ערך קטן למניעת אפסים (NMF לא אוהב אפסים מוחלטים לפעמים)
    V = V + epsilon

    # 3. בניית מטריצת האתחול (W_init) - זהו ה-Seeding!
    # שורות = גנים משותפים, עמודות = סוגי תאים
    cell_types = list(markers_dict.keys())
    n_components = len(cell_types)  # מספר סוגי התאים הוא ה-Rank של ה-NMF
    n_genes = len(common_genes)
    n_samples = V.shape[1]

    # אתחול המטריצה בערך נמוך (epsilon)
    W_init = np.full((n_genes, n_components), epsilon)

    # מילוי ה-Seed: אם גן הוא מרקר של תא מסוים, ניתן לו ערך גבוה (למשל 1 או ממוצע השורה)
    # הערה: אפשר לשכלל ולהכניס כאן את הביטוי הממוצע האמיתי מה-Single Cell אם יש לכם אותו.
    # לצורך ההאקתון, סממן בינארי (1) מספיק טוב בתור התחלה.
    for col_idx, cell_type in enumerate(cell_types):
        # עבור כל תא, קח את הגנים המרקרים שלו
        specific_markers = markers_dict[cell_type]

        # בדוק אילו מהם נמצאים ברשימת הגנים המשותפים שלנו
        valid_markers = [g for g in specific_markers if g in common_genes]

        # עדכן את המטריצה: עבור גנים אלו, שים ערך גבוה בטור של התא הזה
        # שימוש בממוצע הביטוי של הגן ב-Bulk נותן נקודת התחלה טובה לסקאלה
        for gene in valid_markers:
            row_idx = common_genes.index(gene)
            W_init[row_idx, col_idx] = np.mean(V.loc[gene])

            # 4. אתחול מטריצת H (פרופורציות)
    # נאתחל אותה באופן אקראי (או אחיד), האלגוריתם ילמד אותה
    H_init = np.random.rand(n_components, n_samples)

    # 5. הרצת המודל (Scikit-Learn NMF)
    # init='custom' אומר למודל: אל תנחש, קח את המטריצות שבנינו
    model = NMF(n_components=n_components,
                init='custom',
                solver='cd',  # Coordinate Descent - יעיל ל-NMF
                beta_loss='frobenius',
                max_iter=5000,
                random_state=42)

    # התאמת המודל. שימו לב שאנחנו מעבירים את המטריצות שיצרנו
    W_learned = model.fit_transform(V, W=W_init, H=H_init)
    H_learned = model.components_

    # 6. ארגון התוצאות ונרמול
    # המרת H ל-DataFrame לקריאות נוחה
    H_df = pd.DataFrame(H_learned, index=cell_types, columns=V.columns)

    # המרת W ל-DataFrame
    W_df = pd.DataFrame(W_learned, index=common_genes, columns=cell_types)

    # נרמול: הפרופורציות של כל חולה צריכות להסתכם ל-1 (100%)
    # מחלקים כל עמודה בסכום שלה
    H_normalized = H_df.div(H_df.sum(axis=0), axis=1)

    return H_normalized, W_df


# === דוגמה לשימוש (Run Example) ===
if __name__ == "__main__":
    # יצירת דאטה מזויף רק כדי לבדוק שהקוד רץ
    genes = ['CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'ACTB', 'GAPDH']
    samples = ['Patient_1', 'Patient_2', 'Patient_3']

    # נתונים מדומים (Bulk)
    dummy_bulk = pd.DataFrame(
        np.random.rand(7, 3) * 100,
        index=genes,
        columns=samples
    )

    # מילון מרקרים (מה שתקבלו מצוות א')
    dummy_markers = {
        'T-cells': ['CD3D', 'CD4', 'CD8A'],
        'B-cells': ['MS4A1', 'CD19']
    }

    try:
        proportions, signatures = run_seeded_nmf(dummy_bulk, dummy_markers)
        print("Success! Proportions Matrix:")
        print(proportions)
    except Exception as e:
        print(e)