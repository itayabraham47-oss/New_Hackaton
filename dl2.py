import pandas as pd
import requests
import os
import io

# הגדרת נתיבים
data_dir = "data/bulk"
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "tcga_luad_counts.tsv.gz")

# כתובת הנתונים
url_bulk = "https://gdc.xenahubs.net/download/TCGA-LUAD.htseq_counts.tsv.gz"

# --- שלב 1: הורדת הקובץ (רק אם הוא לא קיים כבר) ---
if not os.path.exists(file_path):
    print("File not found locally. Downloading from UCSC Xena...")

    # כאן התיקון: הוספת Headers כדי להיראות כמו דפדפן
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url_bulk, headers=headers, stream=True)
        response.raise_for_status()  # בדיקה אם הייתה שגיאה (כמו 403)

        # שמירת הקובץ בדיסק
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed successfully!")

    except Exception as e:
        print(f"Error downloading data: {e}")
        exit()  # עצור את הסקריפט אם ההורדה נכשלה
else:
    print("File already exists locally. Loading from disk...")

# --- שלב 2: טעינת הנתונים ל-Pandas ---
print("Loading data into Pandas DataFrame...")
try:
    # טעינה מהקובץ המקומי
    df_bulk = pd.read_csv(file_path, sep='\t', index_col=0, compression='gzip')

    # המרת האינדקס מ-Ensembl ID (למשל ENSG000001) לשמות גנים רגילים (Gene Symbols)
    # הערה: Xena לעיתים משתמש בפורמט מעורב, נבדוק את זה בהמשך.
    # כרגע נשאיר את זה כך.

    print(f"Success! Data Shape: {df_bulk.shape}")
    print("First 5 rows and columns:")
    print(df_bulk.iloc[:5, :5])

except Exception as e:
    print(f"Error loading dataframe: {e}")