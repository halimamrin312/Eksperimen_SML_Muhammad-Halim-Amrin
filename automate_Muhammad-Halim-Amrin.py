import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(raw_path, output_dir):
    """
    Mengotomatisasi preprocessing Cardiovascular Disease Dataset.
    """
    print("Memulai otomatisasi preprocessing...")

    # 1. Memuat Dataset
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Berkas dataset raw tidak ditemukan: {raw_path}")

    df = pd.read_csv(raw_path, sep=';')
    print(f"Dataset berhasil dimuat. Shape awal: {df.shape}")

    # 2. Drop kolom id (tidak relevan sebagai fitur)
    df = df.drop('id', axis=1)

    # 3. Konversi age dari hari ke tahun
    df['age'] = (df['age'] / 365).round(1)

    # 4. Hapus data duplikat
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Menghapus duplikasi data. Berkurang {initial_len - len(df)} baris. Shape baru: {df.shape}")

    # 5. Penanganan Outlier menggunakan IQR Capping
    # Kolom biner/kategorikal tidak di-cap
    binary_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(binary_cols)

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print(f"Outlier telah ditangani dengan IQR Capping pada kolom: {list(num_cols)}")

    # 6. Train-Test Split (80% train, 20% test)
    X = df.drop('cardio', axis=1)
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Standarisasi Fitur numerik kontinu
    # Kolom biner/ordinal tidak di-scale
    scale_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    # Gabungkan fitur dan target kembali
    train_processed = pd.concat([X_train_scaled, y_train], axis=1)
    test_processed = pd.concat([X_test_scaled, y_test], axis=1)

    # 8. Menyimpan Output
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train_processed.csv')
    test_path = os.path.join(output_dir, 'test_processed.csv')
    scaler_path = os.path.join(output_dir, 'scaler.joblib')

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"Preprocessing selesai.")
    print(f"Data latih disimpan di: {train_path} (Shape: {train_processed.shape})")
    print(f"Data uji disimpan di: {test_path} (Shape: {test_processed.shape})")
    print(f"StandardScaler disimpan di: {scaler_path}")

    return train_processed, test_processed

if __name__ == "__main__":
    raw = 'cardio_raw/cardio_train.csv'
    out_dir = 'cardio_preprocessing'

    if not os.path.exists(raw):
        raw = 'Eksperimen_SML_Muhammad-Halim-Amrin/cardio_raw/cardio_train.csv'
        out_dir = 'Eksperimen_SML_Muhammad-Halim-Amrin/cardio_preprocessing'

    try:
        preprocess_data(raw, out_dir)
    except Exception as e:
        print(f"Error saat preprocessing: {e}", file=sys.stderr)
        sys.exit(1)
