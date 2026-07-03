import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(raw_red_path, raw_white_path, output_dir):
    """
    Mengotomatisasi preprocessing Wine Quality Dataset.
    """
    print("Memulai otomatisasi preprocessing...")
    
    # 1. Memuat Dataset
    if not os.path.exists(raw_red_path) or not os.path.exists(raw_white_path):
        raise FileNotFoundError("Berkas dataset raw tidak ditemukan.")
        
    red_wine = pd.read_csv(raw_red_path, sep=';')
    white_wine = pd.read_csv(raw_white_path, sep=';')
    
    # Menambahkan tipe (1 untuk red, 0 untuk white)
    red_wine['type'] = 1
    white_wine['type'] = 0
    
    # Menggabungkan data
    df = pd.concat([red_wine, white_wine], ignore_index=True)
    print(f"Dataset berhasil dimuat. Shape awal: {df.shape}")
    
    # 2. Preprocessing & Pembersihan Data
    # Konversi target ke klasifikasi biner (good_quality: 1 untuk quality >= 6, else 0)
    df['good_quality'] = (df['quality'] >= 6).astype(int)
    df = df.drop('quality', axis=1)
    
    # Hapus data duplikat
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Menghapus duplikasi data. Berkurang {initial_len - len(df)} baris. Shape baru: {df.shape}")
    
    # Penanganan Outlier menggunakan IQR Capping
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(['good_quality', 'type'])
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print("Outlier telah ditangani dengan IQR Capping.")
    
    # 3. Train-Test Split (80% train, 20% test)
    X = df.drop('good_quality', axis=1)
    y = df['good_quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Standarisasi Fitur
    scaler = StandardScaler()
    scale_cols = X.columns.drop('type')
    
    # Standardize scale columns
    X_train_scaled_vals = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled_vals = scaler.transform(X_test[scale_cols])
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[scale_cols] = X_train_scaled_vals
    X_test_scaled[scale_cols] = X_test_scaled_vals
    
    # Gabungkan fitur dan target kembali
    train_processed = pd.concat([X_train_scaled, y_train], axis=1)
    test_processed = pd.concat([X_test_scaled, y_test], axis=1)
    
    # 5. Menyimpan Output
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
    # Path default
    raw_red = 'winequality_raw/winequality-red.csv'
    raw_white = 'winequality_raw/winequality-white.csv'
    out_dir = 'winequality_preprocessing'
    
    # Jika dijalankan dari direktori yang berbeda, sesuaikan path
    if not os.path.exists(raw_red):
        raw_red = 'Eksperimen_SML_Muhammad-Halim-Amrin/winequality_raw/winequality-red.csv'
        raw_white = 'Eksperimen_SML_Muhammad-Halim-Amrin/winequality_raw/winequality-white.csv'
        out_dir = 'Eksperimen_SML_Muhammad-Halim-Amrin/winequality_preprocessing'
        
    try:
        preprocess_data(raw_red, raw_white, out_dir)
    except Exception as e:
        print(f"Error saat preprocessing: {e}", file=sys.stderr)
        sys.exit(1)
