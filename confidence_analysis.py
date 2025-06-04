#!/usr/bin/env python3
# confidence_analysis.py - Analisis singkat masalah confidence 1.0

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def quick_analysis():
    """Analisis cepat masalah confidence"""
    print("🔍 ANALISIS MASALAH CONFIDENCE 1.0")
    print("=" * 50)
    
    # Load data
    data = np.load("reshaped_data.npy")
    features = np.load("advanced_wavelet_features.npy")
    labels = np.load("labels.npy")
    
    print(f"📊 Dataset: {data.shape[0]} samples, {data.shape[1]} channels, {data.shape[2]} timepoints")
    
    # Cek karakteristik data yang mencurigakan
    print(f"\n🚨 INDIKATOR MASALAH:")
    
    # 1. Range nilai tidak wajar untuk EEG
    print(f"1. Range nilai: {data.min():.0f} - {data.max():.0f}")
    print(f"   ⚠️ EEG asli biasanya dalam range ±100 μV, bukan ribuan")
    
    # 2. Perbedaan antar kelas sangat kecil
    digit6_mean = data[labels == 0].mean()
    digit9_mean = data[labels == 1].mean()
    diff = abs(digit6_mean - digit9_mean)
    print(f"2. Perbedaan mean antar kelas: {diff:.4f}")
    print(f"   ⚠️ Perbedaan sangat kecil relatif terhadap range data")
    
    # 3. Test dengan model sederhana
    X_flat = data.reshape(data.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, lr.predict(X_train))
    test_acc = accuracy_score(y_test, lr.predict(X_test))
    
    print(f"3. Model sederhana (Logistic Regression):")
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   ⚠️ Akurasi terlalu tinggi untuk data EEG yang kompleks")
    
    # 4. Distribusi confidence
    proba = lr.predict_proba(X_test)
    max_proba = np.max(proba, axis=1)
    high_conf = np.sum(max_proba > 0.99)
    perfect_conf = np.sum(max_proba == 1.0)
    
    print(f"4. Distribusi confidence:")
    print(f"   Samples dengan confidence > 99%: {high_conf}/{len(max_proba)} ({high_conf/len(max_proba)*100:.1f}%)")
    print(f"   Samples dengan confidence = 100%: {perfect_conf}/{len(max_proba)} ({perfect_conf/len(max_proba)*100:.1f}%)")
    print(f"   ⚠️ Terlalu banyak prediksi dengan confidence sangat tinggi")
    
    print(f"\n📝 KESIMPULAN:")
    print("=" * 50)
    print("Berdasarkan analisis, ada indikasi kuat bahwa:")
    print("1. 🎯 Data mungkin bukan EEG asli atau sudah sangat diproses")
    print("2. 📊 Dataset terlalu mudah untuk diklasifikasi")
    print("3. 🔄 Model mengalami overfitting karena data terlalu sederhana")
    print("4. ⚙️ Confidence 1.0 menunjukkan model terlalu yakin (suspicious)")
    
    print(f"\n💡 REKOMENDASI:")
    print("- Verifikasi sumber dan keaslian data EEG")
    print("- Gunakan dataset EEG yang lebih challenging")
    print("- Tambahkan noise atau augmentasi data")
    print("- Implementasi regularization yang lebih kuat")
    print("- Gunakan cross-validation yang lebih ketat")

if __name__ == "__main__":
    quick_analysis()
