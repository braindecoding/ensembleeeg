#!/usr/bin/env python3
# investigate_confidence.py - Investigasi masalah confidence 1.0

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def investigate_data_quality():
    """Investigasi kualitas data"""
    print("🔍 INVESTIGASI KUALITAS DATA")
    print("=" * 50)
    
    # Load data
    data = np.load("reshaped_data.npy")
    features = np.load("advanced_wavelet_features.npy")
    labels = np.load("labels.npy")
    
    print(f"📊 Data shape: {data.shape}")
    print(f"📊 Features shape: {features.shape}")
    print(f"📊 Labels shape: {labels.shape}")
    
    # Cek range nilai yang mencurigakan
    print(f"\n📈 Range nilai data:")
    print(f"  Min: {data.min():.2f}")
    print(f"  Max: {data.max():.2f}")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Std: {data.std():.2f}")
    
    # Cek apakah nilai terlalu uniform
    print(f"\n🎯 Analisis variabilitas:")
    sample_vars = np.var(data.reshape(data.shape[0], -1), axis=1)
    print(f"  Variance per sample - Min: {sample_vars.min():.2f}, Max: {sample_vars.max():.2f}")
    
    # Cek perbedaan antar kelas
    digit6_data = data[labels == 0]
    digit9_data = data[labels == 1]
    
    print(f"\n🔢 Perbedaan antar kelas:")
    print(f"  Digit 6 mean: {digit6_data.mean():.4f}")
    print(f"  Digit 9 mean: {digit9_data.mean():.4f}")
    print(f"  Selisih mean: {abs(digit6_data.mean() - digit9_data.mean()):.4f}")
    
    # Cek apakah data terlihat artificial
    print(f"\n⚠️ Indikator data artificial:")
    
    # 1. Range nilai yang tidak wajar untuk EEG
    if data.min() > 1000:
        print(f"  🚨 Range nilai terlalu tinggi untuk EEG (biasanya μV)")
    
    # 2. Variabilitas terlalu rendah
    if sample_vars.std() < 100:
        print(f"  🚨 Variabilitas antar sample terlalu rendah")
    
    # 3. Perbedaan kelas terlalu kecil
    if abs(digit6_data.mean() - digit9_data.mean()) < 10:
        print(f"  🚨 Perbedaan antar kelas sangat kecil")
    
    return data, features, labels

def test_simple_model_overfitting(data, labels):
    """Test overfitting dengan model sederhana"""
    print(f"\n🧪 TEST OVERFITTING MODEL SEDERHANA")
    print("=" * 50)
    
    # Flatten data
    X_flat = data.reshape(data.shape[0], -1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Model sederhana
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # Evaluasi
    train_acc = accuracy_score(y_train, lr.predict(X_train))
    test_acc = accuracy_score(y_test, lr.predict(X_test))
    
    print(f"📊 Logistic Regression sederhana:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Overfitting gap: {train_acc - test_acc:.4f}")
    
    # Cek confidence distribution
    proba = lr.predict_proba(X_test)
    max_proba = np.max(proba, axis=1)
    
    print(f"\n📈 Distribusi confidence:")
    print(f"  Mean: {max_proba.mean():.4f}")
    print(f"  Std: {max_proba.std():.4f}")
    print(f"  Min: {max_proba.min():.4f}")
    print(f"  Max: {max_proba.max():.4f}")
    print(f"  Samples dengan confidence > 0.99: {np.sum(max_proba > 0.99)}/{len(max_proba)}")
    print(f"  Samples dengan confidence = 1.0: {np.sum(max_proba == 1.0)}/{len(max_proba)}")
    
    if np.sum(max_proba > 0.99) > len(max_proba) * 0.5:
        print(f"  🚨 Terlalu banyak prediksi dengan confidence tinggi!")
    
    return lr

def test_random_baseline(data, labels):
    """Test dengan random labels untuk deteksi memorization"""
    print(f"\n🎲 TEST RANDOM BASELINE")
    print("=" * 50)
    
    X_flat = data.reshape(data.shape[0], -1)
    
    # Generate random labels
    random_labels = np.random.randint(0, 2, size=len(labels))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, random_labels, test_size=0.2, random_state=42
    )
    
    lr_random = LogisticRegression(max_iter=1000, random_state=42)
    lr_random.fit(X_train, y_train)
    
    random_acc = accuracy_score(y_test, lr_random.predict(X_test))
    
    print(f"📊 Akurasi dengan random labels: {random_acc:.4f}")
    
    if random_acc > 0.6:
        print(f"  🚨 Akurasi terlalu tinggi untuk random labels!")
        print(f"  💡 Ini mengindikasikan model bisa memorize data")
    else:
        print(f"  ✅ Akurasi wajar untuk random labels")

def investigate_trained_models():
    """Investigasi model yang sudah dilatih"""
    print(f"\n🤖 INVESTIGASI MODEL TERLATIH")
    print("=" * 50)
    
    try:
        # Load traditional models
        traditional_models = joblib.load('traditional_models.pkl')
        
        # Load sample data
        data = np.load("reshaped_data.npy")
        features = np.load("advanced_wavelet_features.npy")
        labels = np.load("labels.npy")
        
        # Prepare data seperti di ensemble model
        X_flat = data.reshape(data.shape[0], -1)
        X_combined = np.hstack((X_flat, features))
        
        # Test beberapa sample
        test_indices = [0, 1, 2, 500, 501, 502]
        
        for idx in test_indices:
            sample = X_combined[idx:idx+1]
            true_label = labels[idx]
            
            print(f"\n📊 Sample {idx} (True: {true_label}):")
            
            # Test dengan setiap model
            if 'lr' in traditional_models and traditional_models['lr'] is not None:
                lr_proba = traditional_models['lr'].predict_proba(sample)[0]
                lr_pred = np.argmax(lr_proba)
                lr_conf = np.max(lr_proba)
                
                print(f"  LR: pred={lr_pred}, conf={lr_conf:.4f}, proba={lr_proba}")
                
                if lr_conf == 1.0:
                    print(f"    🚨 Confidence = 1.0 detected!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

def main():
    """Main function"""
    print("🚀 INVESTIGASI MASALAH CONFIDENCE 1.0")
    print("=" * 60)
    
    # 1. Investigasi kualitas data
    data, features, labels = investigate_data_quality()
    
    # 2. Test overfitting dengan model sederhana
    simple_model = test_simple_model_overfitting(data, labels)
    
    # 3. Test random baseline
    test_random_baseline(data, labels)
    
    # 4. Investigasi model terlatih
    investigate_trained_models()
    
    print(f"\n📝 KESIMPULAN:")
    print("=" * 50)
    print("Confidence 1.0 bisa disebabkan oleh:")
    print("1. 🎯 Data yang terlalu mudah dipisahkan (linearly separable)")
    print("2. 🔄 Overfitting pada training data")
    print("3. 📊 Data artificial atau synthetic")
    print("4. ⚙️ Regularization yang terlalu lemah")
    print("5. 🧠 Model yang terlalu kompleks untuk dataset sederhana")
    
    print(f"\n💡 REKOMENDASI:")
    print("- Gunakan cross-validation yang lebih ketat")
    print("- Tambahkan regularization yang lebih kuat")
    print("- Cek apakah data benar-benar dari EEG asli")
    print("- Gunakan dataset yang lebih challenging")

if __name__ == "__main__":
    main()
