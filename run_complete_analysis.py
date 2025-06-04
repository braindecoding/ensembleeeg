#!/usr/bin/env python3
# run_complete_analysis.py - Complete EEG analysis workflow

import numpy as np
import os
import subprocess
import sys

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking Dependencies...")
    print("=" * 50)
    
    required_packages = [
        'numpy', 'torch', 'sklearn', 'matplotlib', 'pywt', 'scipy', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies are available!")
        return True

def check_data_files():
    """Check if data files are available"""
    print("\n📂 Checking Data Files...")
    print("=" * 50)
    
    # Check original data
    original_data = "Data/EP1.01.txt"
    if os.path.exists(original_data):
        size_mb = os.path.getsize(original_data) / (1024 * 1024)
        print(f"  ✅ Original data: {original_data} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ Original data: {original_data} (not found)")
    
    # Check processed data files
    processed_files = [
        "Data/EP1.01.npy",
        "Data/EP1.01_labels.npy",
        "reshaped_data.npy",
        "advanced_wavelet_features.npy",
        "labels.npy"
    ]
    
    all_processed_exist = True
    for file in processed_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {file} (not found)")
            all_processed_exist = False
    
    return all_processed_exist

def run_data_preprocessing():
    """Run data preprocessing steps"""
    print("\n🔄 Running Data Preprocessing...")
    print("=" * 50)
    
    # Step 1: Convert raw data to numpy format
    if not os.path.exists("Data/EP1.01.npy"):
        print("  📝 Step 1: Converting raw data to numpy format...")
        try:
            result = subprocess.run([sys.executable, "convert_data.py"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("  ✅ Data conversion completed")
            else:
                print(f"  ❌ Data conversion failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ⚠️ Data conversion timed out")
            return False
    else:
        print("  ✅ Step 1: Raw data already converted")
    
    # Step 2: Extract wavelet features
    if not os.path.exists("advanced_wavelet_features.npy"):
        print("  📝 Step 2: Extracting wavelet features...")
        try:
            result = subprocess.run([sys.executable, "advanced_wavelet_features.py"], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("  ✅ Wavelet feature extraction completed")
            else:
                print(f"  ❌ Wavelet feature extraction failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ⚠️ Wavelet feature extraction timed out")
            return False
    else:
        print("  ✅ Step 2: Wavelet features already extracted")
    
    # Step 3: Reshape data for deep learning
    if not os.path.exists("reshaped_data.npy"):
        print("  📝 Step 3: Reshaping data for deep learning...")
        try:
            result = subprocess.run([sys.executable, "save_reshaped_data.py"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("  ✅ Data reshaping completed")
            else:
                print(f"  ❌ Data reshaping failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ⚠️ Data reshaping timed out")
            return False
    else:
        print("  ✅ Step 3: Data already reshaped")
    
    return True

def run_model_training():
    """Run model training"""
    print("\n🤖 Running Model Training...")
    print("=" * 50)
    
    # Train hybrid CNN-LSTM-Attention model
    print("  🧠 Training Hybrid CNN-LSTM-Attention model...")
    try:
        result = subprocess.run([sys.executable, "hybrid_cnn_lstm_attention.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("  ✅ Hybrid model training completed")
            # Extract accuracy from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Test accuracy:" in line:
                    print(f"    📊 {line.strip()}")
        else:
            print(f"  ❌ Hybrid model training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ⚠️ Hybrid model training timed out")
        return False
    
    # Train ensemble model
    print("  🎯 Training Ensemble model...")
    try:
        result = subprocess.run([sys.executable, "ensemble_model.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("  ✅ Ensemble model training completed")
            # Extract accuracy from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Final ensemble accuracy:" in line:
                    print(f"    📊 {line.strip()}")
        else:
            print(f"  ❌ Ensemble model training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ⚠️ Ensemble model training timed out")
        return False
    
    return True

def show_results():
    """Show final results and summary"""
    print("\n📊 Final Results Summary...")
    print("=" * 50)
    
    # Load data for statistics
    try:
        data = np.load("reshaped_data.npy")
        features = np.load("advanced_wavelet_features.npy")
        labels = np.load("labels.npy")
        
        print(f"📈 Dataset Statistics:")
        print(f"  Total samples: {len(labels)}")
        print(f"  Digit 6 samples: {np.sum(labels == 0)}")
        print(f"  Digit 9 samples: {np.sum(labels == 1)}")
        print(f"  EEG channels: {data.shape[1]}")
        print(f"  Time points: {data.shape[2]}")
        print(f"  Wavelet features: {features.shape[1]}")
        
    except FileNotFoundError:
        print("  ⚠️ Could not load data for statistics")
    
    # Check model files
    model_files = {
        'traditional_models.pkl': 'Traditional ML Models',
        'dl_model.pth': 'Deep Learning Model (Ensemble)',
        'hybrid_cnn_lstm_attention_model.pth': 'Hybrid CNN-LSTM-Attention Model',
        'meta_model.pkl': 'Meta Model (Stacking)'
    }
    
    print(f"\n🤖 Trained Models:")
    for file, description in model_files.items():
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  ✅ {description}: {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {description}: {file} (not found)")
    
    # Check visualization files
    viz_files = [
        'hybrid_cnn_lstm_attention_training_history.png',
        'wavelet_decomposition_digit6.png',
        'wavelet_decomposition_digit9.png',
        'wavelet_scalogram_digit6.png',
        'wavelet_scalogram_digit9.png',
        'eeg_sample_digit6_idx0.png',
        'eeg_sample_digit9_idx500.png'
    ]
    
    print(f"\n📊 Generated Visualizations:")
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            print(f"  ✅ {viz_file}")
        else:
            print(f"  ❌ {viz_file} (not found)")

def main():
    """Main function to run complete analysis"""
    print("🚀 Complete EEG Ensemble Analysis Workflow")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before proceeding.")
        return
    
    # Check data files
    data_available = check_data_files()
    
    # Run preprocessing if needed
    if not data_available:
        if not run_data_preprocessing():
            print("\n❌ Data preprocessing failed. Cannot proceed.")
            return
    else:
        print("\n✅ All processed data files are available!")
    
    # Run model training
    if not run_model_training():
        print("\n❌ Model training failed.")
        return
    
    # Show results
    show_results()
    
    print(f"\n✅ Complete EEG analysis workflow finished successfully!")
    print(f"🎯 The ensemble model combines traditional ML and deep learning approaches")
    print(f"📊 Check the generated visualization files for detailed analysis")

if __name__ == "__main__":
    main()
