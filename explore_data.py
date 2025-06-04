#!/usr/bin/env python3
# explore_data.py - Explore the EEG dataset and show basic information

import numpy as np
import matplotlib.pyplot as plt
import os

def explore_processed_data():
    """Explore the already processed data"""
    print("🔍 Exploring Processed EEG Data")
    print("=" * 50)
    
    # Load processed data
    try:
        print("📂 Loading processed data...")
        reshaped_data = np.load("reshaped_data.npy")
        wavelet_features = np.load("advanced_wavelet_features.npy")
        labels = np.load("labels.npy")
        
        print(f"  ✅ Reshaped data: {reshaped_data.shape}")
        print(f"  ✅ Wavelet features: {wavelet_features.shape}")
        print(f"  ✅ Labels: {labels.shape}")
        
        # Show data statistics
        print(f"\n📊 Data Statistics:")
        print(f"  Total samples: {len(labels)}")
        print(f"  Digit 6 samples: {np.sum(labels == 0)}")
        print(f"  Digit 9 samples: {np.sum(labels == 1)}")
        print(f"  EEG channels: {reshaped_data.shape[1]}")
        print(f"  Time points per channel: {reshaped_data.shape[2]}")
        print(f"  Wavelet features per sample: {wavelet_features.shape[1]}")
        
        # Show data ranges
        print(f"\n📈 Data Ranges:")
        print(f"  Raw EEG data range: [{reshaped_data.min():.4f}, {reshaped_data.max():.4f}]")
        print(f"  Wavelet features range: [{wavelet_features.min():.4f}, {wavelet_features.max():.4f}]")
        
        # Show sample data for first digit 6 and digit 9
        digit6_idx = np.where(labels == 0)[0][0]
        digit9_idx = np.where(labels == 1)[0][0]
        
        print(f"\n🧠 Sample EEG Data:")
        print(f"  First Digit 6 sample (index {digit6_idx}):")
        print(f"    Shape: {reshaped_data[digit6_idx].shape}")
        print(f"    Channel 0 (first 10 values): {reshaped_data[digit6_idx, 0, :10]}")
        
        print(f"  First Digit 9 sample (index {digit9_idx}):")
        print(f"    Shape: {reshaped_data[digit9_idx].shape}")
        print(f"    Channel 0 (first 10 values): {reshaped_data[digit9_idx, 0, :10]}")
        
        return reshaped_data, wavelet_features, labels
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def visualize_sample_data(data, labels, sample_idx=0):
    """Visualize a sample EEG data"""
    if data is None:
        return
        
    print(f"\n📊 Visualizing sample {sample_idx}...")
    
    sample = data[sample_idx]
    label = labels[sample_idx]
    digit = "6" if label == 0 else "9"
    
    # EEG channel names (standard 14-channel setup)
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot all channels
    for i in range(min(14, sample.shape[0])):
        plt.subplot(4, 4, i+1)
        plt.plot(sample[i])
        plt.title(f'{channel_names[i]} (Ch {i+1})')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'EEG Channels for Digit {digit} (Sample {sample_idx})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig(f'eeg_sample_digit{digit}_idx{sample_idx}.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Visualization saved as 'eeg_sample_digit{digit}_idx{sample_idx}.png'")
    plt.close()

def show_model_results():
    """Show results from trained models"""
    print(f"\n🤖 Model Results Summary:")
    print("=" * 50)
    
    # Check if model files exist
    model_files = [
        'traditional_models.pkl',
        'dl_model.pth', 
        'hybrid_cnn_lstm_attention_model.pth',
        'meta_model.pkl'
    ]
    
    print("📁 Available Model Files:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {model_file} (not found)")
    
    # Check visualization files
    viz_files = [
        'hybrid_cnn_lstm_attention_training_history.png',
        'wavelet_decomposition_digit6.png',
        'wavelet_decomposition_digit9.png',
        'wavelet_scalogram_digit6.png',
        'wavelet_scalogram_digit9.png'
    ]
    
    print(f"\n📊 Available Visualization Files:")
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            print(f"  ✅ {viz_file}")
        else:
            print(f"  ❌ {viz_file} (not found)")

def main():
    """Main function"""
    print("🚀 EEG Data Explorer")
    print("=" * 50)
    
    # Explore processed data
    data, features, labels = explore_processed_data()
    
    if data is not None:
        # Visualize samples
        digit6_idx = np.where(labels == 0)[0][0]  # First digit 6
        digit9_idx = np.where(labels == 1)[0][0]  # First digit 9
        
        visualize_sample_data(data, labels, digit6_idx)
        visualize_sample_data(data, labels, digit9_idx)
    
    # Show model results
    show_model_results()
    
    print(f"\n✅ Data exploration completed!")

if __name__ == "__main__":
    main()
