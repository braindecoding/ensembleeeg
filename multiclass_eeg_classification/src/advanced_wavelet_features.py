#!/usr/bin/env python3
# advanced_wavelet_features.py - Advanced wavelet feature extraction for EEG data

import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import coherence
import os

def load_digits_simple(file_path, target_digits=[6, 9], max_per_digit=500):
    """Load EEG data for digit classification"""
    print(f"📂 Loading data for digits {target_digits}...")
    
    if file_path is None or not os.path.exists(file_path):
        print("❌ Dataset file not found!")
        return None, None
    
    print(f"📖 Reading file: {file_path}")
    
    # Initialize data containers
    data_6 = []
    data_9 = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
                
            # Split by TAB
            parts = line.split('\t')
            
            # Need at least 7 columns
            if len(parts) < 7:
                continue
            
            try:
                # Column 5 (index 4) = digit
                digit = int(parts[4])
                
                # Only process if it's in target digits
                if digit in target_digits:
                    # Column 7 (index 6) = data
                    data_string = parts[6]
                    
                    # Parse comma-separated values
                    values = [np.float64(x.strip()) for x in data_string.split(',') if x.strip()]
                    
                    # Store based on digit
                    if digit == 6 and len(data_6) < max_per_digit:
                        data_6.append(values)
                    elif digit == 9 and len(data_9) < max_per_digit:
                        data_9.append(values)
                    
                    # Progress
                    if (len(data_6) + len(data_9)) % 100 == 0:
                        print(f"  Found: {len(data_6)} digit-6, {len(data_9)} digit-9")
                    
                    # Stop when we have enough
                    if len(data_6) >= max_per_digit and len(data_9) >= max_per_digit:
                        break
                        
            except (ValueError, IndexError):
                continue
    
    print(f"✅ Final count: {len(data_6)} digit-6, {len(data_9)} digit-9")
    
    if len(data_6) == 0 or len(data_9) == 0:
        print("❌ Missing data for one or both digits!")
        return None, None
    
    # Combine data and labels
    all_data = data_6 + data_9
    all_labels = [0] * len(data_6) + [1] * len(data_9)  # 0 for digit 6, 1 for digit 9
    
    # Normalize lengths (simple padding/truncating)
    normalized_data = []
    target_length = 1792  # 14 channels * 128 timepoints
    
    for trial in all_data:
        if len(trial) >= target_length:
            # Truncate if too long
            normalized_data.append(trial[:target_length])
        else:
            # Pad with repetition if too short
            trial_copy = trial.copy()
            while len(trial_copy) < target_length:
                trial_copy.extend(trial[:min(len(trial), target_length - len(trial_copy))])
            normalized_data.append(trial_copy[:target_length])
    
    data = np.array(normalized_data, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.int32)
    
    # Check for NaN or infinity values
    if np.isnan(data).any() or np.isinf(data).any():
        print("  ⚠️ Warning: NaN or Infinity values detected in data, replacing with zeros")
        data = np.nan_to_num(data)
    
    print(f"  📊 Data shape: {data.shape}")
    
    return data, labels

def extract_advanced_wavelet_features(data):
    """Extract advanced wavelet features from EEG data"""
    print("\n🧩 Extracting advanced wavelet features...")
    
    # Reshape data to 14 channels x 128 timepoints
    reshaped_data = []
    for trial in data:
        try:
            # Reshape to 14 x 128
            reshaped = trial.reshape(14, 128)
            reshaped_data.append(reshaped)
        except ValueError:
            print(f"  ⚠️ Reshape failed for trial with length {len(trial)}")
            continue
    
    # Define channel groups for regional analysis
    frontal_channels = [0, 1, 2, 3, 11, 12, 13]  # AF3, F7, F3, FC5, F4, F8, AF4
    temporal_channels = [4, 5, 8, 9]             # T7, P7, P8, T8
    occipital_channels = [6, 7]                  # O1, O2
    left_channels = [0, 1, 2, 3, 4, 5, 6]        # Left hemisphere
    right_channels = [7, 8, 9, 10, 11, 12, 13]   # Right hemisphere
    
    # Define wavelet parameters
    wavelet = 'db4'  # Daubechies wavelet with 4 vanishing moments
    level = 5        # Decomposition level for DWT
    wp_level = 3     # Decomposition level for WPD
    
    # Define frequency bands
    fs = 128  # Sampling frequency (Hz)
    
    # Extract wavelet features
    wavelet_features = []
    
    for trial_idx, trial in enumerate(reshaped_data):
        if trial_idx % 50 == 0:
            print(f"  Processing trial {trial_idx+1}/{len(reshaped_data)}...")
            
        trial_features = []
        
        # Process each channel
        for channel in range(trial.shape[0]):
            channel_signal = trial[channel]
            
            # 1. Discrete Wavelet Transform (DWT)
            try:
                coeffs = pywt.wavedec(channel_signal, wavelet, level=level)
                
                # Extract features from each level
                for i in range(level + 1):
                    # Calculate energy
                    energy = np.sum(coeffs[i]**2)
                    
                    # Calculate entropy
                    coef_norm = coeffs[i]**2 / (np.sum(coeffs[i]**2) + 1e-10)
                    entropy = -np.sum(coef_norm * np.log(coef_norm + 1e-10))
                    
                    # Calculate mean and standard deviation
                    mean = np.mean(coeffs[i])
                    std = np.std(coeffs[i])
                    
                    # Calculate kurtosis and skewness
                    kurtosis = np.mean((coeffs[i] - mean)**4) / (std**4 + 1e-10) if std > 0 else 0
                    skewness = np.mean((coeffs[i] - mean)**3) / (std**3 + 1e-10) if std > 0 else 0
                    
                    # Add features
                    trial_features.extend([energy, entropy, mean, std, kurtosis, skewness])
            except Exception as e:
                print(f"  ⚠️ DWT failed for channel {channel}: {str(e)}")
                # Add zeros if DWT fails
                trial_features.extend([0] * (level + 1) * 6)
            
            # 2. Wavelet Packet Decomposition (WPD)
            try:
                wp = pywt.WaveletPacket(channel_signal, wavelet, maxlevel=wp_level)
                
                # Get nodes at level wp_level
                wp_nodes = [node.path for node in wp.get_level(wp_level, 'natural')]
                
                # Calculate energy for each node
                wp_energies = []
                for node in wp_nodes:
                    node_data = wp[node].data
                    energy = np.sum(node_data**2)
                    wp_energies.append(energy)
                
                # Normalize energies
                total_energy = np.sum(wp_energies) + 1e-10
                wp_energies_norm = [e / total_energy for e in wp_energies]
                
                # Add WPD features
                trial_features.extend(wp_energies_norm)
            except Exception as e:
                print(f"  ⚠️ WPD failed for channel {channel}: {str(e)}")
                # Add zeros if WPD fails
                trial_features.extend([0] * 2**wp_level)
            
            # 3. Continuous Wavelet Transform (CWT) for specific scales
            try:
                # Use fewer scales to reduce dimensionality
                scales = np.arange(1, 32, 4)  # Downsample scales
                
                # Perform CWT
                cwtmatr, freqs = pywt.cwt(channel_signal, scales, 'morl')
                
                # Calculate energy at each scale
                cwt_energies = np.sum(abs(cwtmatr)**2, axis=1)
                
                # Normalize energies
                total_energy = np.sum(cwt_energies) + 1e-10
                cwt_energies_norm = [e / total_energy for e in cwt_energies]
                
                # Add CWT features
                trial_features.extend(cwt_energies_norm)
            except Exception as e:
                print(f"  ⚠️ CWT failed for channel {channel}: {str(e)}")
                # Add zeros if CWT fails
                trial_features.extend([0] * len(scales))
        
        # 4. Cross-channel wavelet coherence (simplified version)
        # Calculate coherence between key channel pairs
        key_pairs = [
            (2, 11),   # F3-F4 (left-right frontal)
            (6, 7),    # O1-O2 (left-right occipital)
            (4, 9),    # T7-T8 (left-right temporal)
            (2, 6),    # F3-O1 (front-back left)
            (11, 7)    # F4-O2 (front-back right)
        ]
        
        for ch1, ch2 in key_pairs:
            try:
                # Calculate coherence
                f, coh = coherence(trial[ch1], trial[ch2], fs=fs)
                
                # Extract coherence in different frequency bands
                delta_idx = np.logical_and(f >= 1, f <= 4)
                theta_idx = np.logical_and(f >= 4, f <= 8)
                alpha_idx = np.logical_and(f >= 8, f <= 13)
                beta_idx = np.logical_and(f >= 13, f <= 30)
                
                # Calculate mean coherence in each band
                delta_coh = np.mean(coh[delta_idx]) if np.any(delta_idx) else 0
                theta_coh = np.mean(coh[theta_idx]) if np.any(theta_idx) else 0
                alpha_coh = np.mean(coh[alpha_idx]) if np.any(alpha_idx) else 0
                beta_coh = np.mean(coh[beta_idx]) if np.any(beta_idx) else 0
                
                # Add coherence features
                trial_features.extend([delta_coh, theta_coh, alpha_coh, beta_coh])
            except Exception as e:
                print(f"  ⚠️ Coherence calculation failed for channels {ch1}-{ch2}: {str(e)}")
                # Add zeros if coherence calculation fails
                trial_features.extend([0, 0, 0, 0])
        
        # 5. Regional wavelet features
        # Calculate average wavelet energy in different brain regions
        region_features = []
        
        for region_channels in [frontal_channels, temporal_channels, occipital_channels, left_channels, right_channels]:
            try:
                # Calculate average DWT coefficients for the region
                region_coeffs = []
                for ch in region_channels:
                    coeffs = pywt.wavedec(trial[ch], wavelet, level=3)
                    region_coeffs.append(coeffs)
                
                # Calculate energy at each level
                for i in range(4):  # level+1 = 4
                    region_energy = np.mean([np.sum(rc[i]**2) for rc in region_coeffs])
                    region_features.append(region_energy)
            except Exception as e:
                print(f"  ⚠️ Regional feature calculation failed: {str(e)}")
                # Add zeros if calculation fails
                region_features.extend([0] * 4)
        
        # Add regional features
        trial_features.extend(region_features)
        
        wavelet_features.append(trial_features)
    
    wavelet_features = np.array(wavelet_features, dtype=np.float64)
    
    # Check for NaN or infinity values
    if np.isnan(wavelet_features).any() or np.isinf(wavelet_features).any():
        print(f"  ⚠️ Warning: {np.sum(np.isnan(wavelet_features))} NaN and {np.sum(np.isinf(wavelet_features))} Infinity values in features, replacing with zeros")
        wavelet_features = np.nan_to_num(wavelet_features)
    
    print(f"  ✅ Advanced wavelet features extracted: {wavelet_features.shape}")
    
    return wavelet_features, reshaped_data

def visualize_wavelet_features(data, labels, sample_idx=0):
    """Visualize wavelet decomposition for a sample"""
    print(f"\n📊 Visualizing wavelet decomposition for sample {sample_idx}...")
    
    # Get sample data
    sample = data[sample_idx]
    label = labels[sample_idx]
    digit = "6" if label == 0 else "9"
    
    # Reshape to 14 channels x 128 timepoints
    try:
        sample = sample.reshape(14, 128)
    except ValueError:
        print(f"  ⚠️ Reshape failed for sample with length {len(sample)}")
        return
    
    # Select channels for visualization
    channels = {
        "F3": 2,    # Frontal left
        "F4": 11,   # Frontal right
        "O1": 6,    # Occipital left
        "O2": 7     # Occipital right
    }
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Define wavelet
    wavelet = 'db4'
    level = 4
    
    # Plot for each channel
    for i, (ch_name, ch_idx) in enumerate(channels.items()):
        # Get channel data
        channel_data = sample[ch_idx]
        
        # Perform DWT
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)
        
        # Plot original signal
        plt.subplot(len(channels), level+2, i*(level+2) + 1)
        plt.plot(channel_data)
        plt.title(f"{ch_name} - Original Signal")
        if i == len(channels) - 1:
            plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        # Plot approximation coefficients
        plt.subplot(len(channels), level+2, i*(level+2) + 2)
        plt.plot(coeffs[0])
        plt.title(f"{ch_name} - Approximation (A{level})")
        if i == len(channels) - 1:
            plt.xlabel("Coefficient Index")
        
        # Plot detail coefficients
        for j in range(level):
            plt.subplot(len(channels), level+2, i*(level+2) + j + 3)
            plt.plot(coeffs[j+1])
            plt.title(f"{ch_name} - Detail (D{level-j})")
            if i == len(channels) - 1:
                plt.xlabel("Coefficient Index")
    
    plt.suptitle(f"Wavelet Decomposition for Digit {digit}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig(f"wavelet_decomposition_digit{digit}.png")
    print(f"  ✅ Visualization saved as wavelet_decomposition_digit{digit}.png")
    
    # Create CWT scalogram
    plt.figure(figsize=(15, 10))
    
    for i, (ch_name, ch_idx) in enumerate(channels.items()):
        # Get channel data
        channel_data = sample[ch_idx]
        
        # Perform CWT
        scales = np.arange(1, 64)
        cwtmatr, freqs = pywt.cwt(channel_data, scales, 'morl')
        
        # Plot scalogram
        plt.subplot(2, 2, i+1)
        plt.imshow(abs(cwtmatr), aspect='auto', cmap='jet', 
                   extent=[0, len(channel_data), 1, 64])
        plt.colorbar(label='Magnitude')
        plt.title(f"{ch_name} - Scalogram")
        plt.ylabel("Scale")
        plt.xlabel("Time")
    
    plt.suptitle(f"Wavelet Scalogram for Digit {digit}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig(f"wavelet_scalogram_digit{digit}.png")
    print(f"  ✅ Visualization saved as wavelet_scalogram_digit{digit}.png")

def main():
    """Main function"""
    print("🚀 Advanced Wavelet Feature Extraction for EEG Data")
    print("=" * 50)
    
    # Load data
    file_path = "Data/EP1.01.txt"
    data, labels = load_digits_simple(file_path, max_per_digit=500)
    
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Extract advanced wavelet features
    wavelet_features, reshaped_data = extract_advanced_wavelet_features(data)
    
    # Visualize wavelet decomposition for a sample of each class
    digit6_idx = np.where(labels == 0)[0][0]  # First digit 6
    digit9_idx = np.where(labels == 1)[0][0]  # First digit 9
    
    visualize_wavelet_features(reshaped_data, labels, sample_idx=digit6_idx)
    visualize_wavelet_features(reshaped_data, labels, sample_idx=digit9_idx)
    
    print("\n✅ Advanced wavelet feature extraction completed!")
    print(f"  📊 Feature shape: {wavelet_features.shape}")
    
    # Save features for later use
    np.save("advanced_wavelet_features.npy", wavelet_features)
    np.save("labels.npy", labels)
    print("  💾 Features and labels saved to disk")

if __name__ == "__main__":
    main()
