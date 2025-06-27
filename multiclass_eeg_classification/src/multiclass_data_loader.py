#!/usr/bin/env python3
# multiclass_data_loader.py - Load and prepare multi-class EEG data (0-9)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

def load_multiclass_data(data_path="../data/Data/EP1.01.txt", max_samples_per_digit=100):
    """
    Load multi-class EEG data for digits 0-9
    
    Args:
        data_path: Path to MindBigData file
        max_samples_per_digit: Maximum samples per digit class
    
    Returns:
        X: EEG data array
        y: Labels (0-9)
        metadata: Additional information
    """
    print("🔄 Loading Multi-class EEG Data (Digits 0-9)")
    print("=" * 50)
    
    all_data = []
    all_labels = []
    metadata = {'channels': [], 'sample_counts': {}}
    
    # Initialize counters for each digit
    digit_counts = {i: 0 for i in range(10)}
    
    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"  Processing line {line_num}...")
            
            if not line.strip():
                continue
                
            try:
                parts = line.split('\t')
                if len(parts) >= 7:
                    digit = int(parts[4])
                    channel = parts[3]
                    data_string = parts[6]
                    
                    # Only process digits 0-9
                    if digit in range(10):
                        # Check if we need more samples for this digit
                        if digit_counts[digit] < max_samples_per_digit:
                            # Parse EEG data
                            values = [float(x.strip()) for x in data_string.split(',') if x.strip()]
                            
                            if len(values) > 100:  # Minimum length requirement
                                # Normalize length to 256 samples
                                if len(values) >= 256:
                                    normalized_values = values[:256]
                                else:
                                    # Pad with zeros if too short
                                    normalized_values = values + [0.0] * (256 - len(values))

                                all_data.append(normalized_values)
                                all_labels.append(digit)
                                digit_counts[digit] += 1
                                
                                if channel not in metadata['channels']:
                                    metadata['channels'].append(channel)
                    
                    # Stop if we have enough samples for all digits
                    if all(count >= max_samples_per_digit for count in digit_counts.values()):
                        break
                        
            except (ValueError, IndexError) as e:
                continue
    
    # Convert to numpy arrays
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Store sample counts
    metadata['sample_counts'] = digit_counts
    
    print(f"\n📊 Data Loading Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1] if len(X) > 0 else 0}")
    print(f"  Classes: {len(np.unique(y))}")
    
    for digit in range(10):
        count = np.sum(y == digit)
        print(f"  Digit {digit}: {count} samples")
    
    return X, y, metadata

def preprocess_multiclass_data(X, y, target_length=128):
    """
    Preprocess multi-class EEG data
    
    Args:
        X: Raw EEG data
        y: Labels
        target_length: Target sequence length
    
    Returns:
        X_processed: Processed EEG data
        y_processed: Processed labels
    """
    print(f"\n🔧 Preprocessing Multi-class Data")
    print("=" * 50)
    
    X_processed = []
    y_processed = []
    
    for i, (sample, label) in enumerate(zip(X, y)):
        # Normalize length
        if len(sample) >= target_length:
            # Truncate
            processed_sample = sample[:target_length]
        else:
            # Pad with zeros
            processed_sample = np.pad(sample, (0, target_length - len(sample)), 'constant')
        
        # Normalize amplitude
        processed_sample = (processed_sample - np.mean(processed_sample)) / (np.std(processed_sample) + 1e-8)
        
        X_processed.append(processed_sample)
        y_processed.append(label)
    
    X_processed = np.array(X_processed)
    y_processed = np.array(y_processed)
    
    print(f"  Processed shape: {X_processed.shape}")
    print(f"  Label distribution: {np.bincount(y_processed)}")
    
    return X_processed, y_processed

def create_multiclass_splits(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Create train/validation/test splits for multi-class data"""
    print(f"\n📊 Creating Multi-class Data Splits")
    print("=" * 50)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples") 
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Check class distribution
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {split_name} distribution: {np.bincount(y_split)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main function to demonstrate multi-class data loading"""
    # Load data
    X, y, metadata = load_multiclass_data()
    
    if len(X) > 0:
        # Preprocess
        X_processed, y_processed = preprocess_multiclass_data(X, y)
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_multiclass_splits(X_processed, y_processed)
        
        # Save processed data
        np.save("multiclass_data.npy", X_processed)
        np.save("multiclass_labels.npy", y_processed)
        
        print(f"\n✅ Multi-class data preparation completed!")
        print(f"📁 Saved: multiclass_data.npy, multiclass_labels.npy")
    else:
        print("❌ No data loaded. Please check the data path.")

if __name__ == "__main__":
    main()
