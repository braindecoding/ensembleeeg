#!/usr/bin/env python3
# convert_data.py - Convert original data to numpy format

import numpy as np
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

# Load data
file_path = "Data/EP1.01.txt"
data, labels = load_digits_simple(file_path, max_per_digit=500)

# Save data and labels
if data is not None:
    os.makedirs("Data", exist_ok=True)
    np.save("Data/EP1.01.npy", data)
    np.save("Data/EP1.01_labels.npy", labels)
    print("Data and labels saved to disk")
else:
    print("Failed to load data")
