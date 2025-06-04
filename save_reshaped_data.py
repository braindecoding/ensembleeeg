#!/usr/bin/env python3
# save_reshaped_data.py - Save reshaped data for hybrid model

import numpy as np
import os

# Load data and labels
print("Loading features and labels...")
wavelet_features = np.load("advanced_wavelet_features.npy")
labels = np.load("labels.npy")

# Load original data
print("Loading original data...")
data = np.load("Data/EP1.01.npy")

# Reshape data to 14 channels x 128 timepoints
print("Reshaping data...")
reshaped_data = []
for trial in data:
    try:
        # Reshape to 14 x 128
        reshaped = trial.reshape(14, 128)
        reshaped_data.append(reshaped)
    except ValueError:
        print(f"  ⚠️ Reshape failed for trial with length {len(trial)}")
        # Create a dummy array with zeros
        reshaped = np.zeros((14, 128))
        reshaped_data.append(reshaped)

reshaped_data = np.array(reshaped_data)
print(f"Reshaped data shape: {reshaped_data.shape}")

# Save reshaped data
np.save("reshaped_data.npy", reshaped_data)
print("Reshaped data saved to disk")
