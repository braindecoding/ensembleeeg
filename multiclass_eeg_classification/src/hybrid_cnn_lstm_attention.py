#!/usr/bin/env python3
# hybrid_cnn_lstm_attention.py - Hybrid CNN-LSTM-Attention model for EEG classification

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_data(data, wavelet_features, labels, test_size=0.2, val_size=0.2):
    """Preprocess data for hybrid model"""
    print("\n🔄 Preprocessing data...")

    # Convert to numpy array
    X_raw = np.array(data)
    X_wavelet = np.array(wavelet_features)
    y = np.array(labels)

    # Split into train, validation, and test sets
    X_raw_train_val, X_raw_test, X_wavelet_train_val, X_wavelet_test, y_train_val, y_test = train_test_split(
        X_raw, X_wavelet, y, test_size=test_size, random_state=42, stratify=y
    )

    X_raw_train, X_raw_val, X_wavelet_train, X_wavelet_val, y_train, y_val = train_test_split(
        X_raw_train_val, X_wavelet_train_val, y_train_val,
        test_size=val_size/(1-test_size), random_state=42, stratify=y_train_val
    )

    # Standardize raw data
    for i in range(X_raw_train.shape[0]):
        for j in range(X_raw_train.shape[1]):  # For each channel
            scaler = StandardScaler()
            X_raw_train[i, j, :] = scaler.fit_transform(X_raw_train[i, j, :].reshape(-1, 1)).flatten()

            # Use the same scaler for validation and test data
            if i < X_raw_val.shape[0] and j < X_raw_val.shape[1]:
                X_raw_val[i, j, :] = scaler.transform(X_raw_val[i, j, :].reshape(-1, 1)).flatten()
            if i < X_raw_test.shape[0] and j < X_raw_test.shape[1]:
                X_raw_test[i, j, :] = scaler.transform(X_raw_test[i, j, :].reshape(-1, 1)).flatten()

    # Standardize wavelet features
    scaler_wavelet = StandardScaler()
    X_wavelet_train = scaler_wavelet.fit_transform(X_wavelet_train)
    X_wavelet_val = scaler_wavelet.transform(X_wavelet_val)
    X_wavelet_test = scaler_wavelet.transform(X_wavelet_test)

    # Data augmentation for training set
    print("  🔄 Performing data augmentation...")
    X_raw_train_aug = []
    X_wavelet_train_aug = []
    y_train_aug = []

    for i in range(X_raw_train.shape[0]):
        # Original sample
        X_raw_train_aug.append(X_raw_train[i])
        X_wavelet_train_aug.append(X_wavelet_train[i])
        y_train_aug.append(y_train[i])

        # Add Gaussian noise
        noise_level = 0.1
        noise = np.random.normal(0, noise_level, X_raw_train[i].shape)
        X_raw_train_aug.append(X_raw_train[i] + noise)
        X_wavelet_train_aug.append(X_wavelet_train[i])  # Keep wavelet features the same
        y_train_aug.append(y_train[i])

        # Time shift (shift right by 5 samples)
        shifted = np.zeros_like(X_raw_train[i])
        shifted[:, 5:] = X_raw_train[i][:, :-5]
        X_raw_train_aug.append(shifted)
        X_wavelet_train_aug.append(X_wavelet_train[i])  # Keep wavelet features the same
        y_train_aug.append(y_train[i])

        # Channel dropout (randomly zero out 2 channels)
        channel_dropout = X_raw_train[i].copy()
        dropout_channels = np.random.choice(X_raw_train[i].shape[0], 2, replace=False)
        channel_dropout[dropout_channels, :] = 0
        X_raw_train_aug.append(channel_dropout)
        X_wavelet_train_aug.append(X_wavelet_train[i])  # Keep wavelet features the same
        y_train_aug.append(y_train[i])

    X_raw_train = np.array(X_raw_train_aug)
    X_wavelet_train = np.array(X_wavelet_train_aug)
    y_train = np.array(y_train_aug)

    print(f"  📊 Augmented training set: {X_raw_train.shape}")

    # Convert to PyTorch tensors
    # For CNN, we need shape [batch_size, channels, height, width]
    # Our data is [batch_size, channels, timepoints], so we'll add a dimension
    X_raw_train_tensor = torch.FloatTensor(X_raw_train).unsqueeze(1)
    X_raw_val_tensor = torch.FloatTensor(X_raw_val).unsqueeze(1)
    X_raw_test_tensor = torch.FloatTensor(X_raw_test).unsqueeze(1)

    X_wavelet_train_tensor = torch.FloatTensor(X_wavelet_train)
    X_wavelet_val_tensor = torch.FloatTensor(X_wavelet_val)
    X_wavelet_test_tensor = torch.FloatTensor(X_wavelet_test)

    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)

    print(f"  📊 Training set: X_raw={X_raw_train_tensor.shape}, X_wavelet={X_wavelet_train_tensor.shape}, y={y_train_tensor.shape}")
    print(f"  📊 Validation set: X_raw={X_raw_val_tensor.shape}, X_wavelet={X_wavelet_val_tensor.shape}, y={y_val_tensor.shape}")
    print(f"  📊 Test set: X_raw={X_raw_test_tensor.shape}, X_wavelet={X_wavelet_test_tensor.shape}, y={y_test_tensor.shape}")

    # Create DataLoaders
    train_dataset = TensorDataset(X_raw_train_tensor, X_wavelet_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_raw_val_tensor, X_wavelet_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_raw_test_tensor, X_wavelet_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, y_test, X_wavelet.shape[1]

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important channels"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        attn = self.conv(x)  # [batch_size, 1, height, width]
        attn = torch.sigmoid(attn)

        return x * attn  # Apply attention weights

class TemporalAttention(nn.Module):
    """Temporal attention module for focusing on important time points"""
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attn_weights shape: [batch_size, seq_len, 1]

        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        # context_vector shape: [batch_size, hidden_size]

        return context_vector, attn_weights

class HybridCNNLSTMAttention(nn.Module):
    """Hybrid CNN-LSTM-Attention model for EEG classification"""
    def __init__(self, input_channels=14, seq_length=128, wavelet_features_dim=0, num_classes=2):
        super(HybridCNNLSTMAttention, self).__init__()

        # Model parameters
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.wavelet_features_dim = wavelet_features_dim

        # CNN layers
        self.spatial_attention = SpatialAttention(1)  # Attention on input channels

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 10), padding=(0, 5))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(input_channels, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))

        # Calculate size after CNN
        cnn_output_size = 64 * 1 * (seq_length // 8)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(128*2)  # *2 for bidirectional

        # Fully connected layers for wavelet features
        if wavelet_features_dim > 0:
            self.fc_wavelet1 = nn.Linear(wavelet_features_dim, 128)
            self.bn_wavelet = nn.BatchNorm1d(128)

            # Combined fully connected layers
            self.fc_combined = nn.Linear(128*2 + 128, 64)  # LSTM + wavelet
        else:
            # Without wavelet features
            self.fc_combined = nn.Linear(128*2, 64)

        self.fc_out = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x_raw, x_wavelet=None):
        # x_raw shape: [batch_size, 1, channels, seq_length]

        # Apply spatial attention
        x = self.spatial_attention(x_raw)

        # CNN feature extraction
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Reshape for LSTM
        batch_size = x.size(0)
        seq_len = x.size(3)  # This is the sequence length
        features = x.size(1)  # Number of features

        # Reshape to [batch_size, seq_len, features]
        x = x.view(batch_size, seq_len, features)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Apply temporal attention
        context, _ = self.temporal_attention(lstm_out)

        # Process wavelet features if available
        if x_wavelet is not None and self.wavelet_features_dim > 0:
            x_wavelet = self.fc_wavelet1(x_wavelet)
            x_wavelet = self.bn_wavelet(x_wavelet)
            x_wavelet = F.elu(x_wavelet)
            x_wavelet = self.dropout(x_wavelet)

            # Combine features
            combined = torch.cat((context, x_wavelet), dim=1)
        else:
            combined = context

        # Final classification
        x = self.fc_combined(combined)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc_out(x)

        return x

def train_model(model, train_loader, val_loader, num_epochs=100, weight_decay=1e-4):
    """Train the model with regularization and early stopping"""
    print("\n🚀 Training model...")

    # Loss function and optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs_raw, inputs_wavelet, labels in train_loader:
            inputs_raw, inputs_wavelet, labels = inputs_raw.to(device), inputs_wavelet.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_raw, inputs_wavelet)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs_raw.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs_raw, inputs_wavelet, labels in val_loader:
                inputs_raw, inputs_wavelet, labels = inputs_raw.to(device), inputs_wavelet.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs_raw, inputs_wavelet)
                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item() * inputs_raw.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  ⚠️ Early stopping at epoch {epoch+1}")
            break

        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('hybrid_cnn_lstm_attention_training_history.png')
    print("\n📊 Training history plot saved as 'hybrid_cnn_lstm_attention_training_history.png'")

    return model

def evaluate_model(model, test_loader, y_test):
    """Evaluate the model"""
    print("\n📊 Evaluating model...")

    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs_raw, inputs_wavelet, _ in test_loader:
            inputs_raw, inputs_wavelet = inputs_raw.to(device), inputs_wavelet.to(device)

            # Forward pass
            outputs = model(inputs_raw, inputs_wavelet)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(y_test, all_preds)
    print(f"  ✅ Test accuracy: {accuracy:.4f}")

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, all_preds, target_names=['Digit 6', 'Digit 9']))

    # Confusion matrix
    cm = confusion_matrix(y_test, all_preds)
    print(f"  Confusion Matrix:")
    print(f"  {cm[0][0]:4d} {cm[0][1]:4d} | Digit 6")
    print(f"  {cm[1][0]:4d} {cm[1][1]:4d} | Digit 9")
    print(f"    6    9   <- Predicted")

    # Calculate sensitivity and specificity
    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1])  # True positive rate for digit 6
    specificity = cm[1][1] / (cm[1][0] + cm[1][1])  # True positive rate for digit 9

    print(f"  Sensitivity (Digit 6): {sensitivity:.4f}")
    print(f"  Specificity (Digit 9): {specificity:.4f}")

    return accuracy, all_preds, all_probs

def main():
    """Main function"""
    print("🚀 Hybrid CNN-LSTM-Attention Model for EEG Classification")
    print("=" * 50)

    # Load data and features
    try:
        print("📂 Loading data and features...")
        data = np.load("reshaped_data.npy")
        wavelet_features = np.load("advanced_wavelet_features.npy")
        labels = np.load("labels.npy")
        print(f"  ✅ Data loaded: {data.shape}, Features: {wavelet_features.shape}, Labels: {labels.shape}")
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {str(e)}")
        print("Please run advanced_wavelet_features.py first.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return

    # Preprocess data
    train_loader, val_loader, test_loader, y_test, wavelet_dim = preprocess_data(data, wavelet_features, labels)

    # Build model
    model = HybridCNNLSTMAttention(
        input_channels=14,
        seq_length=128,
        wavelet_features_dim=wavelet_dim,
        num_classes=2
    ).to(device)

    print(model)

    # Train model
    model = train_model(model, train_loader, val_loader, num_epochs=100, weight_decay=1e-4)

    # Evaluate model
    accuracy, predictions, probabilities = evaluate_model(model, test_loader, y_test)

    # Save model
    torch.save(model.state_dict(), 'hybrid_cnn_lstm_attention_model.pth')
    print("\n💾 Model saved as 'hybrid_cnn_lstm_attention_model.pth'")

    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
