#!/usr/bin/env python3
# ensemble_model.py - Ensemble model combining multiple classifiers for EEG classification

import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from hybrid_cnn_lstm_attention import HybridCNNLSTMAttention

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_data_for_ensemble(data, wavelet_features, labels, test_size=0.2, val_size=0.2):
    """Preprocess data for ensemble model"""
    print("\n🔄 Preprocessing data for ensemble model...")

    # Convert to numpy array
    X_raw = np.array(data)
    X_wavelet = np.array(wavelet_features)
    y = np.array(labels)

    # Reshape raw data for traditional ML models
    n_samples = X_raw.shape[0]
    X_raw_flat = X_raw.reshape(n_samples, -1)  # Flatten to 2D

    print(f"  📊 Raw data shape: {X_raw.shape}, flattened: {X_raw_flat.shape}")
    print(f"  📊 Wavelet features shape: {X_wavelet.shape}")

    # Split into train, validation, and test sets
    # First split off the test set
    X_raw_train_val, X_raw_test, X_raw_flat_train_val, X_raw_flat_test, X_wavelet_train_val, X_wavelet_test, y_train_val, y_test = train_test_split(
        X_raw, X_raw_flat, X_wavelet, y, test_size=test_size, random_state=42, stratify=y
    )

    # Then split the train_val set into train and validation
    X_raw_train, X_raw_val, X_raw_flat_train, X_raw_flat_val, X_wavelet_train, X_wavelet_val, y_train, y_val = train_test_split(
        X_raw_train_val, X_raw_flat_train_val, X_wavelet_train_val, y_train_val,
        test_size=val_size/(1-test_size), random_state=42, stratify=y_train_val
    )

    # Standardize raw data for deep learning models
    for i in range(X_raw_train.shape[0]):
        for j in range(X_raw_train.shape[1]):  # For each channel
            scaler = StandardScaler()
            X_raw_train[i, j, :] = scaler.fit_transform(X_raw_train[i, j, :].reshape(-1, 1)).flatten()

            # Use the same scaler for validation and test data
            if i < X_raw_val.shape[0] and j < X_raw_val.shape[1]:
                X_raw_val[i, j, :] = scaler.transform(X_raw_val[i, j, :].reshape(-1, 1)).flatten()
            if i < X_raw_test.shape[0] and j < X_raw_test.shape[1]:
                X_raw_test[i, j, :] = scaler.transform(X_raw_test[i, j, :].reshape(-1, 1)).flatten()

    # Standardize flattened raw data for traditional ML models
    scaler_flat = StandardScaler()
    X_raw_flat_train = scaler_flat.fit_transform(X_raw_flat_train)
    X_raw_flat_val = scaler_flat.transform(X_raw_flat_val)
    X_raw_flat_test = scaler_flat.transform(X_raw_flat_test)

    # Standardize wavelet features
    scaler_wavelet = StandardScaler()
    X_wavelet_train = scaler_wavelet.fit_transform(X_wavelet_train)
    X_wavelet_val = scaler_wavelet.transform(X_wavelet_val)
    X_wavelet_test = scaler_wavelet.transform(X_wavelet_test)

    # Combine raw flattened data and wavelet features for traditional ML models
    X_combined_train = np.hstack((X_raw_flat_train, X_wavelet_train))
    X_combined_val = np.hstack((X_raw_flat_val, X_wavelet_val))
    X_combined_test = np.hstack((X_raw_flat_test, X_wavelet_test))

    print(f"  📊 Combined features shape: {X_combined_train.shape}")

    # Convert to PyTorch tensors for deep learning models
    X_raw_train_tensor = torch.FloatTensor(X_raw_train).unsqueeze(1)  # Add channel dimension
    X_raw_val_tensor = torch.FloatTensor(X_raw_val).unsqueeze(1)
    X_raw_test_tensor = torch.FloatTensor(X_raw_test).unsqueeze(1)

    X_wavelet_train_tensor = torch.FloatTensor(X_wavelet_train)
    X_wavelet_val_tensor = torch.FloatTensor(X_wavelet_val)
    X_wavelet_test_tensor = torch.FloatTensor(X_wavelet_test)

    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)

    # Create DataLoaders for deep learning models
    train_dataset = TensorDataset(X_raw_train_tensor, X_wavelet_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_raw_val_tensor, X_wavelet_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_raw_test_tensor, X_wavelet_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Return both traditional ML data and deep learning data
    traditional_ml_data = {
        'X_train': X_combined_train,
        'X_val': X_combined_val,
        'X_test': X_combined_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    deep_learning_data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'y_test': y_test,
        'wavelet_dim': X_wavelet.shape[1]
    }

    return traditional_ml_data, deep_learning_data

def train_traditional_models(data):
    """Train traditional machine learning models"""
    print("\n🤖 Training traditional ML models...")

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    # Train SVM model
    print("  🔍 Training SVM model...")
    svm = SVC(C=20.0, kernel='rbf', gamma=0.01, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_val_acc = accuracy_score(y_val, svm.predict(X_val))
    print(f"  ✅ SVM validation accuracy: {svm_val_acc:.4f}")

    # Train Random Forest model
    print("  🔍 Training Random Forest model...")
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
        print(f"  ✅ Random Forest validation accuracy: {rf_val_acc:.4f}")
    except Exception as e:
        print(f"  ❌ Random Forest training failed: {str(e)}")
        rf = None
        rf_val_acc = 0.0

    # Train Logistic Regression model
    print("  🔍 Training Logistic Regression model...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_val_acc = accuracy_score(y_val, lr.predict(X_val))
    print(f"  ✅ Logistic Regression validation accuracy: {lr_val_acc:.4f}")

    # Create voting ensemble
    print("  🔍 Creating voting ensemble...")
    estimators = [('svm', svm), ('lr', lr)]
    if rf is not None:
        estimators.append(('rf', rf))

    try:
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability estimates for voting
        )
        voting.fit(X_train, y_train)
        voting_val_acc = accuracy_score(y_val, voting.predict(X_val))
        print(f"  ✅ Voting ensemble validation accuracy: {voting_val_acc:.4f}")
    except Exception as e:
        print(f"  ❌ Voting ensemble training failed: {str(e)}")
        voting = None
        voting_val_acc = 0.0

    return {
        'svm': svm,
        'rf': rf,
        'lr': lr,
        'voting': voting
    }

def train_deep_learning_model(data):
    """Train deep learning model"""
    print("\n🧠 Training deep learning model...")

    train_loader = data['train_loader']
    val_loader = data['val_loader']
    wavelet_dim = data['wavelet_dim']

    # Build model
    model = HybridCNNLSTMAttention(
        input_channels=14,
        seq_length=128,
        wavelet_features_dim=wavelet_dim,
        num_classes=2
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(50):  # Reduced epochs for faster training
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs_raw.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

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
        print(f"Epoch {epoch+1}/50: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    return model

def get_model_predictions(models, dl_model, trad_data, dl_data):
    """Get predictions from all models for stacking"""
    print("\n🔮 Getting predictions from all models...")

    X_train = trad_data['X_train']
    y_train = trad_data['y_train']
    X_val = trad_data['X_val']
    y_val = trad_data['y_val']
    X_test = trad_data['X_test']
    y_test = trad_data['y_test']

    test_loader = dl_data['test_loader']

    # Get predictions from traditional models
    all_train_probas = []
    all_val_probas = []
    all_test_probas = []

    # SVM predictions
    try:
        svm_train_proba = models['svm'].predict_proba(X_train)
        svm_val_proba = models['svm'].predict_proba(X_val)
        svm_test_proba = models['svm'].predict_proba(X_test)

        all_train_probas.append(svm_train_proba)
        all_val_probas.append(svm_val_proba)
        all_test_probas.append(svm_test_proba)
    except Exception as e:
        print(f"  ⚠️ SVM prediction failed: {str(e)}")

    # RF predictions
    if models['rf'] is not None:
        try:
            rf_train_proba = models['rf'].predict_proba(X_train)
            rf_val_proba = models['rf'].predict_proba(X_val)
            rf_test_proba = models['rf'].predict_proba(X_test)

            all_train_probas.append(rf_train_proba)
            all_val_probas.append(rf_val_proba)
            all_test_probas.append(rf_test_proba)
        except Exception as e:
            print(f"  ⚠️ RF prediction failed: {str(e)}")

    # LR predictions
    try:
        lr_train_proba = models['lr'].predict_proba(X_train)
        lr_val_proba = models['lr'].predict_proba(X_val)
        lr_test_proba = models['lr'].predict_proba(X_test)

        all_train_probas.append(lr_train_proba)
        all_val_probas.append(lr_val_proba)
        all_test_probas.append(lr_test_proba)
    except Exception as e:
        print(f"  ⚠️ LR prediction failed: {str(e)}")

    # Voting predictions
    if models['voting'] is not None:
        try:
            voting_train_proba = models['voting'].predict_proba(X_train)
            voting_val_proba = models['voting'].predict_proba(X_val)
            voting_test_proba = models['voting'].predict_proba(X_test)

            all_train_probas.append(voting_train_proba)
            all_val_probas.append(voting_val_proba)
            all_test_probas.append(voting_test_proba)
        except Exception as e:
            print(f"  ⚠️ Voting prediction failed: {str(e)}")

    # Get predictions from deep learning model
    dl_model.eval()
    dl_train_proba = []
    dl_val_proba = []
    dl_test_proba = []

    with torch.no_grad():
        # Get train predictions
        for inputs_raw, inputs_wavelet, _ in dl_data['train_loader']:
            inputs_raw, inputs_wavelet = inputs_raw.to(device), inputs_wavelet.to(device)
            outputs = dl_model(inputs_raw, inputs_wavelet)
            probs = F.softmax(outputs, dim=1)
            dl_train_proba.extend(probs.cpu().numpy())

        # Get validation predictions
        for inputs_raw, inputs_wavelet, _ in dl_data['val_loader']:
            inputs_raw, inputs_wavelet = inputs_raw.to(device), inputs_wavelet.to(device)
            outputs = dl_model(inputs_raw, inputs_wavelet)
            probs = F.softmax(outputs, dim=1)
            dl_val_proba.extend(probs.cpu().numpy())

        # Get test predictions
        for inputs_raw, inputs_wavelet, _ in dl_data['test_loader']:
            inputs_raw, inputs_wavelet = inputs_raw.to(device), inputs_wavelet.to(device)
            outputs = dl_model(inputs_raw, inputs_wavelet)
            probs = F.softmax(outputs, dim=1)
            dl_test_proba.extend(probs.cpu().numpy())

    dl_train_proba = np.array(dl_train_proba)
    dl_val_proba = np.array(dl_val_proba)
    dl_test_proba = np.array(dl_test_proba)

    # Add DL predictions to the list
    all_train_probas.append(dl_train_proba)
    all_val_probas.append(dl_val_proba)
    all_test_probas.append(dl_test_proba)

    # Combine predictions for meta-model
    meta_train_features = np.hstack(all_train_probas)
    meta_val_features = np.hstack(all_val_probas)
    meta_test_features = np.hstack(all_test_probas)

    return {
        'meta_train_features': meta_train_features,
        'meta_val_features': meta_val_features,
        'meta_test_features': meta_test_features,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def train_meta_model(meta_data):
    """Train meta-model (stacking)"""
    print("\n🔝 Training meta-model (stacking)...")

    # Train meta-model
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_data['meta_train_features'], meta_data['y_train'])

    # Evaluate on validation set
    meta_val_acc = accuracy_score(meta_data['y_val'], meta_model.predict(meta_data['meta_val_features']))
    print(f"  ✅ Meta-model validation accuracy: {meta_val_acc:.4f}")

    # Evaluate on test set
    meta_test_pred = meta_model.predict(meta_data['meta_test_features'])
    meta_test_acc = accuracy_score(meta_data['y_test'], meta_test_pred)
    print(f"  ✅ Meta-model test accuracy: {meta_test_acc:.4f}")

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(meta_data['y_test'], meta_test_pred, target_names=['Digit 6', 'Digit 9']))

    # Confusion matrix
    cm = confusion_matrix(meta_data['y_test'], meta_test_pred)
    print(f"  Confusion Matrix:")
    print(f"  {cm[0][0]:4d} {cm[0][1]:4d} | Digit 6")
    print(f"  {cm[1][0]:4d} {cm[1][1]:4d} | Digit 9")
    print(f"    6    9   <- Predicted")

    # Calculate sensitivity and specificity
    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1])  # True positive rate for digit 6
    specificity = cm[1][1] / (cm[1][0] + cm[1][1])  # True positive rate for digit 9

    print(f"  Sensitivity (Digit 6): {sensitivity:.4f}")
    print(f"  Specificity (Digit 9): {specificity:.4f}")

    return meta_model, meta_test_acc

def main():
    """Main function"""
    print("🚀 Ensemble Model for EEG Classification")
    print("=" * 50)

    # Load data and features
    try:
        print("📂 Loading data and features...")
        data = np.load("reshaped_data.npy")
        wavelet_features = np.load("advanced_wavelet_features.npy")
        labels = np.load("labels.npy")
        print(f"  ✅ Data loaded: {data.shape}, Features: {wavelet_features.shape}, Labels: {labels.shape}")
    except FileNotFoundError:
        print("❌ Data files not found. Please run advanced_wavelet_features.py first.")
        return

    # Preprocess data
    traditional_ml_data, deep_learning_data = preprocess_data_for_ensemble(data, wavelet_features, labels)

    # Train traditional ML models
    traditional_models = train_traditional_models(traditional_ml_data)

    # Train deep learning model
    dl_model = train_deep_learning_model(deep_learning_data)

    # Get predictions from all models
    meta_data = get_model_predictions(traditional_models, dl_model, traditional_ml_data, deep_learning_data)

    # Train meta-model
    meta_model, meta_acc = train_meta_model(meta_data)

    # Save models
    print("\n💾 Saving models...")
    try:
        joblib.dump(traditional_models, 'traditional_models.pkl')
        torch.save(dl_model.state_dict(), 'dl_model.pth')
        joblib.dump(meta_model, 'meta_model.pkl')
        print("  ✅ Models saved successfully")
    except Exception as e:
        print(f"  ❌ Error saving models: {str(e)}")

    print("\n✅ Ensemble model training completed!")
    print(f"  📊 Final ensemble accuracy: {meta_acc:.4f}")

if __name__ == "__main__":
    main()
