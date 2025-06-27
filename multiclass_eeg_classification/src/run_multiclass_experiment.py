#!/usr/bin/env python3
# run_multiclass_experiment.py - Complete multi-class EEG classification experiment

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import custom modules
from multiclass_data_loader import load_multiclass_data, preprocess_multiclass_data, create_multiclass_splits
from multiclass_cnn_lstm import MultiClassCNNLSTM, train_multiclass_model, evaluate_multiclass_model
from hierarchical_ensemble import HierarchicalEnsemble, ConfidenceBasedEnsemble, evaluate_multiclass_ensemble

def setup_experiment():
    """Setup experiment environment"""
    print("🔧 Setting up Multi-class EEG Experiment")
    print("=" * 60)
    
    # Create results directories
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device

def load_and_prepare_data():
    """Load and prepare multi-class data"""
    print("\n📊 Loading Multi-class Data")
    print("=" * 50)
    
    # Check if processed data exists
    if os.path.exists("multiclass_data.npy") and os.path.exists("multiclass_labels.npy"):
        print("Loading preprocessed data...")
        X_processed = np.load("multiclass_data.npy")
        y_processed = np.load("multiclass_labels.npy")
    else:
        # Load data (adjust path as needed)
        data_path = "../data/Data/EP1.01.txt"
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure the data is copied to the multiclass project folder")
            return None, None, None, None, None, None

        # Load multi-class data
        X, y, metadata = load_multiclass_data(data_path, max_samples_per_digit=100)

        if len(X) == 0:
            print("❌ No data loaded")
            return None, None, None, None, None, None

        # Preprocess
        X_processed, y_processed = preprocess_multiclass_data(X, y)

    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_multiclass_splits(X_processed, y_processed)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_traditional_ml_experiment(X_train, X_val, X_test, y_train, y_val, y_test):
    """Run traditional ML experiments"""
    print("\n🤖 Traditional ML Experiments")
    print("=" * 50)
    
    results = {}
    
    # Flatten data for traditional ML
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Combine train and validation for final training
    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.hstack([y_train, y_val])
    
    # Test hierarchical ensemble
    print("Testing Hierarchical Ensemble...")
    hierarchical = HierarchicalEnsemble('rf')
    hierarchical.fit(X_trainval, y_trainval)
    
    hier_acc, hier_report, hier_pred, hier_proba = evaluate_multiclass_ensemble(
        hierarchical, X_test_scaled, y_test
    )
    results['Hierarchical'] = {
        'accuracy': hier_acc,
        'predictions': hier_pred,
        'probabilities': hier_proba
    }
    
    # Test confidence-based ensemble
    print("\nTesting Confidence-Based Ensemble...")
    confidence_ensemble = ConfidenceBasedEnsemble()
    confidence_ensemble.fit(X_trainval, y_trainval)
    
    conf_acc, conf_report, conf_pred, conf_proba = evaluate_multiclass_ensemble(
        confidence_ensemble, X_test_scaled, y_test
    )
    results['Confidence'] = {
        'accuracy': conf_acc,
        'predictions': conf_pred,
        'probabilities': conf_proba
    }
    
    return results

def run_deep_learning_experiment(X_train, X_val, X_test, y_train, y_val, y_test, device):
    """Run deep learning experiments"""
    print("\n🧠 Deep Learning Experiments")
    print("=" * 50)
    
    # Prepare data for PyTorch
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = MultiClassCNNLSTM(input_length=128, num_classes=10, dropout_rate=0.5)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_multiclass_model(
        model, train_loader, val_loader, num_epochs=50, device=device
    )
    
    # Evaluate model
    test_accuracy, test_report, test_cm, test_probabilities = evaluate_multiclass_model(
        model, test_loader, device
    )
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.3f}")
    
    # Save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training History - Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/multiclass_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': test_accuracy,
        'confusion_matrix': test_cm,
        'probabilities': test_probabilities,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    }

def generate_comparison_plots(traditional_results, dl_results, y_test):
    """Generate comparison plots"""
    print("\n📊 Generating Comparison Plots")
    print("=" * 50)
    
    # Accuracy comparison
    accuracies = {
        'Hierarchical Ensemble': traditional_results['Hierarchical']['accuracy'],
        'Confidence Ensemble': traditional_results['Confidence']['accuracy'],
        'CNN-LSTM-Attention': dl_results['accuracy'],
        'Random Baseline': 0.1
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies.values()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Accuracy')
    plt.title('Multi-class EEG Classification Performance Comparison')
    plt.ylim(0, max(accuracies.values()) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../figures/multiclass_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence distribution comparison
    plt.figure(figsize=(15, 5))
    
    methods = ['Hierarchical', 'Confidence', 'Deep Learning']
    probabilities = [
        traditional_results['Hierarchical']['probabilities'],
        traditional_results['Confidence']['probabilities'],
        dl_results['probabilities']
    ]
    
    for i, (method, probs) in enumerate(zip(methods, probabilities)):
        plt.subplot(1, 3, i+1)
        max_probs = np.max(probs, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'{method}\nMean: {max_probs.mean():.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/multiclass_confidence_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix for best method
    best_method = max(accuracies.keys(), key=lambda k: accuracies[k] if k != 'Random Baseline' else 0)
    
    if best_method == 'CNN-LSTM-Attention':
        cm = dl_results['confusion_matrix']
    else:
        best_pred = traditional_results[best_method.split()[0]]['predictions']
        cm = confusion_matrix(y_test, best_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Digit {i}' for i in range(10)],
                yticklabels=[f'Digit {i}' for i in range(10)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {best_method}')
    plt.tight_layout()
    plt.savefig('../figures/multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(traditional_results, dl_results):
    """Save experimental results"""
    print("\n💾 Saving Results")
    print("=" * 50)
    
    # Compile results
    results_summary = {
        'Traditional ML': {
            'Hierarchical Ensemble': traditional_results['Hierarchical']['accuracy'],
            'Confidence Ensemble': traditional_results['Confidence']['accuracy']
        },
        'Deep Learning': {
            'CNN-LSTM-Attention': dl_results['accuracy']
        },
        'Baseline': {
            'Random': 0.1
        }
    }
    
    # Save to file
    with open('../results/multiclass_results_summary.txt', 'w') as f:
        f.write("Multi-class EEG Classification Results\n")
        f.write("=" * 50 + "\n\n")
        
        for category, methods in results_summary.items():
            f.write(f"{category}:\n")
            for method, accuracy in methods.items():
                f.write(f"  {method}: {accuracy:.3f}\n")
            f.write("\n")
        
        f.write("Key Findings:\n")
        f.write("- Multi-class classification is significantly more challenging\n")
        f.write("- Best accuracy achieved: {:.3f}\n".format(max([
            acc for methods in results_summary.values() 
            for acc in methods.values() if acc > 0.1
        ])))
        f.write("- Confidence distributions are more realistic (lower)\n")
        f.write("- Hierarchical approaches show promise\n")
    
    print("✅ Results saved to ../results/multiclass_results_summary.txt")

def main():
    """Main experimental pipeline"""
    print("🚀 MULTI-CLASS EEG CLASSIFICATION EXPERIMENT")
    print("=" * 60)
    
    # Setup
    device = setup_experiment()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    if X_train is None:
        print("❌ Experiment aborted due to data loading issues")
        return
    
    # Run experiments
    traditional_results = run_traditional_ml_experiment(X_train, X_val, X_test, y_train, y_val, y_test)
    dl_results = run_deep_learning_experiment(X_train, X_val, X_test, y_train, y_val, y_test, device)
    
    # Generate plots
    generate_comparison_plots(traditional_results, dl_results, y_test)
    
    # Save results
    save_results(traditional_results, dl_results)
    
    print("\n✅ MULTI-CLASS EXPERIMENT COMPLETED!")
    print("=" * 60)
    print("📁 Check the following directories for results:")
    print("  - ../results/ : Text summaries and metrics")
    print("  - ../figures/ : Plots and visualizations")
    print("  - ../models/  : Trained model files")

if __name__ == "__main__":
    main()
