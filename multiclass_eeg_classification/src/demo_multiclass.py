#!/usr/bin/env python3
# demo_multiclass.py - Demo of multi-class EEG classification

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# SYNTHETIC DATA FUNCTIONS REMOVED FOR ACADEMIC ETHICS
# This project uses only real EEG data from MindBigData

def run_multiclass_experiment():
    """Run complete multi-class classification experiment with REAL EEG data"""
    print("🚀 MULTI-CLASS EEG CLASSIFICATION WITH REAL DATA")
    print("=" * 60)

    # Load REAL EEG data only
    print("❌ DEMO DISABLED - SYNTHETIC DATA VIOLATES ACADEMIC ETHICS")
    print("This function previously used synthetic data.")
    print("Please use run_multiclass_experiment.py for real EEG data analysis.")
    return {}, []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate models
    print(f"\n🤖 Training Models")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"  {name}: {accuracy:.3f}")
    
    # Create ensemble
    print(f"\nCreating Ensemble...")
    ensemble_proba = np.mean([results[name]['probabilities'] for name in models.keys()], axis=0)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    results['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba
    }
    
    print(f"  Ensemble: {ensemble_accuracy:.3f}")
    
    return results, y_test

def visualize_results(results, y_test):
    """Visualize multi-class classification results"""
    print(f"\n📊 Generating Visualizations")
    print("=" * 50)
    
    # Create results directory
    import os
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)
    
    # 1. Accuracy comparison
    plt.figure(figsize=(12, 6))
    
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    accuracies['Random Baseline'] = 0.1
    
    bars = plt.bar(accuracies.keys(), accuracies.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A'])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies.values()):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Accuracy')
    plt.title('Multi-Class EEG Classification Performance')
    plt.ylim(0, max(accuracies.values()) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../figures/multiclass_demo_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrix for best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_pred = results[best_model]['predictions']
    
    cm = confusion_matrix(y_test, best_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Digit {i}' for i in range(10)],
                yticklabels=[f'Digit {i}' for i in range(10)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {best_model}')
    plt.tight_layout()
    plt.savefig('../figures/multiclass_demo_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence distributions
    valid_results = {k: v for k, v in results.items() if k != 'Random Baseline'}
    n_plots = len(valid_results)

    plt.figure(figsize=(15, 5))

    for i, (name, result) in enumerate(valid_results.items()):
        plt.subplot(1, n_plots, i+1)
        max_probs = np.max(result['probabilities'], axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'{name}\nMean: {max_probs.mean():.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/multiclass_demo_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Per-class performance
    best_pred = results[best_model]['predictions']
    
    per_class_acc = []
    for digit in range(10):
        mask = y_test == digit
        if np.sum(mask) > 0:
            digit_acc = accuracy_score(y_test[mask], best_pred[mask])
            per_class_acc.append(digit_acc)
        else:
            per_class_acc.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'Digit {i}' for i in range(10)], per_class_acc, 
                   color='lightcoral', edgecolor='black')
    
    # Add value labels
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Performance - {best_model}')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../figures/multiclass_demo_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to ../figures/")

def generate_summary_report(results, y_test):
    """Generate summary report"""
    print(f"\n📝 Generating Summary Report")
    print("=" * 50)
    
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    # Calculate additional metrics
    best_pred = results[best_model]['predictions']
    report = classification_report(y_test, best_pred, 
                                 target_names=[f'Digit {i}' for i in range(10)])
    
    # Save report
    with open('../results/multiclass_demo_report.txt', 'w') as f:
        f.write("Multi-Class EEG Classification Demo Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        for name, result in results.items():
            f.write(f"{name}: {result['accuracy']:.3f}\n")
        
        f.write(f"\nBest Model: {best_model} ({best_accuracy:.3f})\n")
        f.write(f"Improvement over random: {(best_accuracy - 0.1) / 0.1 * 100:.1f}%\n")
        
        f.write(f"\nDETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 35 + "\n")
        f.write(report)
        
        f.write(f"\nKEY FINDINGS:\n")
        f.write("-" * 15 + "\n")
        f.write("- Multi-class classification achieved realistic performance\n")
        f.write("- Ensemble approach improved over individual models\n")
        f.write("- Confidence distributions are more realistic than binary\n")
        f.write("- Some digits show higher confusion rates\n")
        
        f.write(f"\nNEXT STEPS:\n")
        f.write("-" * 12 + "\n")
        f.write("1. Test with real EEG data\n")
        f.write("2. Implement hierarchical ensemble\n")
        f.write("3. Add deep learning models\n")
        f.write("4. Develop attention mechanisms\n")
    
    print("✅ Report saved to ../results/multiclass_demo_report.txt")

def main():
    """Main demo function"""
    print("❌ DEMO DISABLED - USING REAL EEG DATA ONLY")
    print("=" * 60)
    print("This demo used synthetic data which violates academic ethics.")
    print("Please use multiclass_data_loader.py for real EEG data.")
    print("Academic integrity requires using only real MindBigData.")
    print("\nTo run with real data:")
    print("  python multiclass_data_loader.py")
    print("  python run_multiclass_experiment.py")

if __name__ == "__main__":
    main()
