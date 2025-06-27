#!/usr/bin/env python3
# real_data_analysis.py - Analysis using ONLY real EEG data from MindBigData

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from multiclass_data_loader import load_multiclass_data, preprocess_multiclass_data, create_multiclass_splits
from hierarchical_ensemble import HierarchicalEnsemble, ConfidenceBasedEnsemble, evaluate_multiclass_ensemble

def analyze_real_eeg_data():
    """Analyze real EEG data for multi-class classification"""
    print("🧠 REAL EEG DATA ANALYSIS - MULTI-CLASS CLASSIFICATION")
    print("=" * 70)
    print("📋 ACADEMIC ETHICS: Using ONLY real MindBigData EEG signals")
    print("🚫 NO synthetic data - maintaining research integrity")
    print("=" * 70)
    
    # Load real EEG data
    print("\n📊 Loading Real EEG Data")
    print("=" * 50)
    
    data_path = "../data/Data/EP1.01.txt"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return {}, []
    
    # Load multi-class data (digits 0-9)
    X, y, metadata = load_multiclass_data(data_path, max_samples_per_digit=150)
    
    if len(X) == 0:
        print("❌ No real EEG data loaded")
        return {}, []
    
    print(f"✅ Loaded {len(X)} real EEG samples")
    print(f"📊 Classes: {np.unique(y)}")
    print(f"📈 Distribution: {dict(zip(np.unique(y), np.bincount(y)))}")
    
    # Preprocess real data
    X_processed, y_processed = preprocess_multiclass_data(X, y)
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_multiclass_splits(X_processed, y_processed)
    
    return analyze_multiclass_performance(X_train, X_val, X_test, y_train, y_val, y_test)

def analyze_multiclass_performance(X_train, X_val, X_test, y_train, y_val, y_test):
    """Analyze multi-class performance with real EEG data"""
    print(f"\n🔬 ANALYZING MULTI-CLASS PERFORMANCE")
    print("=" * 50)
    
    # Combine train and validation for final training
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    
    # Flatten and scale data
    X_trainval_flat = X_trainval.reshape(X_trainval.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    results = {}
    
    # Test individual models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42, C=1.0)
    }
    
    print("🤖 Training Individual Models on Real EEG Data")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_trainval_scaled, y_trainval)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"  ✅ {name}: {accuracy:.3f}")
    
    # Test Hierarchical Ensemble
    print(f"\n🌳 Testing Hierarchical Ensemble")
    print("-" * 50)
    
    hierarchical = HierarchicalEnsemble('rf')
    hierarchical.fit(X_trainval_scaled, y_trainval)
    
    hier_acc, hier_report, hier_pred, hier_proba = evaluate_multiclass_ensemble(
        hierarchical, X_test_scaled, y_test
    )
    
    results['Hierarchical Ensemble'] = {
        'accuracy': hier_acc,
        'predictions': hier_pred,
        'probabilities': hier_proba
    }
    
    # Test Confidence-Based Ensemble
    print(f"\n🎯 Testing Confidence-Based Ensemble")
    print("-" * 50)
    
    confidence_ensemble = ConfidenceBasedEnsemble()
    confidence_ensemble.fit(X_trainval_scaled, y_trainval)
    
    conf_acc, conf_report, conf_pred, conf_proba = evaluate_multiclass_ensemble(
        confidence_ensemble, X_test_scaled, y_test
    )
    
    results['Confidence Ensemble'] = {
        'accuracy': conf_acc,
        'predictions': conf_pred,
        'probabilities': conf_proba
    }
    
    return results, y_test

def generate_real_data_visualizations(results, y_test):
    """Generate visualizations for real EEG data analysis"""
    print(f"\n📊 Generating Real Data Visualizations")
    print("=" * 50)
    
    # Create directories
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../figures", exist_ok=True)
    
    # 1. Performance comparison
    plt.figure(figsize=(12, 8))
    
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    accuracies['Random Baseline'] = 0.1  # 10% for 10-class
    
    # Sort by accuracy
    sorted_items = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    names, accs = zip(*sorted_items)
    
    colors = ['#2E8B57' if acc > 0.1 else '#DC143C' for acc in accs]
    bars = plt.bar(names, accs, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Multi-Class EEG Classification Performance\n(Real MindBigData)', fontsize=14, fontweight='bold')
    plt.ylim(0, max(accs) * 1.15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add improvement annotation
    best_acc = max([acc for acc in accs if acc > 0.1])
    improvement = (best_acc - 0.1) / 0.1 * 100
    plt.text(0.02, 0.98, f'Best improvement over random: {improvement:.1f}%', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('../figures/real_multiclass_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Real data visualizations saved")

def save_real_data_report(results, y_test):
    """Save comprehensive report for real EEG data analysis"""
    print(f"\n📝 Saving Real Data Analysis Report")
    print("=" * 50)
    
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    with open('../results/real_multiclass_analysis.txt', 'w') as f:
        f.write("REAL EEG DATA MULTI-CLASS CLASSIFICATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ACADEMIC ETHICS COMPLIANCE:\n")
        f.write("- Uses ONLY real EEG data from MindBigData\n")
        f.write("- NO synthetic data used\n")
        f.write("- Maintains research integrity\n\n")
        
        f.write("PERFORMANCE RESULTS:\n")
        f.write("-" * 25 + "\n")
        for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            f.write(f"{name}: {result['accuracy']:.3f}\n")
        
        f.write(f"\nBest Model: {best_model}\n")
        f.write(f"Best Accuracy: {best_accuracy:.3f}\n")
        f.write(f"Random Baseline: 0.100\n")
        f.write(f"Improvement: {(best_accuracy - 0.1) / 0.1 * 100:.1f}%\n")
    
    print("✅ Real data analysis report saved")

def main():
    """Main function for real EEG data analysis"""
    # Analyze real data
    results, y_test = analyze_real_eeg_data()
    
    if not results:
        print("❌ Analysis failed - no real data available")
        return
    
    # Generate visualizations
    generate_real_data_visualizations(results, y_test)
    
    # Save report
    save_real_data_report(results, y_test)
    
    print(f"\n🎉 REAL EEG DATA ANALYSIS COMPLETED!")
    print("=" * 60)
    print("📁 Results saved to:")
    print("  - ../results/real_multiclass_analysis.txt")
    print("  - ../figures/real_multiclass_performance.png")
    
    # Print summary
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    print(f"\n🏆 SUMMARY:")
    print(f"  Best Model: {best_model}")
    print(f"  Best Accuracy: {best_accuracy:.3f}")
    print(f"  Improvement over random: {(best_accuracy - 0.1) / 0.1 * 100:.1f}%")
    print(f"  Academic Ethics: ✅ REAL DATA ONLY")

if __name__ == "__main__":
    main()
