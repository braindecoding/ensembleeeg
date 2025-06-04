#!/usr/bin/env python3
# demo_prediction.py - Demonstrate how to use trained models for prediction

import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from hybrid_cnn_lstm_attention import HybridCNNLSTMAttention

def load_trained_models():
    """Load all trained models"""
    print("🔄 Loading trained models...")
    
    models = {}
    
    # Load traditional ML models
    try:
        models['traditional'] = joblib.load('traditional_models.pkl')
        print("  ✅ Traditional ML models loaded")
    except FileNotFoundError:
        print("  ❌ Traditional ML models not found")
        models['traditional'] = None
    
    # Load meta model
    try:
        models['meta'] = joblib.load('meta_model.pkl')
        print("  ✅ Meta model loaded")
    except FileNotFoundError:
        print("  ❌ Meta model not found")
        models['meta'] = None
    
    # Load deep learning model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load hybrid CNN-LSTM-Attention model
        hybrid_model = HybridCNNLSTMAttention(
            input_channels=14,
            seq_length=128,
            wavelet_features_dim=768,
            num_classes=2
        ).to(device)
        
        hybrid_model.load_state_dict(torch.load('hybrid_cnn_lstm_attention_model.pth', map_location=device))
        hybrid_model.eval()
        models['hybrid'] = hybrid_model
        print("  ✅ Hybrid CNN-LSTM-Attention model loaded")
        
        # Load ensemble deep learning model
        ensemble_dl_model = HybridCNNLSTMAttention(
            input_channels=14,
            seq_length=128,
            wavelet_features_dim=768,
            num_classes=2
        ).to(device)
        
        ensemble_dl_model.load_state_dict(torch.load('dl_model.pth', map_location=device))
        ensemble_dl_model.eval()
        models['ensemble_dl'] = ensemble_dl_model
        print("  ✅ Ensemble deep learning model loaded")
        
    except FileNotFoundError:
        print("  ❌ Deep learning models not found")
        models['hybrid'] = None
        models['ensemble_dl'] = None
    except Exception as e:
        print(f"  ❌ Error loading deep learning models: {e}")
        models['hybrid'] = None
        models['ensemble_dl'] = None
    
    return models

def preprocess_sample(raw_data, wavelet_features):
    """Preprocess a single sample for prediction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure correct shape
    if raw_data.shape != (14, 128):
        print(f"  ⚠️ Warning: Expected shape (14, 128), got {raw_data.shape}")
        return None, None, None
    
    # Standardize raw data (simple standardization)
    raw_data_std = (raw_data - raw_data.mean()) / (raw_data.std() + 1e-8)
    
    # Prepare for traditional ML (flatten)
    raw_flat = raw_data.flatten()
    combined_features = np.concatenate([raw_flat, wavelet_features])
    
    # Prepare for deep learning
    raw_tensor = torch.FloatTensor(raw_data_std).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 14, 128]
    wavelet_tensor = torch.FloatTensor(wavelet_features).unsqueeze(0).to(device)  # [1, 768]
    
    return combined_features, raw_tensor, wavelet_tensor

def predict_with_models(models, raw_data, wavelet_features):
    """Make predictions using all available models"""
    print("🔮 Making predictions...")
    
    # Preprocess data
    combined_features, raw_tensor, wavelet_tensor = preprocess_sample(raw_data, wavelet_features)
    
    if combined_features is None:
        return None
    
    predictions = {}
    
    # Traditional ML models
    if models['traditional'] is not None:
        try:
            # SVM
            if 'svm' in models['traditional'] and models['traditional']['svm'] is not None:
                svm_pred = models['traditional']['svm'].predict([combined_features])[0]
                svm_proba = models['traditional']['svm'].predict_proba([combined_features])[0]
                predictions['SVM'] = {
                    'prediction': svm_pred,
                    'confidence': max(svm_proba),
                    'probabilities': svm_proba
                }
            
            # Random Forest
            if 'rf' in models['traditional'] and models['traditional']['rf'] is not None:
                rf_pred = models['traditional']['rf'].predict([combined_features])[0]
                rf_proba = models['traditional']['rf'].predict_proba([combined_features])[0]
                predictions['Random Forest'] = {
                    'prediction': rf_pred,
                    'confidence': max(rf_proba),
                    'probabilities': rf_proba
                }
            
            # Logistic Regression
            if 'lr' in models['traditional'] and models['traditional']['lr'] is not None:
                lr_pred = models['traditional']['lr'].predict([combined_features])[0]
                lr_proba = models['traditional']['lr'].predict_proba([combined_features])[0]
                predictions['Logistic Regression'] = {
                    'prediction': lr_pred,
                    'confidence': max(lr_proba),
                    'probabilities': lr_proba
                }
            
            # Voting Ensemble
            if 'voting' in models['traditional'] and models['traditional']['voting'] is not None:
                voting_pred = models['traditional']['voting'].predict([combined_features])[0]
                voting_proba = models['traditional']['voting'].predict_proba([combined_features])[0]
                predictions['Voting Ensemble'] = {
                    'prediction': voting_pred,
                    'confidence': max(voting_proba),
                    'probabilities': voting_proba
                }
                
        except Exception as e:
            print(f"  ⚠️ Error with traditional models: {e}")
    
    # Deep learning models
    if models['hybrid'] is not None:
        try:
            with torch.no_grad():
                outputs = models['hybrid'](raw_tensor, wavelet_tensor)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = np.argmax(proba)
                
                predictions['Hybrid CNN-LSTM-Attention'] = {
                    'prediction': pred,
                    'confidence': max(proba),
                    'probabilities': proba
                }
        except Exception as e:
            print(f"  ⚠️ Error with hybrid model: {e}")
    
    if models['ensemble_dl'] is not None:
        try:
            with torch.no_grad():
                outputs = models['ensemble_dl'](raw_tensor, wavelet_tensor)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = np.argmax(proba)
                
                predictions['Ensemble Deep Learning'] = {
                    'prediction': pred,
                    'confidence': max(proba),
                    'probabilities': proba
                }
        except Exception as e:
            print(f"  ⚠️ Error with ensemble DL model: {e}")
    
    return predictions

def demo_predictions():
    """Demonstrate predictions on sample data"""
    print("🚀 EEG Digit Classification Demo")
    print("=" * 50)
    
    # Load data
    try:
        data = np.load("reshaped_data.npy")
        features = np.load("advanced_wavelet_features.npy")
        labels = np.load("labels.npy")
        print(f"✅ Data loaded: {data.shape[0]} samples")
    except FileNotFoundError:
        print("❌ Data files not found. Please run the complete analysis first.")
        return
    
    # Load models
    models = load_trained_models()
    
    # Test on a few samples
    test_indices = [0, 500, 250, 750]  # Mix of digit 6 and 9 samples
    
    for idx in test_indices:
        print(f"\n📊 Testing Sample {idx}")
        print("-" * 30)
        
        raw_sample = data[idx]
        wavelet_sample = features[idx]
        true_label = labels[idx]
        true_digit = "6" if true_label == 0 else "9"
        
        print(f"True digit: {true_digit}")
        
        # Make predictions
        predictions = predict_with_models(models, raw_sample, wavelet_sample)
        
        if predictions:
            print("Model predictions:")
            for model_name, result in predictions.items():
                pred_digit = "6" if result['prediction'] == 0 else "9"
                confidence = result['confidence']
                correct = "✅" if result['prediction'] == true_label else "❌"
                print(f"  {model_name}: Digit {pred_digit} (confidence: {confidence:.3f}) {correct}")
        else:
            print("  ⚠️ No predictions available")

def main():
    """Main function"""
    demo_predictions()
    
    print(f"\n✅ Prediction demo completed!")
    print(f"💡 The ensemble approach combines multiple models for better accuracy")
    print(f"🎯 Individual models may disagree, but the ensemble provides robust predictions")

if __name__ == "__main__":
    main()
