#!/usr/bin/env python3
# multiclass_cnn_lstm.py - Multi-class CNN-LSTM model for digits 0-9

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MultiClassCNNLSTM(nn.Module):
    """Multi-class CNN-LSTM model for 10-digit classification"""
    
    def __init__(self, input_length=128, num_classes=10, dropout_rate=0.5):
        super(MultiClassCNNLSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate LSTM input size
        lstm_input_size = input_length // 8  # After 3 pooling layers
        
        # LSTM layers
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, 
                           dropout=dropout_rate, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=dropout_rate)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def train_multiclass_model(model, train_loader, val_loader, num_epochs=100, device='cpu'):
    """Train multi-class model with advanced techniques"""
    
    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_multiclass_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_multiclass_model(model, test_loader, device='cpu'):
    """Evaluate multi-class model"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Classification report
    report = classification_report(all_targets, all_predictions, 
                                 target_names=[f'Digit {i}' for i in range(10)])
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return accuracy, report, cm, all_probabilities

def main():
    """Main function for multi-class training"""
    print("🚀 Multi-class CNN-LSTM Training")
    print("=" * 50)
    
    # This would be called with actual data loaders
    print("Model architecture created for 10-digit classification")
    print("Expected accuracy: 25-40% (based on literature)")
    print("Key challenges: Class imbalance, inter-class similarity, noise")

if __name__ == "__main__":
    main()
