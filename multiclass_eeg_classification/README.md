# Multi-Class EEG Digit Classification (0-9)

## Overview
This project extends the binary EEG classification (6 vs 9) to a full multi-class classification task for digits 0-9. This represents a significantly more challenging problem with expected accuracy in the 25-40% range based on literature.

## Project Structure
```
multiclass_eeg_classification/
├── data/                   # Data files
├── src/                    # Source code
├── models/                 # Trained models
├── results/                # Results and metrics
├── figures/                # Publication figures
├── tables/                 # Publication tables
├── notebooks/              # Jupyter notebooks
└── docs/                   # Documentation
```

## Key Differences from Binary Classification

### Complexity Increase
- **Classes**: 10 (digits 0-9) vs 2 (digits 6, 9)
- **Decision Boundaries**: 45 (one-vs-one) vs 1
- **Random Baseline**: 10% vs 50%
- **Expected Accuracy**: 25-40% vs 80%+

### Technical Challenges
1. **Class Imbalance**: Uneven distribution across digits
2. **Inter-class Similarity**: Many digits have similar EEG patterns
3. **Increased Noise**: Signal-to-noise ratio becomes critical
4. **Overfitting**: More prone to memorization
5. **Confidence Distribution**: Much lower confidence scores

## Methodology

### Enhanced Data Processing
- Balanced sampling across all 10 digits
- Advanced data augmentation techniques
- Robust preprocessing pipeline

### Advanced Model Architecture
- Deeper CNN-LSTM-Attention networks
- Multi-head attention mechanisms
- Regularization techniques (dropout, batch norm)
- Class-weighted loss functions

### Ensemble Approaches
- Hierarchical classification (binary trees)
- One-vs-rest ensemble
- Confidence-based voting
- Meta-learning with uncertainty quantification

## Expected Results

### Literature Benchmarks
- Kaongoen & Jo (2017): 31.2%
- Bird et al. (2019): 28.7%
- Spampinato et al. (2017): 40.0%

### Target Performance
- **Primary Goal**: 35-45% accuracy
- **Stretch Goal**: 45-50% accuracy
- **Confidence**: Realistic distribution (0.3-0.7)

## Research Questions

1. How does classification difficulty scale with number of classes?
2. Which digits are most/least distinguishable in EEG?
3. Can hierarchical approaches improve performance?
4. What is the role of attention in multi-class EEG classification?
5. How does confidence distribution change with task complexity?

## Publication Strategy

### Target Journals
- IEEE Transactions on Neural Systems and Rehabilitation Engineering
- Journal of Neural Engineering
- Frontiers in Human Neuroscience
- IEEE Access

### Key Contributions
1. Comprehensive multi-class EEG classification framework
2. Hierarchical ensemble approaches
3. Attention mechanism analysis for EEG
4. Confidence calibration for BCI applications
5. Scalability analysis of EEG classification

## Usage

```bash
# Load and preprocess multi-class data
python src/multiclass_data_loader.py
python run_multiclass_experiment.py
# Train multi-class models
python src/multiclass_cnn_lstm.py

# Run ensemble experiments
python src/multiclass_ensemble.py

# Generate results and figures
python src/generate_results.py
```

## Future Work
- Real-time multi-class BCI implementation
- Cross-subject validation
- Integration with other modalities
- Clinical applications

## References
1. Kaongoen, N., & Jo, S. (2017). A novel online BCI system using CNN
2. Bird, J. J., et al. (2019). Mental emotional sentiment classification
3. Spampinato, C., et al. (2017). Deep learning human mind for automated visual classification

---
*This project builds upon the successful binary classification work and represents the next step toward practical multi-class BCI applications.*
