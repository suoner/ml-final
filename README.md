# hERG Channel Blocker Prediction

This project implements two machine learning approaches to predict hERG channel blockers using molecular fingerprints:
1. Random Forest Classification
2. Neural Network Classification

Both models use Morgan fingerprints (radius=2, 2048 bits) generated from SMILES representations of molecules.

## Model Performance

### Random Forest
- Accuracy: 0.8483
- AUC Score: 0.9207
- Configuration:
  - 500 trees
  - Default scikit-learn parameters

### Neural Network
- Accuracy: 0.8412
- AUC Score: 0.9105
- Architecture:
  - Three-layer network: 2048 → 512 → 256 → 1
  - ReLU activations
  - Sigmoid output
- Training:
  - Batch size: 64
  - Learning rate: 0.0005
  - Epochs: 30
  - Adam optimizer

Both models achieve comparable performance on the test set, with the Random Forest showing slightly better metrics.

## Project Structure

- `random_forest_model.py`: Random Forest implementation
- `neural_network_model.py`: Neural Network implementation
- `train_and_evaluate.py`: Main script to train and compare both models
- `utils.py`: Data processing utilities
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate both models:
```bash
python train_and_evaluate.py
```

This will:
1. Load and clean the hERG dataset
2. Train both models
3. Evaluate their performance
4. Generate comparison visualizations

## Dependencies

- PyTDC (dataset)
- RDKit (molecular fingerprints)
- scikit-learn (Random Forest)
- PyTorch (Neural Network)
- pandas
- numpy
- matplotlib
