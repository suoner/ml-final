import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from random_forest_model import HERGPredictor as RFPredictor
from neural_network_model import HERGPredictor as NNPredictor
from tdc.single_pred import Tox
from utils import clean_data
import torch

def train_and_evaluate_models():
    # Set device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    print("Loading dataset...")
    data = Tox(name='hERG_Karim')
    split = data.get_split()
    cleaned_split = clean_data(split)
    
    # Initialize models
    print("\nInitializing models...")
    rf_model = RFPredictor()
    nn_model = NNPredictor(
        learning_rate=0.0005,
        batch_size=64,
        num_epochs=30,
        device=device
    )
    
    # Prepare data for both models
    print("\nPreparing data...")
    rf_model.prepare_data(cleaned_split)
    nn_model.prepare_data(cleaned_split)
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_model.train()
    rf_metrics = rf_model.evaluate()
    
    # Train Neural Network
    print("\nTraining Neural Network model...")
    nn_model.train()
    nn_metrics = nn_model.evaluate()
    
    # Plot comparison
    plot_model_comparison(
        rf_metrics['y_true'], rf_metrics['y_pred_proba'],
        nn_metrics['y_true'], nn_metrics['y_pred_proba']
    )
    
    # Print metrics comparison
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, AUC: {rf_metrics['auc_score']:.4f}")
    print(f"Neural Network - Accuracy: {nn_metrics['accuracy']:.4f}, AUC: {nn_metrics['auc_score']:.4f}")

def plot_model_comparison(rf_y_true, rf_y_pred_proba, nn_y_true, nn_y_pred_proba):
    # Calculate ROC curves
    rf_fpr, rf_tpr, _ = roc_curve(rf_y_true, rf_y_pred_proba)
    nn_fpr, nn_tpr, _ = roc_curve(nn_y_true, nn_y_pred_proba)
    
    # Calculate AUC scores
    rf_auc = roc_auc_score(rf_y_true, rf_y_pred_proba)
    nn_auc = roc_auc_score(nn_y_true, nn_y_pred_proba)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot both ROC curves
    plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, 
             label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(nn_fpr, nn_tpr, color='red', lw=2, 
             label=f'Neural Network (AUC = {nn_auc:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison: Random Forest vs Neural Network')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig('plots/model_comparison.png')
    plt.close()

if __name__ == "__main__":
    train_and_evaluate_models()
