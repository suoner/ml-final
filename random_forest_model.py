import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_fingerprints

class HERGPredictor:
    def __init__(self):
        """Initialize the Random Forest model."""
        self.model = RandomForestClassifier(n_estimators=500, random_state=42)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def prepare_data(self, split):
        """Prepare data for training and testing."""
        # Generate fingerprints for training data
        train_fps = []
        for smiles in split['train']['Drug']:
            fp = generate_fingerprints(smiles)
            train_fps.append(fp)
        X_train = np.array(train_fps)
        y_train = np.array(split['train']['Y'])

        # Generate fingerprints for test data
        test_fps = []
        for smiles in split['test']['Drug']:
            fp = generate_fingerprints(smiles)
            test_fps.append(fp)
        X_test = np.array(test_fps)
        y_test = np.array(split['test']['Y'])

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return X_train, y_train, X_test, y_test

    def train(self):
        """Train the model."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluate the model and return metrics."""
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_true': self.y_test
        }

    def plot_roc_curve(self):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        auc_score = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('plots/rf_roc_curve.png')
        plt.close()
