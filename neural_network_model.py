import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from utils import generate_fingerprints

class MorganFingerprintDataset(Dataset):
    """Dataset class for Morgan fingerprints."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HERGNet(nn.Module):
    """Neural network for hERG channel blocker prediction."""
    def __init__(self, input_size=2048):
        super(HERGNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

class HERGPredictor:
    def __init__(self, learning_rate=0.0005, batch_size=64, num_epochs=30, device=None):
        """Initialize the neural network model."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = HERGNet().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
    def prepare_data(self, data_split):
        """Prepare fingerprints from the data split."""
        train = data_split['train']
        test = data_split['test']
        
        # Generate fingerprints for training data
        train_fingerprints = train['Drug'].apply(generate_fingerprints).to_list()
        X_train = np.array(train_fingerprints)
        y_train = train['Y'].values
        
        # Generate fingerprints for test data
        test_fingerprints = test['Drug'].apply(generate_fingerprints).to_list()
        X_test = np.array(test_fingerprints)
        y_test = test['Y'].values
        
        # Create data loaders
        train_dataset = MorganFingerprintDataset(X_train, y_train)
        test_dataset = MorganFingerprintDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return X_train, y_train, X_test, y_test
    
    def train(self):
        self.model.train()
        losses = []
        
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch in self.train_loader:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs.squeeze(), y.float())
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        # Plot training loss
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.savefig('plots/training_loss.png')
        plt.close()
    
    def evaluate(self):
        self.model.eval()
        y_pred = []
        y_pred_proba = []
        y_true = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(X)
                predicted = (outputs.squeeze() > 0.5).float()
                
                y_pred.extend(predicted.cpu().numpy().flatten())
                y_pred_proba.extend(outputs.squeeze().cpu().numpy().flatten())
                y_true.extend(y.cpu().numpy().flatten())
        
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        y_true = np.array(y_true)
        
        accuracy = accuracy_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_true': y_true
        }
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig('plots/roc_curve.png')
        plt.close()
