import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, num_classes=2, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)
