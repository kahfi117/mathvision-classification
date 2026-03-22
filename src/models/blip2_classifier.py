import torch
import torch.nn as nn

class Blip2Classifier(nn.Module):
    def __init__(self, num_classes, modelBlip2, hidden_dim=256, dropout=0.3):
        super(Blip2Classifier, self).__init__()
        self.blip2 = modelBlip2
        self.blip2.eval() # Freeze backbone by default
        
        self.classifier = nn.Sequential(
            nn.Linear(1408, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.blip2.get_image_features(pixel_values)
            features = outputs.pooler_output
        return self.classifier(features)
