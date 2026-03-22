import torch
import torch.nn as nn
from models.parallel_cnn import ParallelCNN

class FusionBlip2CNN(nn.Module):
    def __init__(self, num_classes, modelBlip2, cnn_hidden=128, blip2_hidden=1408, hidden_dim=256, dropout=0.3):
        super(FusionBlip2CNN, self).__init__()
        
        # Parallel CNN part
        self.cnn = ParallelCNN(in_channels=3, num_classes=num_classes)
        merged_channels = self.cnn.branch1.out_channels + self.cnn.branch2.out_channels
        
        # Modify the head of ParallelCNN to output the extracted feature vector
        self.cnn.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(merged_channels, cnn_hidden),
            nn.ReLU(inplace=True),
        )

        # BLIP-2 Vision backbone
        self.blip2 = modelBlip2
        self.blip2.eval() # Freeze backbone by default
        
        # Fusion Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(blip2_hidden + cnn_hidden, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_fused_features(self, pixel_values):
        # Extract BLIP-2 features
        with torch.no_grad():
            outputs = self.blip2.get_image_features(pixel_values)
            blip_features = outputs.pooler_output
            
        # Extract CNN features
        cnn_features = self.cnn(pixel_values)
        
        # Concatenate features horizontally
        fused_features = torch.cat((blip_features, cnn_features), dim=1)
        return fused_features

    def forward(self, pixel_values):
        # Feature extraction via both subnetworks
        fused_features = self.get_fused_features(pixel_values)
        return self.classifier(fused_features)
