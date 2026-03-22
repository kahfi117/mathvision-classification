import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from tqdm.auto import tqdm

def prepare_smote_data(model, dataset, batch_size, device):
    model.eval()
    all_features = []
    all_labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print("Extracting features for SMOTE...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device).half()
            outputs = model.blip2.get_image_features(imgs)
            features = outputs.pooler_output
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    X_resampled = torch.tensor(X_resampled)
    y_resampled = torch.tensor(y_resampled)

    return TensorDataset(X_resampled, y_resampled)
