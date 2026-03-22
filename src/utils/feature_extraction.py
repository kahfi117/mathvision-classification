import torch
import numpy as np
from tqdm.auto import tqdm
from imblearn.over_sampling import SMOTE

def extract_all_features(model, dataloader, device):
    """
    Passes all images through the model's feature extraction method 
    and returns Tensors for features and labels.
    """
    model.eval()
    all_features = []
    all_labels = []

    print("Extracting features from the dataset...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device).half()
            
            # extract features by bypassing the head network
            features = model.get_fused_features(imgs)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    
    return torch.tensor(X), torch.tensor(y)

def apply_smote_to_tensors(X, y):
    """ Apply SMOTE solely to pre-extracted torch Tensors """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X.numpy(), y.numpy())
    return torch.tensor(X_res), torch.tensor(y_res)
