import os
import sys
import json
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from data.dataset import load_and_prep_data, MathVisionDataset, get_transforms, get_augmented_transforms
from models.fusion_classifier import FusionBlip2CNN
from utils.feature_extraction import extract_all_features, apply_smote_to_tensors

def train_ml_pipeline(config_path="../config/extractor_config.yaml"):
    config = load_config(config_path)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = config['training']['batch_size']
    SCENARIOS = config['training']['scenarios']
    SAVE_DIR = os.path.join("..", config['data']['output_dir'])
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data...")
    df_train, df_val, df_test = load_and_prep_data(
        os.path.join("..", config['data']['train_path']),
        os.path.join("..", config['data']['val_path']),
        os.path.join("..", config['data']['test_path'])
    )
    
    print("Initializing Feature Extractor...")
    pretrained_name = config['model']['pretrained_name']
    processor = Blip2Processor.from_pretrained(pretrained_name, use_fast=True)
    modelBlip2 = Blip2Model.from_pretrained(pretrained_name).to(DEVICE)
    modelBlip2.eval()
    
    train_ds_normal = MathVisionDataset(dataframe=df_train, processor=processor, transform=get_transforms())
    train_ds_aug = MathVisionDataset(dataframe=df_train, processor=processor, transform=get_augmented_transforms())
    val_ds = MathVisionDataset(dataframe=df_val, processor=processor, transform=get_transforms())

    num_classes = len(train_ds_normal.df['label_idx'].unique())
    
    feature_extractor = FusionBlip2CNN(num_classes, modelBlip2, 
                                   cnn_hidden=config['model']['cnn_hidden'],
                                   blip2_hidden=config['model']['blip2_hidden'],
                                   hidden_dim=config['model']['hidden_dim'], 
                                   dropout=config['model']['dropout']).to(DEVICE)
    feature_extractor.eval() 

    print("\n--- Pre-extracting Features ---")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print("Extracting Validation Data...")
    X_val, y_val = extract_all_features(feature_extractor, val_loader, DEVICE)
    X_val_np, y_val_np = X_val.numpy(), y_val.numpy()
    
    models_to_evaluate = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    final_results = {}

    for scenario in SCENARIOS:
        final_results[scenario] = {}
        print(f"\n=====================================")
        print(f"       SCENARIO: {scenario}")
        print(f"=====================================")
        
        if scenario in ['Normal', 'SMOTE']:
            dl = DataLoader(train_ds_normal, batch_size=BATCH_SIZE, shuffle=False)
            X_train, y_train = extract_all_features(feature_extractor, dl, DEVICE)
        elif scenario in ['Augmentasi', 'Aug_SMOTE']:
            dl = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=False)
            X_train, y_train = extract_all_features(feature_extractor, dl, DEVICE)
            
        if 'SMOTE' in scenario:
            print("Applying SMOTE...")
            X_train, y_train = apply_smote_to_tensors(X_train, y_train)
            
        X_train_np, y_train_np = X_train.numpy(), y_train.numpy()

        for model_name, clf in models_to_evaluate.items():
            print(f"\n-> Training {model_name}...")
            clf.fit(X_train_np, y_train_np)
            
            # Predict & Evaluate
            y_pred_val = clf.predict(X_val_np)
            val_acc = accuracy_score(y_val_np, y_pred_val)
            report = classification_report(y_val_np, y_pred_val, output_dict=True)
            
            print(f"[{model_name}] Val Accuracy: {val_acc:.4f}")
            
            final_results[scenario][model_name] = {
                "val_acc": val_acc,
                "classification_report": report
            }
            
            # Save Model
            joblib.dump(clf, os.path.join(SAVE_DIR, f"ml_{model_name}_{scenario}.joblib"))

    # Save final JSON results
    results_path = os.path.join(SAVE_DIR, "training_results_ml_classifiers.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nFinished ML Trainings. Results saved to: {results_path}")

if __name__ == "__main__":
    train_ml_pipeline()
