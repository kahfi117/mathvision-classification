import os
import sys

# Add src to sys.path so modules can import from the root of src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2Model

from config import load_config
from data.dataset import load_and_prep_data, MathVisionDataset, get_transforms, get_augmented_transforms
from models.fusion_classifier import FusionBlip2CNN
from models.mlp_classifier import MLPClassifier
from utils.feature_extraction import extract_all_features, apply_smote_to_tensors

def train_extractor_pipeline(config_path="../config/extractor_config.yaml"):
    config = load_config(config_path)
    
    # 1. SETUP
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = config['training']['batch_size']
    MLP_BATCH_SIZE = config['training'].get('mlp_batch_size', 64)
    EPOCHS = config['training']['epochs']
    LEARNING_RATES = config['training']['learning_rates']
    SCENARIOS = config['training']['scenarios']
    SAVE_DIR = os.path.join("..", config['data']['output_dir'])
    LOG_DIR = os.path.join("..", config['data']['log_dir'])
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. DATA
    print("Loading data...")
    df_train, df_val, df_test = load_and_prep_data(
        os.path.join("..", config['data']['train_path']),
        os.path.join("..", config['data']['val_path']),
        os.path.join("..", config['data']['test_path'])
    )
    
    # 3. MODELS & FEATURE EXTRACTION
    print("Initializing Feature Extractor...")
    pretrained_name = config['model']['pretrained_name']
    processor = Blip2Processor.from_pretrained(pretrained_name, use_fast=True)
    modelBlip2 = Blip2Model.from_pretrained(pretrained_name).to(DEVICE)
    modelBlip2.eval() # Pre-extraction mode
    
    transforms = get_transforms()
    train_ds_normal = MathVisionDataset(dataframe=df_train, processor=processor, transform=transforms)
    train_ds_aug = MathVisionDataset(dataframe=df_train, processor=processor, transform=get_augmented_transforms())
    val_ds = MathVisionDataset(dataframe=df_val, processor=processor, transform=transforms)

    num_classes = len(train_ds_normal.df['label_idx'].unique())
    
    # Using FusionBlip2CNN strictly as feature extractor by bypassing standard forward()
    feature_extractor = FusionBlip2CNN(num_classes, modelBlip2, 
                                   cnn_hidden=config['model']['cnn_hidden'],
                                   blip2_hidden=config['model']['blip2_hidden'],
                                   hidden_dim=config['model']['hidden_dim'], 
                                   dropout=config['model']['dropout']).to(DEVICE)
    feature_extractor.eval() 

    print("\n--- Pre-extracting Features ---")
    val_loader_img = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print("Extracting Validation Data...")
    X_val, y_val = extract_all_features(feature_extractor, val_loader_img, DEVICE)
    
    final_logs = {}

    # 4. LOOP SCENARIOS
    for scenario in SCENARIOS:
        final_logs[scenario] = {}
        print(f"\n===== SCENARIO: {scenario} =====")
        
        # 4a. Prepare scenario-specific features 
        if scenario in ['Normal', 'SMOTE']:
            dl = DataLoader(train_ds_normal, batch_size=BATCH_SIZE, shuffle=False)
            X_train, y_train = extract_all_features(feature_extractor, dl, DEVICE)
        elif scenario in ['Augmentasi', 'Aug_SMOTE']:
            dl = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=False)
            X_train, y_train = extract_all_features(feature_extractor, dl, DEVICE)
            
        if 'SMOTE' in scenario:
            print("Applying SMOTE on pre-extracted tensors...")
            X_train, y_train = apply_smote_to_tensors(X_train, y_train)
            
        # We form pure Tensor Dataloaders for blazing fast iterations.
        train_loader_mlp = DataLoader(TensorDataset(X_train, y_train), batch_size=MLP_BATCH_SIZE, shuffle=True)
        val_loader_mlp = DataLoader(TensorDataset(X_val, y_val), batch_size=MLP_BATCH_SIZE, shuffle=False)
        input_dim = X_train.shape[1]

        # 4b. MLP Iteration Loop
        for lr in LEARNING_RATES:
            print(f"\nTesting Learning Rate: {lr}")

            # Every LR step starts with a fresh initialized MLP.
            mlp = MLPClassifier(input_dim=input_dim, 
                                hidden_dim=config['model']['hidden_dim'], 
                                num_classes=num_classes,
                                dropout=config['model']['dropout']).to(DEVICE)
            
            optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            best_val_loss = float("inf")
            best_val_acc = 0
            patience = 4
            patience_counter = 0

            # Training Epochs purely on pre-extracted 1D variables instead of 3D image arrays!
            for epoch in range(EPOCHS):
                mlp.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                train_bar = tqdm(train_loader_mlp, desc=f"{scenario} | LR {lr} | Epoch {epoch+1}/{EPOCHS}")

                for X_batch, y_batch in train_bar:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    optimizer.zero_grad()

                    outputs = mlp(X_batch)

                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == y_batch).sum().item()
                    train_total += y_batch.size(0)

                    train_bar.set_postfix(
                        loss=train_loss/(train_bar.n+1),
                        acc=train_correct/train_total
                    )

                train_loss_epoch = train_loss / len(train_loader_mlp)
                train_acc_epoch = train_correct / train_total

                history["train_loss"].append(train_loss_epoch)
                history["train_acc"].append(train_acc_epoch)

                # Validation
                mlp.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0

                with torch.no_grad():
                    val_bar = tqdm(val_loader_mlp, desc="Validation", leave=False)
                    for X_batch, y_batch in val_bar:
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        outputs = mlp(X_batch)
                        loss = criterion(outputs, y_batch)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == y_batch).sum().item()
                        val_total += y_batch.size(0)

                        val_bar.set_postfix(
                            loss=val_loss/(val_bar.n+1),
                            acc=val_correct/val_total
                        )

                val_loss_epoch = val_loss / len(val_loader_mlp)
                val_acc_epoch = val_correct / val_total

                history["val_loss"].append(val_loss_epoch)
                history["val_acc"].append(val_acc_epoch)

                print(f"\nEpoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss_epoch:.4f} | Acc: {train_acc_epoch:.4f} || Val Loss: {val_loss_epoch:.4f} | Acc: {val_acc_epoch:.4f}")

                # Save best models
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    patience_counter = 0
                    torch.save(mlp.state_dict(), os.path.join(SAVE_DIR, f"extractor_{scenario}_lr{lr}_loss.pth"))
                    print("Best loss model saved!")
                else:
                    patience_counter += 1
                    print(f"EarlyStopping counter: {patience_counter} out of {patience}")

                if val_acc_epoch > best_val_acc:
                    best_val_acc = val_acc_epoch
                    torch.save(mlp.state_dict(), os.path.join(SAVE_DIR, f"extractor_{scenario}_lr{lr}_acc.pth"))
                    print("Best accuracy model saved!")

                if patience_counter >= patience:
                    print("Early stopping triggered. Stopping training for this LR/Scenario.")
                    break

            final_logs[scenario][f"lr_{lr}"] = history

    # Save final JSON results
    results_path = os.path.join(SAVE_DIR, f"training_results_extractor_{config['run_name']}.json")
    with open(results_path, "w") as f:
        json.dump(final_logs, f, indent=4)
    print(f"Training finished. Logs saved to: {results_path}")

if __name__ == "__main__":
    train_extractor_pipeline()
