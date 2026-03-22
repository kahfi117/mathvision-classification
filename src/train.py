import os
import sys

# Menambahkan src ke dalam path agar module di root src bisa dipanggil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2Model

from config import load_config
from data.dataset import load_and_prep_data, MathVisionDataset, get_transforms, get_augmented_transforms
from models.blip2_classifier import Blip2Classifier
from utils.smote import prepare_smote_data

def train_pipeline(config_path="../config/blip2_config.yaml"):
    config = load_config(config_path)
    
    # 1. SETUP
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = config['training']['batch_size']
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
    
    # 3. MODELS
    print("Initializing models & datasets...")
    pretrained_name = config['model']['pretrained_name']
    processor = Blip2Processor.from_pretrained(pretrained_name, use_fast=True)
    modelBlip2 = Blip2Model.from_pretrained(pretrained_name).to(DEVICE)
    
    transforms = get_transforms()
    train_ds = MathVisionDataset(dataframe=df_train, processor=processor, transform=transforms)
    val_ds = MathVisionDataset(dataframe=df_val, processor=processor, transform=transforms)

    num_classes = len(train_ds.df['label_idx'].unique())
    final_logs = {}

    # 4. LOOP SCENARIOS
    for scenario in SCENARIOS:
        final_logs[scenario] = {}
        print(f"\n===== SCENARIO: {scenario} =====")

        for lr in LEARNING_RATES:
            print(f"\nTesting Learning Rate: {lr}")

            model = Blip2Classifier(num_classes, modelBlip2, 
                                    hidden_dim=config['model']['hidden_dim'], 
                                    dropout=config['model']['dropout']).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            best_val_loss = float("inf")
            best_val_acc = 0

            # Mode setup
            if scenario == 'Normal':
                train_ds.transform = get_transforms()
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            elif scenario == 'Augmentasi':
                train_ds.transform = get_augmented_transforms()
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            elif scenario == 'SMOTE':
                train_ds.transform = get_transforms()
                resampled_ds = prepare_smote_data(model, train_ds, BATCH_SIZE, DEVICE)
                train_loader = DataLoader(resampled_ds, batch_size=BATCH_SIZE, shuffle=True)
            elif scenario == 'Aug_SMOTE':
                train_ds.transform = get_augmented_transforms()
                resampled_ds = prepare_smote_data(model, train_ds, BATCH_SIZE, DEVICE)
                train_loader = DataLoader(resampled_ds, batch_size=BATCH_SIZE, shuffle=True)
                train_ds.transform = get_transforms() # reset back

            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

            # Training Epochs
            for epoch in range(EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                train_bar = tqdm(train_loader, desc=f"{scenario} | LR {lr} | Epoch {epoch+1}/{EPOCHS}")

                for batch in train_bar:
                    optimizer.zero_grad()

                    if 'SMOTE' in scenario:
                        feats, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                        outputs = model.classifier(feats)
                    else:
                        imgs, labels = batch[0].to(DEVICE).half(), batch[1].to(DEVICE)
                        outputs = model(imgs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)

                    train_bar.set_postfix(
                        loss=train_loss/(train_bar.n+1),
                        acc=train_correct/train_total
                    )

                train_loss_epoch = train_loss / len(train_loader)
                train_acc_epoch = train_correct / train_total

                history["train_loss"].append(train_loss_epoch)
                history["train_acc"].append(train_acc_epoch)

                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0

                with torch.no_grad():
                    val_bar = tqdm(val_loader, desc="Validation", leave=False)
                    for imgs, labels in val_bar:
                        imgs, labels = imgs.to(DEVICE).half(), labels.to(DEVICE)
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)

                        val_bar.set_postfix(
                            loss=val_loss/(val_bar.n+1),
                            acc=val_correct/val_total
                        )

                val_loss_epoch = val_loss / len(val_loader)
                val_acc_epoch = val_correct / val_total

                history["val_loss"].append(val_loss_epoch)
                history["val_acc"].append(val_acc_epoch)

                print(f"\nEpoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss_epoch:.4f} | Acc: {train_acc_epoch:.4f} || Val Loss: {val_loss_epoch:.4f} | Acc: {val_acc_epoch:.4f}")

                # Save best models
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_{scenario}_lr{lr}_loss.pth"))
                    print("Best loss model saved!")

                if val_acc_epoch > best_val_acc:
                    best_val_acc = val_acc_epoch
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_{scenario}_lr{lr}_acc.pth"))
                    print("Best accuracy model saved!")

            final_logs[scenario][f"lr_{lr}"] = history

    # Save final JSON results
    results_path = os.path.join(SAVE_DIR, f"training_results_final_{config['run_name']}.json")
    with open(results_path, "w") as f:
        json.dump(final_logs, f, indent=4)
    print(f"Training finished. Logs saved to: {results_path}")

if __name__ == "__main__":
    train_pipeline()