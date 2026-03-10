import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATASET_DIR
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from head import ConceptHead
from dataset import ConceptDatasetwithSite
from utils import load_labels, save_vocabs, save_distribution

# Set-up
LABEL_JSON = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", "wsi_bench_Report_train.json")
os.makedirs("output", exist_ok=True)
OUTPUT_PREFIX = "output/classifier"

# Hyper-parameters
BATCH_SIZE = 128
EPOCHS = 20
LR = 3e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
SITE_EMB_DIM=16

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training 
def train():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ids, labels, sites, feature_paths = load_labels(LABEL_JSON)
    
    # histology vocab
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Site vocab
    unique_sites = sorted(set(sites))
    site_to_idx = {s: i for i, s in enumerate(unique_sites)}

    # Save vocabs to json
    save_vocabs(label_to_idx, site_to_idx)
    save_distribution(labels, sites)
    
    # Print dataset statistics
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(ids)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Number of sites: {len(unique_sites)}")

    # Train val split
    X_train, X_val, y_train, y_val, sites_train, sites_val, fp_train, fp_val = train_test_split(
        ids, labels, sites, feature_paths, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Device: {device}\n")

    # Create dataloaders
    train_ds = ConceptDatasetwithSite(X_train, y_train, sites_train, fp_train, label_to_idx, site_to_idx)
    val_ds = ConceptDatasetwithSite(X_val, y_val, sites_val, fp_val, label_to_idx, site_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Class weights calculation
    train_label_counts = Counter(y_train)
    num_classes = len(unique_labels)
    total_samples = len(y_train)

    class_weights = []

    for label in unique_labels:
        count = train_label_counts[label]
        weight = total_samples / (num_classes * count) # Inverse frequency weighting
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Infer embedding dim
    with h5py.File(feature_paths[0], 'r') as f:
        sample_emb = f['features'][:]
    model = ConceptHead(sample_emb.shape[0], len(unique_sites), SITE_EMB_DIM, len(unique_labels)).to(device)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # Set LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # History 
    train_losses, val_losses = [], []
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for x, site_id, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            x, site_id, y = x.to(device), site_id.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, site_id)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        preds, gts = [], []

        with torch.no_grad():
            for x, site_id, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ", leave=False):
                x, site_id, y = x.to(device), site_id.to(device), y.to(device)
                logits = model(x, site_id)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                gts.extend(y.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        val_acc = accuracy_score(gts, preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{OUTPUT_PREFIX}_best.pt")
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    # Metrics
    acc = accuracy_score(gts, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gts, preds, average="macro"
    )

    # Save outputs
    torch.save(model.state_dict(), f"{OUTPUT_PREFIX}_final.pt")
    
    print(f"\n{'='*60}")
    print(f"Final Validation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"{'='*60}\n")

    with open(f"{OUTPUT_PREFIX}_metrics.txt", "w") as f:
        f.write(f"Classes: {unique_labels}\n")
        f.write(f"Number of classes: {len(unique_labels)}\n")
        f.write(f"Total samples: {len(ids)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Val samples: {len(X_val)}\n\n")
        f.write(f"Final Validation Metrics:\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})\n")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig(f"{OUTPUT_PREFIX}_loss.png", dpi=300)
    plt.close()

    print("Training complete.")

# Run script
if __name__ == "__main__":
    train()
