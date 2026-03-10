import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import VLM_ROOT, DATA_DIR, DATASET_DIR
import csv
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from head import ConceptHead
from dataset import ConceptDatasetwithSite
from utils import load_labels


# Set-up
SPLIT = "test"
LABEL_JSON = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", f"wsi_bench_Report_{SPLIT}.json")
TRAIN_JSON = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", "wsi_bench_Report_train.json")
EMBEDDING_DIR = os.path.join(DATA_DIR, "wsi_bench", "slide_features", SPLIT)
MODEL_PATH = os.path.join(VLM_ROOT, "auxiliary_classifier", "output", "classifier_best.pt")
OUTPUT_PREFIX = f"output/classifier_{SPLIT}"

# Hyper-parameters
BATCH_SIZE = 128
SITE_EMB_DIM = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

# Inference
def inference():
    print(f"\n{'='*60}")
    print(f"Running inference on {LABEL_JSON}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    num_classes = ckpt["fc.3.weight"].shape[0]
    num_sites = ckpt["site_emb.weight"].shape[0]

    # Build label vocab from train CSV (must match the order used during training)
    train_ids, train_labels, train_sites, _ = load_labels(TRAIN_JSON)
    unique_labels = sorted(set(train_labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for i, l in enumerate(unique_labels)}

    # Build site vocab from train CSV (must match the order used during training)
    unique_sites = sorted(set(train_sites))
    site_to_idx  = {s: i for i, s in enumerate(unique_sites)}

    print(f"Classes (from checkpoint): {num_classes}")
    print(f"Sites   (from checkpoint): {num_sites}")

    # Load ids, GT labels, and sites from test CSV
    ids, gt_labels, sites, feature_paths = load_labels(LABEL_JSON)
    
    # Filter out samples without embedding files
    print("Checking for missing embedding files...")
    valid_entries = []
    missing_count = 0

    for i, id_ in enumerate(ids):
        emb_path = os.path.join(EMBEDDING_DIR, f"{id_}.h5")
        if os.path.exists(emb_path):
            valid_entries.append(i)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Removed {missing_count} samples without embedding files")

    ids       = [ids[i]       for i in valid_entries]
    gt_labels = [gt_labels[i] for i in valid_entries]
    sites     = [sites[i]     for i in valid_entries]
    
    print(f"Total samples: {len(ids)}")
    print(f"Device: {device}\n")
    
    # Create dataset
    dataset    = ConceptDatasetwithSite(ids, gt_labels, sites, feature_paths, label_to_idx, site_to_idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model
    with h5py.File(os.path.join(EMBEDDING_DIR, f"{ids[0]}.h5"), 'r') as f:
        sample_emb = f['features'][:]
    
    model = ConceptHead(sample_emb.shape[0], num_sites, SITE_EMB_DIM, num_classes).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}\n")
    
    # Inference
    all_preds = []

    with torch.no_grad():
        for x, site_id, _ in tqdm(dataloader, desc="Inference"):
            x       = x.to(device)
            site_id = site_id.to(device)
            logits  = model(x, site_id)
            preds   = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    # Save predictions in ground truth format
    output_csv  = f"{OUTPUT_PREFIX}_pred.csv"
    metrics_txt = f"{OUTPUT_PREFIX}_inference_metrics.txt"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_labels = [idx_to_label[p] for p in all_preds]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tcga_id", "predicted_label"])
        for tcga_id, pred_label in zip(ids, pred_labels):
            writer.writerow([tcga_id, pred_label])

    print(f"\nInference complete.")
    print(f"Saved predictions (GT format) to {output_csv}")

    # Metrics
    if gt_labels is not None:
        acc      = accuracy_score(gt_labels, pred_labels)
        macro_f1 = f1_score(gt_labels, pred_labels, average="macro", zero_division=0)
        report   = classification_report(gt_labels, pred_labels, zero_division=0)

        with open(metrics_txt, "w") as mf:
            mf.write(f"{'='*60}\n")
            mf.write(f"Metrics\n")
            mf.write(f"{'='*60}\n\n")
            mf.write(f"  Accuracy  : {acc * 100:.2f}%\n")
            mf.write(f"  Macro-F1  : {macro_f1:.4f}\n\n")
            mf.write("Per-class report:\n")
            mf.write(report)
            mf.write("\n")

        print(f"Saved metrics to {metrics_txt}")

        # Confusion matrix
        present_labels = sorted(set(gt_labels) | set(pred_labels))
        cm = confusion_matrix(gt_labels, pred_labels, labels=present_labels)

        fig_w = max(12, len(present_labels) * 0.6)
        fig_h = max(10, len(present_labels) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        # Annotate each cell
        thresh = cm.max() / 2.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[row, col] > thresh else "black",
                )

        ax.set_xticks(range(len(present_labels)))
        ax.set_xticklabels(present_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(present_labels)))
        ax.set_yticklabels(present_labels, rotation=0, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Ground Truth", fontsize=12)
        ax.set_title(f"Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=13)
        plt.tight_layout()

        cm_path = f"{OUTPUT_PREFIX}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_path}")
    else:
        print("\n[Note] No 'label' column found in CSV — skipping metrics and confusion matrix.")

if __name__ == "__main__":
    inference()