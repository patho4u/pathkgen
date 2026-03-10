import json
import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths_config import VLM_ROOT, DATA_DIR, DATASET_DIR
import copy
import pandas as pd
from collections import Counter

# ── Input paths ──────────────────────────────────────────────────────────────
WSI_TRAIN_SET = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", "wsi_bench_Report_train.json")
WSI_TEST_SET = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", "wsi_bench_Report_test.json")

# train_histology_preds = os.path.join(VLM_ROOT, "auxiliary_classifier", "output", "classifier_train_pred.csv")
# test_histology_preds = os.path.join(VLM_ROOT, "auxiliary_classifier", "output", "classifier_test_pred.csv")
train_histology_preds = os.path.join(DATASET_DIR, "wsi_bench_Report_knowledge", "WSI-Bench-train_Report_knowledge.json")
test_histology_preds = os.path.join(DATASET_DIR, "wsi_bench_Report_knowledge", "WSI-Bench-test_Report_knowledge.json")

knowledge_path = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "histology_descriptions_with_features.json")

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(DATASET_DIR)

updated_knowledge_train_set = os.path.join(OUT_DIR, "WSI-Bench-train_Report_knowledge.json")
updated_knowledge_test_set  = os.path.join(OUT_DIR, "WSI-Bench-test_Report_knowledge.json")

train_log_path = os.path.join(OUT_DIR, "WSI-Bench-train_Report_knowledge.log")
test_log_path  = os.path.join(OUT_DIR, "WSI-Bench-test_Report_knowledge.log")


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_table(rows, col_headers, col_widths):
    """Return a simple fixed-width table string."""
    sep = "-" * (sum(col_widths) + len(col_widths))
    header = "".join(h.ljust(w) for h, w in zip(col_headers, col_widths))
    lines = [header, sep]
    for row in rows:
        lines.append("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    return "\n".join(lines)


def compute_stats(dataset: list):
    """Compute per-field statistics for logging."""
    stats = {}
    for field in ("project", "project_name", "primary_site", "disease_type", "histology"):
        counter = Counter(entry.get(field, "N/A") for entry in dataset)
        total = sum(counter.values())
        stats[field] = (counter, total)
    return stats


def write_log(log_path: str, source_path: str, dataset: list,
              dropped_ids: list = None, knowledge_miss: list = None):
    """Write a statistics log file similar to the existing .log format."""
    stats = compute_stats(dataset)

    lines = []
    lines.append(f"Enrichment Log for {os.path.relpath(source_path, DATA_DIR)}")
    lines.append("=" * 40)
    lines.append("")

    lines.append("--- Post-Enrichment Statistics ---")
    lines.append(f"Total Data Points: {len(dataset)}")
    lines.append("")

    field_labels = {
        "project":      "Project Statistics:",
        "project_name": "Project Name Statistics:",
        "primary_site": "Primary Site Statistics:",
        "disease_type": "Disease Type Statistics:",
        "histology":    "Histology Statistics:",
    }

    for field, label in field_labels.items():
        counter, total = stats[field]
        lines.append(label)
        rows = []
        for name, cnt in sorted(counter.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100 if total > 0 else 0
            rows.append((name, cnt, f"{pct:.2f}    %"))
        col_widths = [55, 11, 12]
        lines.append(format_table(rows, ["Name", "Count", "Percentage"], col_widths))
        lines.append(f"Total entries: {len(counter)}")
        lines.append("")

    # Knowledge-miss summary
    if knowledge_miss:
        lines.append(f"--- Knowledge Lookup Misses ({len(knowledge_miss)} entries) ---")
        for item in knowledge_miss:
            lines.append(f"  {item}")
        lines.append("")

    # Dropped IDs summary
    if dropped_ids:
        lines.append(f"--- Dropped IDs (no prediction found, {len(dropped_ids)} entries) ---")
        for tid in dropped_ids:
            lines.append(f"  {tid}")
        lines.append("")

    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Log written → {log_path}")


# ── Core enrichment functions ─────────────────────────────────────────────────

def enrich_dataset(source_path: str, output_path: str, knowledge_dict: dict,
                   pred_json_path: str, log_path: str):
    """
    General enrichment function:
      - look up the predicted histology from pred_json_path via tcga_id
        (reads the 'pred_histology' field from each entry in the JSON list)
      - replace 'knowledge' field with knowledge_dict[predicted_histology]
      - drop entries whose tcga_id is missing in the predictions or knowledge_dict
    """
    with open(source_path, "r") as f:
        reports = json.load(f)

    print(f"  Loaded {len(reports)} reports from {source_path}")

    # Build tcga_id → pred_histology mapping from the knowledge JSON
    with open(pred_json_path, "r") as f:
        pred_entries = json.load(f)
    pred_map = {
        e["tcga_id"].strip(): e["pred_histology"].strip()
        for e in pred_entries
        if e.get("tcga_id") and e.get("pred_histology")
    }
    print(f"  Loaded {len(pred_map)} predictions from {pred_json_path}")

    updated = []
    dropped_ids = []
    knowledge_miss = []

    for entry in reports:
        tcga_id = entry.get("tcga_id", "").strip()

        if tcga_id not in pred_map:
            dropped_ids.append(tcga_id if tcga_id else "unknown_id")
            continue

        predicted_histology = pred_map[tcga_id].strip()

        if not predicted_histology or predicted_histology not in knowledge_dict:
            knowledge_miss.append(f"{tcga_id}: predicted_histology='{predicted_histology}'")
            dropped_ids.append(tcga_id)   # drop — no usable knowledge
            continue

        new_entry = copy.deepcopy(entry)
        new_entry["pred_histology"] = predicted_histology
        new_entry["knowledge"] = knowledge_dict[predicted_histology]
        updated.append(new_entry)

    print(f"  Kept {len(updated)} entries  |  Dropped {len(dropped_ids)} entries (no prediction or unmatched histology)")
    if knowledge_miss:
        print(f"  [DROPPED] {len(knowledge_miss)} entries had no knowledge match")

    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"  Saved → {output_path}")

    write_log(log_path, source_path, updated,
              dropped_ids=dropped_ids, knowledge_miss=knowledge_miss)
    return updated


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load knowledge descriptions
    with open(knowledge_path, "r") as f:
        knowledge_dict = json.load(f)
    print(f"Loaded {len(knowledge_dict)} knowledge entries from {knowledge_path}")
    print("\n" + "=" * 80)

    # ── Training set ──
    print("\nProcessing Training Set...")
    print(f"  Input    : {WSI_TRAIN_SET}")
    print(f"  Pred JSON: {train_histology_preds}")
    print(f"  Output   : {updated_knowledge_train_set}")
    enrich_dataset(source_path=WSI_TRAIN_SET, 
                   output_path=updated_knowledge_train_set, 
                   knowledge_dict=knowledge_dict, 
                   pred_json_path=train_histology_preds, 
                   log_path=train_log_path)

    print("\n" + "=" * 80)

    # ── Test set ──
    print("\nProcessing Test Set...")
    print(f"  Input    : {WSI_TEST_SET}")
    print(f"  Pred JSON: {test_histology_preds}")
    print(f"  Output   : {updated_knowledge_test_set}")
    enrich_dataset(source_path=WSI_TEST_SET, 
                   output_path=updated_knowledge_test_set, 
                   knowledge_dict=knowledge_dict, 
                   pred_json_path=test_histology_preds, 
                   log_path=test_log_path)

    print("\n" + "=" * 80)
    print("\nAll datasets enriched successfully!")


if __name__ == "__main__":
    main()
