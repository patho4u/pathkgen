import json
import os

# Load labels from JSON file
def load_labels(json_path):
    ids, labels, sites, feature_paths = [], [], [], []

    with open(json_path, "r") as f:
        data = json.load(f)

    for entry in data:
        ids.append(entry["tcga_id"])
        labels.append(entry["histology"])
        sites.append(entry["primary_site"])
        feature_paths.append(entry.get("feature_path", None))

    return ids, labels, sites, feature_paths

# Save vocabs to json
def save_vocabs(label_to_idx, site_to_idx):
    os.makedirs("misc", exist_ok=True)
    with open("misc/label_vocab.json", "w") as f:
        json.dump(label_to_idx, f, indent=4)
    with open("misc/site_vocab.json", "w") as f:
        json.dump(site_to_idx, f, indent=4)
    print("Vocabs saved to misc")

# Save distribution of labels and sites to txt
def save_distribution(labels, sites, out_path="misc/distribution.txt"):
    from collections import Counter
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    total_samples = len(labels)
    label_counts = Counter(labels)
    site_counts = Counter(sites)
    
    with open(out_path, "w") as f:
        # Save Class Distribution
        f.write("="*50 + "\n")
        f.write(f"Class Distribution ({len(label_counts)} classes, {total_samples} total samples)\n")
        f.write("="*50 + "\n")
        for label, count in label_counts.most_common():
            pct = count / total_samples * 100
            f.write(f"  {label:<45} {count:>5}  ({pct:>5.1f}%)\n")
        
        f.write("\n")
        
        # Save Site Distribution
        f.write("="*50 + "\n")
        f.write(f"Site Distribution ({len(site_counts)} sites)\n")
        f.write("="*50 + "\n")
        for site, count in site_counts.most_common():
            pct = count / total_samples * 100
            f.write(f"  {site:<45} {count:>5}  ({pct:>5.1f}%)\n")
    
    print(f"Distribution saved to {out_path}")