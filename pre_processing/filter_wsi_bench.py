import json
import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR, VLM_ROOT
import re
import argparse
import random
from collections import Counter
from lookup import SPLIT_RULES, FINAL_CLASS_REMAP, HIST_TO_CUI


def split_compound_class(tcga_id: str, project_name: str, report_text: str) -> str:
    text = report_text.lower()
    for final_class, patterns in SPLIT_RULES[project_name]:
        for pat in patterns:
            if re.search(pat, text):
                return final_class
    print(f"W: Exact class not decided for {tcga_id}")
    return None

def get_histology(item):
    """Extract histology label from a WSI-Bench item using the lookup rules."""
    tcga_id = item.get('tcga_id', '')
    project = item.get('project_name', '')
    report = item.get('T-answer', '')

    if project in SPLIT_RULES:
        final_class = split_compound_class(tcga_id, project, report)
        if final_class is None:
            return None
    else:
        final_class = project

    final_class = FINAL_CLASS_REMAP.get(final_class, final_class)
    return final_class

def write_stats(data, output_log_file, title="Statistics"):
    total_count = len(data)
    
    with open(output_log_file, 'a') as f: # Append mode to keep history if called multiple times
        f.write(f"--- {title} ---\n")
        f.write(f"Total Data Points: {total_count}\n\n")
        
        if total_count == 0:
            f.write("No data points.\n\n")
            return

        # Helper for printing a section
        def print_section(title, items, count_map):
            f.write(f"{title}:\n")
            max_len = len("Name")
            if items:
                max_len = max(len(str(x)) for x in items)
            
            f.write(f"{'Name':<{max_len+2}} {'Count':<10} {'Percentage':<10}\n")
            f.write("-" * (max_len + 2 + 10 + 10 + 2) + "\n")
            
            for item, count in count_map.most_common():
                percentage = (count / total_count) * 100
                f.write(f"{str(item):<{max_len+2}} {count:<10} {percentage:<10.2f}%\n")
            f.write(f"Total entries: {len(count_map)}\n\n")

        fields_to_log = [
            ('task', 'Task Statistics'),
            ('project', 'Project Statistics'),
            ('project_name', 'Project Name Statistics'),
            ('primary_site', 'Primary Site Statistics'),
            ('disease_type', 'Disease Type Statistics')
        ]

        for field, log_title in fields_to_log:
            values = [str(item.get(field, 'N/A')) for item in data]
            counts = Counter(values)
            print_section(log_title, list(counts.keys()), counts)

def filter_wsi_bench(input_file, output_dir, tasks, feature_dir, split_ratio, seed):
    is_train = "train" in os.path.basename(input_file).lower()
    split_name = "train" if is_train else "test"
    
    tasks_str = "_".join(tasks) if isinstance(tasks, list) else tasks
    
    # 1. Setup Directories
    original_dir = os.path.join(output_dir, f"wsi_bench_{tasks_str}_original")
    with_features_dir = os.path.join(output_dir, f"wsi_bench_{tasks_str}_with_features")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(with_features_dir, exist_ok=True)
    
    # Define paths
    task_only_file = os.path.join(original_dir, f"wsi_bench_{tasks_str}_{split_name}.json")
    task_only_log = os.path.join(original_dir, f"wsi_bench_{tasks_str}_{split_name}.log")
    
    final_file = os.path.join(with_features_dir, f"wsi_bench_{tasks_str}_{split_name}.json")
    final_log = os.path.join(with_features_dir, f"wsi_bench_{tasks_str}_{split_name}.log")

    print(f"I: Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    initial_count = len(data)
    print(f"I: Initial count: {initial_count}")
    
    # 1. Filter by Task
    if tasks:
        print(f"I: Filtering by tasks: {tasks}")
        data = [item for item in data if item.get('task') in tasks]
        print(f"I: Count after task filter: {len(data)}")
        
        # Histology extraction for Report task
        if "Report" in tasks:
            print(f"I: Extracting histology labels for Report task...")
            # Annotate histology for each item first (safe single pass)
            for item in data:
                item['_histology'] = get_histology(item)

            # Now filter out items where histology could not be determined
            null_histology = 0
            kept_data = []
            for item in data:
                histology = item.pop('_histology')
                if histology is None:
                    tcga_id = item.get('tcga_id', '?')
                    print(f"W: Could not determine histology for {tcga_id} - skipping")
                    with open("skipped_samples.txt", "a") as f:
                        f.write(f"{tcga_id}\n")
                    null_histology += 1
                    print(f"I: Removed item {tcga_id} - Null histology")
                else:
                    item['histology'] = histology
                    kept_data.append(item)
            data = kept_data
            print(f"I: Histology extraction complete. Total: {len(data)}, Null histology: {null_histology}")
        
        # Save task-filtered data
        print(f"I: Saving task-filtered data to {task_only_file}...")
        with open(task_only_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Log stats for task-filtered data
        with open(task_only_log, 'w') as f:
            f.write(f"Task-Only Filtering Log for {input_file}\n")
            f.write("========================================\n\n")
        write_stats(data, task_only_log, title="Task-Only Statistics")
        
    # 2. Filter by Feature Existence
    if feature_dir:
        print(f"I: Filtering by feature existence in {feature_dir}...")
        filtered_data = []
        found_features = 0
        missing_features = 0
        
        available_files = set(os.listdir(feature_dir))
        
        for item in data:
            tcga_id = item.get('tcga_id')
            if not tcga_id:
                missing_features += 1
                continue
                
            expected_filename = f"{tcga_id}.h5"
            
            if expected_filename in available_files:
                abs_path = os.path.join(feature_dir, expected_filename)
                # Store relative to DATA_DIR so the JSON is portable across machines
                item['feature_path'] = os.path.relpath(abs_path, DATA_DIR)
                filtered_data.append(item)
                found_features += 1
            else:
                missing_features += 1
        
        data = filtered_data
        print(f"I: Count after feature filter: {len(data)} (Found: {found_features}, Missing: {missing_features})")

    # Final Save (Feature Filtered)
    print(f"I: Saving final filtered data to {final_file}...")
    with open(final_file, 'w') as f:
        json.dump(data, f, indent=4)
        
    with open(final_log, 'w') as f:
        f.write(f"Final Filtering Log for {input_file}\n")
        f.write("========================================\n\n")
    write_stats(data, final_log, title="Post-Filtering Statistics")

    print(f"I: Done. Final results in {final_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and split WSI-Bench dataset.")
    parser.add_argument("--input_file", help="Path to input processed JSON")
    parser.add_argument("--output_dir", help="Directory to save filtered results")
    parser.add_argument("--tasks", nargs='+', help="List of tasks to keep")
    parser.add_argument("--feature_dir", help="Path to feature directory (.h5 files)")
    parser.add_argument("--split_ratio", type=float, default=0.0, help="Test set ratio (0.0 to 1.0). Default 0.0 (no split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()

    # Preset paths (Defaults)
    input_train_base = os.path.join(DATA_DIR, "dataset", "wsi_bench_full")
    input_train_file = os.path.join(input_train_base, "WSI-Bench-train.json")
    input_test_file  = os.path.join(input_train_base, "WSI-Bench-test.json")
    
    output_dir = args.output_dir if args.output_dir else os.path.join(DATA_DIR, "dataset")

    default_train_feature_dir = os.path.join(DATA_DIR, "slide_features", "train")
    default_test_feature_dir = os.path.join(DATA_DIR, "slide_features", "test")
    default_tasks = ["Report"]

    print("--- Processing Train Set ---")
    filter_wsi_bench(
        input_file=input_train_file,
        output_dir=output_dir,
        tasks=args.tasks if args.tasks else default_tasks,
        feature_dir=args.feature_dir if args.feature_dir else default_train_feature_dir,
        split_ratio=0,
        seed=args.seed
    )

    print("\n--- Processing Test Set ---")
    filter_wsi_bench(
        input_file=input_test_file,
        output_dir=output_dir,
        tasks=args.tasks if args.tasks else default_tasks,
        feature_dir=args.feature_dir if args.feature_dir else default_test_feature_dir,
        split_ratio=0,
        seed=args.seed
    )
