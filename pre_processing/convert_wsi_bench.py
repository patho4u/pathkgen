import json
import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR
import argparse
import requests
import time
from collections import Counter

def write_stats(data, output_log_file):
    total_count = len(data)
    
    # helper for printing a section
    def print_section(f, title, items, count_map):
        f.write(f"{title}:\n")
        # Determine max length for alignment
        max_len = len("Name")
        if items:
            max_len = max(len(str(x)) for x in items)
        
        # Header
        f.write(f"{'Name':<{max_len+2}} {'Count':<10} {'Percentage':<10}\n")
        f.write("-" * (max_len + 2 + 10 + 10 + 2) + "\n")
        
        for item, count in count_map.most_common():
            percentage = (count / total_count) * 100
            f.write(f"{str(item):<{max_len+2}} {count:<10} {percentage:<10.2f}%\n")
        f.write(f"Total entries: {len(count_map)}\n\n")

    with open(output_log_file, 'w') as f:
        f.write(f"Total Data Points: {total_count}\n\n")

        fields_to_log = [
            ('task', 'Task Statistics'),
            ('project', 'Project Statistics'),
            ('project_name', 'Project Name Statistics'),
            ('primary_site', 'Primary Site Statistics'),
            ('disease_type', 'Disease Type Statistics')
        ]

        for field, title in fields_to_log:
            # Filter out None/empty (but keep 'Unknown' if you prefer, currently extraction defaults to None if missing)
            # Actually, let's treat None as 'Missing/Not Fetched' for clarity if needed, or just exclude.
            # The previous code used 'Unknown' in the Counter init. 
            # I'll stick to collecting values and converting None to 'N/A' for stats.
            values = [str(item.get(field, 'N/A')) for item in data]
            counts = Counter(values)
            print_section(f, title, list(counts.keys()), counts)

def fetch_gdc_metadata(case_ids):
    """
    Fetches metadata for a list of case IDs from the GDC API.
    Returns a dictionary mapping case_id to metadata (project_name, primary_site, disease_type).
    """
    print(f"Fetching GDC metadata for {len(case_ids)} cases...")
    
    url = "https://api.gdc.cancer.gov/cases"
    metadata_map = {}
    
    # GDC API allows up to a certain number of filters. Batching is safer.
    # We'll batch by 500.
    batch_size = 500
    case_ids_list = list(case_ids)
    
    for i in range(0, len(case_ids_list), batch_size):
        batch = case_ids_list[i:i + batch_size]
        
        filters = {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": batch
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "submitter_id,project.project_id,project.name,primary_site,disease_type",
            "format": "json",
            "size": batch_size
        }
        
        try:
            response = requests.post(url, json=params) # Using POST to avoid long URL issues with filters
            response.raise_for_status()
            data = response.json()
            
            for hit in data.get("data", {}).get("hits", []):
                submitter_id = hit.get("submitter_id")
                project_info = hit.get("project", {})
                
                metadata_map[submitter_id] = {
                    "project_name": project_info.get("name"),
                    "primary_site": hit.get("primary_site"),
                    "disease_type": hit.get("disease_type")
                }
            
            time.sleep(0.1) # Be nice to the API
            print(f"Processed batch {i} - {min(i + batch_size, len(case_ids_list))}")
            
        except Exception as e:
            print(f"Error fetching metadata for batch starting at index {i}: {e}")
            
    return metadata_map

def convert_wsi_bench_train(input_file, output_file, log_file, use_gdc=False):
    print(f"Converting train set from {input_file} to {output_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 1. Pre-fetch metadata if flag is set
    gdc_metadata = {}
    if use_gdc:
        # Extract unique case IDs
        case_ids = set()
        for item in data:
            image_path = item.get('image', '')
            filename = os.path.basename(image_path)
            # Assuming format TCGA-XX-XXXX...
            # Case ID is usually the first 12 characters: TCGA-XX-XXXX
            if len(filename) >= 12 and filename.startswith("TCGA"):
                 case_ids.add(filename[:12])
        
        if case_ids:
            gdc_metadata = fetch_gdc_metadata(case_ids)
    
    converted_data = []
    
    for item in data:
        # Extract fields
        image_path = item.get('image', '')
        if '/' in image_path:
            filename = image_path.split('/')[-1]
            image_id = os.path.splitext(filename)[0]
            project = image_path.split('/')[0]
        else:
            filename = image_path
            image_id = os.path.splitext(filename)[0]
            project = "UNKNOWN"

        conversations = item.get('conversations', [])
        question = ""
        answer = ""
        
        for conv in conversations:
            if conv.get('from') == 'human':
                # question = conv.get('value', '').replace("<image>\n", "")
                question = conv.get('value', '')    
            elif conv.get('from') == 'gpt':
                answer = conv.get('value', '')

        new_entry = {
            "id": item.get('id'),
            "tcga_id": image_id,
            "project": project,
            "question": question,
            "T-answer": answer,
        }
        
        # Task extraction for train set
        id_val = item.get('id', '')
        if '_TCGA' in id_val:
            parts = id_val.split('_TCGA')
            if len(parts) > 1:
                new_entry['task'] = parts[0]
            else:
                 new_entry['task'] = "Morphology_choice"
        else:
             new_entry['task'] = "Morphology_choice"
        
        # Add GDC metadata if available
        if use_gdc:
            # Extract case ID again
            case_id = None
            if len(filename) >= 12 and filename.startswith("TCGA"):
                case_id = filename[:12]
            
            if case_id and case_id in gdc_metadata:
                meta = gdc_metadata[case_id]
                new_entry["project_name"] = meta.get("project_name")
                new_entry["primary_site"] = meta.get("primary_site")
                new_entry["disease_type"] = meta.get("disease_type")
            else:
                new_entry["project_name"] = None
                new_entry["primary_site"] = None
                new_entry["disease_type"] = None

        converted_data.append(new_entry)

    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    write_stats(converted_data, log_file)
    print(f"Train set conversion complete. Stats written to {log_file}")

def convert_wsi_bench_test(input_file, output_file, log_file, use_gdc=False):
    print(f"Converting test set from {input_file} to {output_file}...")
    
    # Read all lines first to extract IDs for batch fetching
    raw_items = []
    with open(input_file, 'r') as f:
        for line in f:
            raw_items.append(json.loads(line))
            
    # 1. Pre-fetch metadata if flag is set
    gdc_metadata = {}
    if use_gdc:
        case_ids = set()
        for item in raw_items:
            image_path = item.get('image', '')
            filename = os.path.basename(image_path)
            if len(filename) >= 12 and filename.startswith("TCGA"):
                 case_ids.add(filename[:12])
        
        if case_ids:
            gdc_metadata = fetch_gdc_metadata(case_ids)

    converted_data = []
    
    for item in raw_items:
        image_path = item.get('image', '')
        if '/' in image_path:
            filename = image_path.split('/')[-1]
            image_id = os.path.splitext(filename)[0]
            project = image_path.split('/')[0]
        else:
            filename = image_path
            image_id = os.path.splitext(filename)[0]
            project = "UNKNOWN"
        
        new_entry = {
            "id": item.get('question_id'),
            "tcga_id": image_id,
            "project": project,
            "question": item.get('question', ''),
            "T-answer": item.get('T-answer', ''),
            "task": item.get('metadata', '')
        }
        
        # Add GDC metadata if available
        if use_gdc:
            case_id = None
            if len(filename) >= 12 and filename.startswith("TCGA"):
                case_id = filename[:12]
            
            if case_id and case_id in gdc_metadata:
                meta = gdc_metadata[case_id]
                new_entry["project_name"] = meta.get("project_name")
                new_entry["primary_site"] = meta.get("primary_site")
                new_entry["disease_type"] = meta.get("disease_type")
            else:
                new_entry["project_name"] = None
                new_entry["primary_site"] = None
                new_entry["disease_type"] = None

        converted_data.append(new_entry)

    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    write_stats(converted_data, log_file)
    print(f"Test set conversion complete. Stats written to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WSI-Bench dataset to formatted JSON.")
    parser.add_argument("--use_gdc", action="store_true", help="Fetch metadata from GDC API")
    args = parser.parse_args()

    base_dir = os.path.join(DATA_DIR, "wsi_bench", "dataset_raw")
    output_dir = os.path.join(os.path.dirname(base_dir), "dataset/wsi_bench_full")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train set
    train_input = os.path.join(base_dir, "WSI-Bench-train.json")
    train_output = os.path.join(output_dir, "WSI-Bench-train.json")
    train_log = os.path.join(output_dir, "WSI-Bench-train-stats.log")
    convert_wsi_bench_train(train_input, train_output, train_log, use_gdc=args.use_gdc)
    
    # Test set
    test_input = os.path.join(base_dir, "WSI-Bench-test.jsonl")
    test_output = os.path.join(output_dir, "WSI-Bench-test.json")
    test_log = os.path.join(output_dir, "WSI-Bench-test-stats.log")
    convert_wsi_bench_test(test_input, test_output, test_log, use_gdc=args.use_gdc)
