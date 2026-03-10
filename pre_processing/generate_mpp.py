"""In TCGA There can be mpp (microns per pxel) which is inversely related to magnification missing from metadata.
This script creates a csv file for the slide encoder to know the WSI's mpp. If absent, deletes the file and removes from JSON."""

import os
import json
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR, DATASET_DIR
import openslide
import pandas as pd
import tqdm

# Configuration
wsi_dir = os.path.join(DATA_DIR, "temp_wsis")
output_csv = os.path.join(wsi_dir, "slide_list.csv")
wsi_json = os.path.join(DATASET_DIR, "tcga_report_histgen", "primary_site_splits", "breast_dataset.json")

results = []
files_to_delete = []  # Track files without MPP

print(f"Generating TRIDENT custom list from: {wsi_dir}")

# Iterate and Extract Metadata
for filename in tqdm.tqdm(os.listdir(wsi_dir)):
    if filename.endswith(".svs"):
        slide_path = os.path.join(wsi_dir, filename)
        try:
            slide = openslide.OpenSlide(slide_path)
            
            # Extract standard MPP tag
            mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            
            slide.close()  # Close before potential deletion
            
            # If MPP is missing, mark for deletion
            if mpp is None:
                print(f"Missing MPP for {filename} - marking for deletion")
                files_to_delete.append(filename)
            else:
                # Only add to results if MPP exists
                results.append({
                    "wsi": filename,
                    "mpp": float(mpp)
                })
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

# Delete files and update JSON (do this after loop for efficiency)
if files_to_delete:
    print(f"\nDeleting {len(files_to_delete)} files without valid MPP...")
    
    # Delete the files
    for filename in files_to_delete:
        slide_path = os.path.join(wsi_dir, filename)
        try:
            os.remove(slide_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
    
    if os.path.exists(wsi_json):
        # Update JSON file - remove entries from all splits
        print(f"\nUpdating JSON file: {wsi_json}")
        with open(wsi_json, "r") as f:
            data = json.load(f)
        
        # Get IDs to remove (without .svs extension)
        ids_to_remove = set([f.replace('.svs', '') for f in files_to_delete])
        
        total_removed = 0
        # Iterate through all splits (train, val, test, etc.)
        for split_name in data.keys():
            if isinstance(data[split_name], list):
                original_count = len(data[split_name])
                # Filter out entries where id matches deleted files
                data[split_name] = [
                    entry for entry in data[split_name] 
                    if entry.get('id') not in ids_to_remove
                ]
                removed = original_count - len(data[split_name])
                total_removed += removed
                if removed > 0:
                    print(f"  - Removed {removed} entries from '{split_name}' split")
        
        # Save updated JSON
        with open(wsi_json, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"Total JSON entries removed: {total_removed}")   
    else:
        print(f"\nJSON file not found: {wsi_json}")

# 3. Save exactly in TRIDENT format
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print("-" * 30)
print(f"SUCCESS: {output_csv} generated with {len(df)} slides.")
print(f"Total slides processed: {len(df) + len(files_to_delete)}")
print(f"Slides with valid MPP (kept): {len(df)}")
print(f"Slides without MPP (deleted): {len(files_to_delete)}")
print("-" * 30)