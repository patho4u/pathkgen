import pandas as pd
import requests
import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR

# --- Configuration ---
CSV_FILE = os.path.join(DATA_DIR, "temp_wsis", "missing_wsis.csv")
MANIFEST_FILE = os.path.join(DATA_DIR, "temp_wsis", "missing_wsis_manifest.txt")
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
CSV_ID_COLUMN = "tcga_id"
BATCH_SIZE = 100 

def fetch_slide_files(short_barcodes):
    """Queries GDC for all SVS files associated with the base barcodes."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.samples.portions.slides.submitter_id", "value": short_barcodes}},
            {"op": "in", "content": {"field": "files.data_format", "value": ["SVS", "svs"]}}
        ]
    }
    # We must fetch 'file_name' to verify the suffix part
    params = {
        "filters": filters,
        "fields": "file_id,file_name,md5sum,file_size,state,cases.samples.portions.slides.submitter_id",
        "format": "JSON",
        "size": 5000
    }
    try:
        r = requests.post(GDC_FILES_ENDPOINT, json=params)
        r.raise_for_status()
        return r.json().get("data", {}).get("hits", [])
    except Exception as e:
        print(f"[!] API Request failed: {e}")
        return []

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[!] File not found: {CSV_FILE}")
        return

    # 1. Prepare Tracking
    df = pd.read_csv(CSV_FILE)
    # Store full strings: 'TCGA-SI-A71Q-01Z-00-DX4.3674A3B2...'
    raw_ids = [str(rid).strip().replace(".svs","") for rid in df[CSV_ID_COLUMN].dropna().unique()]
    
    # Create a lookup for the API (using just the barcode part)
    api_query_barcodes = list(set([rid.split('.')[0].upper() for rid in raw_ids]))
    
    print(f"[*] Total IDs to match: {len(raw_ids)}")
    
    final_manifest = {}
    matched_full_ids = set()

    # 2. Batch query
    for i in range(0, len(api_query_barcodes), BATCH_SIZE):
        batch = api_query_barcodes[i:i + BATCH_SIZE]
        hits = fetch_slide_files(batch)

        for hit in hits:
            file_name = hit.get("file_name", "")
            # Remove extension for matching: 'TCGA-SI-A71Q-01Z-00-DX4.3674A3B2'
            file_name_clean = file_name.replace(".svs", "").replace(".SVS", "")
            
            # STRICT MATCH: Check if this GDC filename exists in your list of full IDs
            for target_id in raw_ids:
                if target_id.upper() == file_name_clean.upper():
                    final_manifest[target_id] = {
                        "id": hit.get("file_id"),
                        "filename": file_name,
                        "md5": hit.get("md5sum"),
                        "size": hit.get("file_size"),
                        "state": hit.get("state")
                    }
                    matched_full_ids.add(target_id)

    # 3. Save Manifest
    if final_manifest:
        out_df = pd.DataFrame(final_manifest.values())
        # Reorder for GDC client
        out_df = out_df[["id", "filename", "md5", "size", "state"]]
        out_df.to_csv(MANIFEST_FILE, sep='\t', index=False)
        print(f"\n[✔] SUCCESS: Matched {len(out_df)} files with exact suffix.")
        print(f"[*] Manifest saved to: {MANIFEST_FILE}")
    else:
        print("\n[!] No exact filename matches found.")

    # 4. Log Missing
    missing = set(raw_ids) - matched_full_ids
    if missing:
        log_path = "still_missing.log"
        with open(log_path, "w") as f:
            for m in sorted(list(missing)):
                f.write(f"{m}\n")
        print(f"[!] {len(missing)} IDs did not have an exact filename match. See '{log_path}'.")

if __name__ == "__main__":
    main()