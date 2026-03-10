
import sys
from pathlib import Path
from collections import Counter
import csv
import os
import json

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT_DIR)
from paths_config import DATA_DIR
sys.path.insert(0, os.path.join(ROOT_DIR, "knowledge_path"))
from graphs import Neo4jQuery

TOP_N = 150

def get_top_n_cuis(all_cuis, n):
    cui_counter = Counter(all_cuis)
    top_n_cuis = cui_counter.most_common(n)
    top_n_cuis = [cui[0] for cui in top_n_cuis]

    return top_n_cuis 

if __name__ == "__main__":  
    NEO4J_URL = "url" # Change this to your Neo4j URL
    NEO4J_USER = "user" # Change this to your Neo4j username
    NEO4J_PASSWORD = "password" # Change this to your Neo4j password

    neo4j_query = Neo4jQuery(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)

    histology_jsons = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "cui_jsons")
    top_cuis_json = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "top_cuis", f"top_{TOP_N}_cuis.json")
    top_cuis_names_json = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "top_cuis", f"top_{TOP_N}_cuis_names.json")

    # Load existing data or create new dict
    if os.path.exists(top_cuis_json):
        with open(top_cuis_json, "r") as f:
            top_cuis_data = json.load(f)
    else:
        top_cuis_data = {}

    if os.path.exists(top_cuis_names_json):
        with open(top_cuis_names_json, "r") as f:
            top_cuis_names_data = json.load(f)
    else:
        top_cuis_names_data = {}

    for json_file in os.listdir(histology_jsons):
        if not json_file.endswith(".json"):
            continue

        histology_name = json_file.replace(".json", "")
        print(histology_name)

        with open(os.path.join(histology_jsons, json_file), "r") as f:
            data = json.load(f)

        all_cuis = []
        for item in data:
            all_cuis.extend(item["cuis"])

        top_n_cuis = get_top_n_cuis(all_cuis, TOP_N)

        # Initialize entry for this histology
        top_cuis_data[histology_name] = {}
        top_cui_names = []

        for rank, cui in enumerate(top_n_cuis):
            entity = neo4j_query.get_entity_by_cui(cui)

            top_cui_names.append(entity['name'])
            top_cuis_data[histology_name][rank+1] = {
                "cui": cui,
                "name": entity['name'],
                "semantic_type_tui": entity['tui'],
                "semantic_type_name": entity['semantic_type']
            }

        top_cuis_names_data[histology_name] = top_cui_names

    

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(top_cuis_json), exist_ok=True)
    os.makedirs(os.path.dirname(top_cuis_names_json), exist_ok=True)
    
    # Save the complete dictionary as valid JSON
    with open(top_cuis_json, "w") as f:
        json.dump(top_cuis_data, f, indent=2)
    with open(top_cuis_names_json, "w") as f:
        json.dump(top_cuis_names_data, f, indent=2)

    print(f"Saved top {TOP_N} CUIs to {top_cuis_json}")

    # Write semantic-type summary log file (ranked by count per histology)
    log_path = top_cuis_json.replace(".json", "_by_semtype.log")
    from collections import defaultdict
    with open(log_path, "w") as log_f:
        for histology_name, ranked_entries in sorted(top_cuis_data.items()):
            log_f.write(f"{'='*60}\n")
            log_f.write(f"  {histology_name}\n")
            log_f.write(f"{'='*60}\n")

            # Count occurrences per semantic type
            semtype_counts = defaultdict(lambda: {"count": 0, "tui": ""})
            for entry in ranked_entries.values():
                st = entry["semantic_type_name"]
                semtype_counts[st]["count"] += 1
                semtype_counts[st]["tui"] = entry["semantic_type_tui"]

            # Sort by count descending
            sorted_semtypes = sorted(semtype_counts.items(), key=lambda x: -x[1]["count"])

            for rank_st, (semtype, info) in enumerate(sorted_semtypes, 1):
                log_f.write(f"  {rank_st:>2}. [{info['tui']}] {semtype}  ({info['count']})\n")

            log_f.write("\n")

    print(f"Saved semantic-type log to {log_path}")