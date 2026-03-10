import json
import os
import sys
import pandas as pd

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths_config import DATA_DIR


descriptions_path = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "descriptions.json")
morphs_path = os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv")
output_path = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "histology_descriptions_with_features.json")

with open(descriptions_path) as f:
    descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} descriptions from {descriptions_path}")

histologies = {}
data = {}
df = pd.read_csv(morphs_path)

for _, row in df.iterrows():
    histology = row['reference_entity']
    relation = row['relation_name']
    if histology not in histologies:
        histologies[histology] = {}
    if relation not in histologies[histology]:
        histologies[histology][relation] = []
    
    feature = row['related_entity']
    histologies[histology][relation].append(feature)

r = {'found_in': 'found in', 
     'has_growth_pattern': 'has growth pattern', 
     'has_cytological_feature': 'has cytological feature',
     'has_invasion_pattern': 'has invasion pattern', 
     'has_background_feature': 'has background feature', 
     'has_nuclear_feature': 'has nuclear feature', 
     'has_mitotic_activity': 'has mitotic activity'}


for histology, features in histologies.items():
    knowledge_prompt = f"<knowledge>\nHistology: {histology}\n"
    for relation, feature_list in features.items():
        if feature_list:
            # paragraph_parts.append(f"{r.get(relation, relation)}: {', '.join(feature_list)}.")
            knowledge_prompt += f"{r.get(relation, relation)}: {', '.join(feature_list)}\n"
    knowledge_prompt += "</knowledge>"
    print(f"Histology: {histology}\n{knowledge_prompt}\n")
    data[histology] = knowledge_prompt

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)
print(f"Saved histology descriptions with features to {output_path}")
