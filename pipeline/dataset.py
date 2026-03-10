import os
import json
import sys
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR

class WSI_Report_Dataset(Dataset):
    def __init__(self, json_path, dataset_format='wsi-bench', split='train', feature_key='tokens', use_knowledge_guidance=False):
        """
        Custom Dataset for loading Whole Slide Image (WSI) features and corresponding reports.
        
        Args:
            json_path (str): Path to the JSON file containing dataset information.
            dataset_format (str): Format of the dataset - 'wsi-bench' (flat list) or 'histgen' (nested dict).
            split (str or None): Dataset split to use ('train', 'val', 'test'). 
                                 Set to None for WSI-Bench (already pre-split files).
            feature_key (str): Key in the H5 file where slide features are stored.
            use_knowledge_guidance (bool): Whether to use knowledge-guided prompts.
        """
        self.feature_key = feature_key
        self.split = split
        self.use_knowledge_guidance = use_knowledge_guidance
        self.dataset_format = dataset_format

        # Load dataset information from JSON
        with open(json_path, 'r') as info_file:
            data = json.load(info_file)
        
        # Handle different dataset formats
        if dataset_format == 'histgen':
            # HistGen format: nested dict with splits {"train": [...], "val": [...], "test": [...]}
            if split is None:
                raise ValueError("split must be specified for HistGen format")
            
            if isinstance(split, str):
                self.dataset = data[split]
            elif isinstance(split, (list, tuple, set)):
                self.dataset = []
                for s in split:
                    self.dataset.extend(data[s])
            else:
                raise ValueError("split must be a string or a list/tuple/set of strings.")
            
            # Field mappings for HistGen format
            self.answer_key = 'report'
            self.has_questions = False
            
        elif dataset_format == 'wsi-bench':
            # WSI-Bench format: flat list OR dict (already split into separate files)
            if isinstance(data, dict):
                self.dataset = list(data.values())
            elif isinstance(data, list):
                self.dataset = data
            else:
                raise ValueError(
                    f"WSI-Bench format expects a flat list or dict, got {type(data)}. "
                    f"Are you using the correct dataset_format?"
                )
            
            # Field mappings for WSI-Bench format
            self.answer_key = 'T-answer'
            self.has_questions = True
            
        else:
            raise ValueError(f"Unknown dataset_format: {dataset_format}. Use 'histgen' or 'wsi-bench'.")
        
        # Filter out samples without feature_path (some val samples may not have features extracted)
        original_len = len(self.dataset)
        self.dataset = [item for item in self.dataset if 'feature_path' in item]
        filtered_len = len(self.dataset)
        
        if filtered_len < original_len:
            print(f"Warning: Filtered out {original_len - filtered_len} samples without feature_path")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a single data point from the dataset.
        """
        item = self.dataset[idx]
        # feature_path is stored relative to DATA_DIR for portability;
        # reconstruct the absolute path at load time.
        feature_path = os.path.join(DATA_DIR, item['feature_path'])
        
        # Extract question and answer based on dataset format
        if self.has_questions:
            # WSI-Bench format: has question field.
            # Strip legacy <image>\n prefix — image is already embedded via <image_start><image_end>
            # tokens in the prompt template; including the raw tag would create duplicate garbage tokens.
            question = item['question']
            if question.startswith('<image>'):
                question = question[len('<image>'):].lstrip('\n').strip()
            answer = item[self.answer_key]  # 'T-answer'
        else:

            # HistGen format: no question field, will use fixed prompt
            question = None
            answer = item[self.answer_key]  # 'report'

        # Get pre-computed knowledge text if using knowledge guidance
        if self.use_knowledge_guidance:
            knowledge_text = item.get("knowledge", "")
        else:
            knowledge_text = ""

        # Load slide features from H5 file
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        with h5py.File(feature_path, 'r') as h5_file:
            slide_features = h5_file[self.feature_key][:]
            
            # Validate shape: must be 2D spatial features [num_tokens, feature_dim]
            if slide_features.ndim != 2:
                raise ValueError(
                    f"Feature file {feature_path} has shape {slide_features.shape}. "
                    f"Expected 2D spatial features [num_tokens, feature_dim]."
                )
            
            slide_embedding = torch.tensor(slide_features, dtype=torch.float32)
            # Shape: [128, 768] or [N, 768] depending on encoder

        # Get sample metadata
        slide_id = item.get('tcga_id', item.get('id', str(idx)))
        organ = item.get('primary_site', item.get('organ', 'unknown'))

        return {
            'id': slide_id,
            'organ': organ,
            'slide_embedding': slide_embedding,
            'question': question,  # None for HistGen, actual question for WSI-Bench
            'answer': answer,      # 'report' for HistGen, 'T-answer' for WSI-Bench
            'knowledge_text': knowledge_text
        }
            


if __name__ == "__main__":
    import sys
    from config import WSI_BENCH_TRAIN_KNOWLEDGE, WSI_BENCH_TEST_KNOWLEDGE

    PASS = "\033[92m\u2713\033[0m"
    FAIL = "\033[91m\u2717\033[0m"
    errors = []

    def check(name, condition, detail=""):
        if condition:
            print(f"  {PASS} {name}")
        else:
            print(f"  {FAIL} {name}" + (f": {detail}" if detail else ""))
            errors.append(name)

    for split_name, json_path in [("TRAIN", WSI_BENCH_TRAIN_KNOWLEDGE),
                                   ("TEST",  WSI_BENCH_TEST_KNOWLEDGE)]:
        print(f"\n{'=' * 55}")
        print(f"WSI-Bench Knowledge Dataset - {split_name}: {json_path}")
        print("=" * 55)

        # --- Without knowledge guidance ---
        ds_nk = WSI_Report_Dataset(json_path, dataset_format='wsi-bench',
                                   split=None, feature_key='tokens',
                                   use_knowledge_guidance=False)
        print(f"\n  Length (no knowledge): {len(ds_nk)}")
        check("dataset non-empty", len(ds_nk) > 0)

        item = ds_nk[0]
        print(f"  id            : {item['id']}")
        print(f"  organ         : {item['organ']}")
        print(f"  question      : {item['question'][:80]!r}")
        print(f"  answer (T-ans): {item['answer'][:80]!r}")
        print(f"  knowledge_text: {repr(item['knowledge_text'])[:60]}")
        print(f"  embedding shape: {item['slide_embedding'].shape}")

        check("question is non-empty string", isinstance(item['question'], str) and len(item['question']) > 5)
        check("answer is T-answer (non-empty)", isinstance(item['answer'], str) and len(item['answer']) > 5)
        check("knowledge_text is empty when guidance=False", item['knowledge_text'] == "")
        check("slide_embedding is 2D tensor", item['slide_embedding'].ndim == 2)

        # --- With knowledge guidance ---
        ds_k = WSI_Report_Dataset(json_path, dataset_format='wsi-bench',
                                  split=None, feature_key='tokens',
                                  use_knowledge_guidance=True)
        item_k = ds_k[0]
        print(f"\n  With knowledge:")
        print(f"  knowledge_text: {item_k['knowledge_text'][:120]!r}")
        check("knowledge_text non-empty with guidance=True",
              isinstance(item_k['knowledge_text'], str) and len(item_k['knowledge_text']) > 10,
              f"got: {item_k['knowledge_text']!r}")
        check("knowledge starts with <knowledge>",
              item_k['knowledge_text'].strip().startswith("<knowledge>"),
              f"starts with: {item_k['knowledge_text'][:20]!r}")
        check("T-answer preserved with knowledge", item_k['answer'] == item['answer'])

    print(f"\n{'=' * 55}")
    if errors:
        print(f"\033[91mFAILED ({len(errors)}): {errors}\033[0m")
        sys.exit(1)
    else:
        print("\033[92mAll dataset tests passed!\033[0m")
    print("=" * 55)