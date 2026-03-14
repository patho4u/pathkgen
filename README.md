# PathKGen

Repository for the paper PathKGen: Pathology Knowledge Assisted MLLM for Explainable Histopathology

## Repository Structure

```
PathKGen/
├── pipeline/               # Main VLM training and inference pipeline
│   ├── train_stage1.sh     # Stage 1 training script
│   ├── train_stage2.sh     # Stage 2 training script (knowledge-guided)
│   ├── inference.sh        # Inference script
│   ├── model.py            # Model architecture
│   ├── config.py           # Training configuration
│   └── ...
├── auxiliary_classifier/   # Auxiliary histology classifier (used at the start of the knowledge path)
│   ├── classifier.py
│   ├── head.py
│   ├── inference.py
│   └── ...
├── knowledge_path/         # Knowledge creation, graph traversal, and visualization
│   ├── graphs.py           # Neo4j graph query interface
│   ├── subgraph_visualization.py
│   └── graph_construct/    # CUI selection, knowledge prompt generation, and enrichment
│       ├── cui_selecter.py
│       ├── create_knowledge_prompt.py
│       └── enrich_with_knowledge.py
├── pre_processing/         # Data pre-processing utilities
│   ├── convert_wsi_bench.py
│   ├── filter_wsi_bench.py
│   ├── encode_wsis.sh      # WSI patch & slide feature encoding (TRIDENT)
│   └── ...
├── paths_config.sh         # Central path configuration (edit before running)
├── paths_config.py         # Python equivalent of the path configuration
├── environment.yml         # Conda environment specification
└── requirements.txt        # Pip requirements
```

---

## Replication Guide

> **Data and model weights will be available soon.**

### 1. Environment Setup

Clone the repository and create the conda environment:

```bash
git clone <repo-url>
cd PathKGen

conda env create -f environment.yml
conda activate vlm
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

> **Note:** Tested with CUDA 12.1. Ensure your system has a compatible CUDA toolkit installed before setting up the environment.

---

### 2. Configure Paths

Edit `paths_config.sh` to point to your local directories before running any scripts:

```bash
# paths_config.sh
VLM_ROOT="/path/to/PathKGen"
DATA_DIR="$VLM_ROOT/data"
DATASET_DIR="$DATA_DIR/dataset"
CHECKPOINTS_DIR="$VLM_ROOT/checkpoints"
RESULTS_DIR="$VLM_ROOT/results"
```

Similarly, update `paths_config.py` for any Python scripts that require it.

---

### 3. Data Download

> **Data will be available soon.**

Once released, download the data directory from [here](https://huggingface.co/datasets/deepshark43/wsi_bench_pathkgen) and place it under `PathKGen/data/` as shown:

```
data/
├── dataset/              # Dataset JSONs with pre-curated knowledge prompts
│   ├── WSI-Bench-train_Report_knowledge.json
│   ├── WSI-Bench-test_Report_knowledge.json
│   ├── WSI-Bench-train_Report_knowledge.log
│   └── WSI-Bench-test_Report_knowledge.log
└── slide_features/       # Pre-computed slide embeddings (TITAN slide-level features)
    ├── train/
    └── test/
```

> **Slide embeddings** — Patch features are extracted using CONCH v1.5 and aggregated into slide-level representations using the TITAN slide encoder. Both 1D slide features and 2D visual tokens are provided.
>
> **Knowledge prompts** — Curated from a UMLS-derived knowledge graph using auxiliary classifier predictions as anchor nodes. These are already embedded in the `*_knowledge.json` files; no separate knowledge path construction step is required.

You can proceed directly to [Training](#6-training) using the downloaded data.

---

> **Model weights will be available soon.**

Once released, download the model weights from [here](https://huggingface.co/deepshark43/pathkgen) and place them under `PathKGen/checkpoints/` as shown:

```
checkpoints/
├── pathkgen_weights/       # Trained weights for Stage 1 and Stage 2
│   ├── pathkgen_stage1/
│   └── pathkgen_stage2/
└── Qwen2.5-7B-Instruct/    # Base Qwen2.5 LLM weights
```

Download the base Qwen2.5-7B-Instruct model from [here](#) and place it under `checkpoints/` as shown above.

---

### 4. Data Pre-processing *(Optional — already done, data provided)*

> The pre-processed dataset JSONs and pre-computed slide embeddings are available for direct download (see [Section 3](#3-data-download)). The steps below are provided for reference, or if you wish to reproduce the pre-processing from scratch.

#### 4a. Convert and Filter WSI-Bench Data

```bash
cd pre_processing

# Convert raw WSI-Bench JSON/JSONL to standardised format
python convert_wsi_bench.py

# Filter dataset by task and available feature files
python filter_wsi_bench.py
```

#### 4b. Encode WSI Patch Features

WSI patch and slide-level features are extracted using [TRIDENT](https://github.com/mahmoodlab/TRIDENT) with the CONCH v1.5 patch encoder and TITAN slide encoder.

```bash
# Ensure TRIDENT is installed and run_batch_of_slides.py is accessible.
# Copy encode_wsis.sh into the TRIDENT directory and run it from there.
bash encode_wsis.sh
```

This runs two stages:
1. **Stage 1** — Patch encoding with CONCH v1.5 at 20× magnification
2. **Stage 2** — Slide-level encoding with TITAN

Logs are saved to `data/trident_processed/pipeline_status.log`.

> **Note:** The TITAN repository must be modified to also retrieve and save the 2D visual tokens in addition to the standard 1D slide features.

---

### 5. Knowledge Path Construction *(Optional — already done, data provided)*

> Pre-curated knowledge prompts are already included in the `*_knowledge.json` dataset files available for download. The entire pipeline — including auxiliary classifier inference — has already been run and its outputs are bundled with the dataset. The steps below are provided for reference only.

The knowledge path enriches each sample with contextual knowledge derived from the knowledge graph prior to training. The relevant code is located in `auxiliary_classifier/` and `knowledge_path/`.

---

### 6. Training

Training is split into two stages.

#### Stage 1 — Projection Alignment

Trains the projection layer that maps WSI features into the LLM token space. The LLM backbone is frozen during this stage.

```bash
cd pipeline
bash train_stage1.sh
```

#### Stage 2 — Knowledge-Guided Fine-tuning

Fine-tunes the full model (projection layer + LoRA adapters on the LLM) with knowledge-augmented inputs and BLEU-penalised loss.

```bash
cd pipeline
bash train_stage2.sh
```

> Update the `STAGE1_CHECKPOINT` variable in `train_stage2.sh` to point to your best Stage 1 checkpoint before running.

---

### 7. Inference

Run inference on the WSI-Bench test set:

```bash
cd pipeline
bash inference.sh
```

Generated reports are saved to `results/inference_wsi_bench/`.

---

### 8. Evaluation

Plot training curves:

```bash
python plot_training.py
```

---
