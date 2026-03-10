#!/bin/bash
# Inference script for WSI-Bench test set
source "$(dirname "$0")/../paths_config.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

KNOWLEDGE_TEST_JSON="$DATASET_DIR/WSI-Bench-test_Report_knowledge.json"
UMLS_LLM_PATH="$CHECKPOINTS_DIR/Qwen-UMLS-7B-Instruct"
STAGE2_CHECKPOINT="$CHECKPOINTS_DIR/q7b_best/s2/checkpoint_epoch_1.pt"
OUTPUT_DIR="$RESULTS_DIR/inference_wsi_bench"

echo "Starting Inference with WSI-Bench Dataset"
echo "================================================="
echo "  Test JSON  : $KNOWLEDGE_TEST_JSON"
echo "  Output dir : $OUTPUT_DIR"
echo "================================================="
echo ""

python inference.py \
    --stage2_checkpoint "$STAGE2_CHECKPOINT" \
    --llm_path          "$UMLS_LLM_PATH" \
    --data_json         "$KNOWLEDGE_TEST_JSON" \
    --dataset_format    wsi-bench \
    --feature_key       tokens \
    --val_batch_size    4  \
    --num_workers       8 \
    --max_new_tokens    512 \
    --num_beams         4 \
    --length_penalty    1.0 \
    --no_repeat_ngram_size 3 \
    --use_knowledge_guidance \
    --output_dir        "$OUTPUT_DIR"

echo ""
echo "Inference complete! Results saved to: $OUTPUT_DIR"