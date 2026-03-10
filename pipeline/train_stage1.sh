#!/bin/bash
# Stage 1 Training script for WSI-Bench dataset
source "$(dirname "$0")/../paths_config.sh"

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 1 Training with WSI-Bench Dataset"
echo "================================================="

# WSI-Bench dataset paths
WSI_BENCH_TRAIN="$DATASET_DIR/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
WSI_BENCH_TEST="$DATASET_DIR/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

KNOWLEDGE_TRAIN_JSON="$DATASET_DIR/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="$DATASET_DIR/WSI-Bench-test_Report_knowledge.json"

OUTPUT_DIR="$CHECKPOINTS_DIR/stage1_wsi_bench"

python train_stage1.py \
    --train_json "$WSI_BENCH_TRAIN" \
    --test_json "$WSI_BENCH_TEST" \
    --dataset_format wsi-bench \
    --epochs 8 \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --lr 5e-4 \
    --early_stopping \
    --patience 2 \
    --weight_decay 0.01 \
    --feature_key tokens \
    --gradient_checkpointing \
    --max_length 1024 \
    --val_batch_size 1 \
    --output_dir "$OUTPUT_DIR" \

echo ""
echo "Stage 1 training complete! Checkpoints saved to: $OUTPUT_DIR"
