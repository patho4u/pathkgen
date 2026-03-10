#!/bin/bash
# Training script for WSI-Bench dataset
source "$(dirname "$0")/../paths_config.sh"

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 2 Training with WSI-Bench Dataset (NLU Optimized)"
echo "=================================================================="

# WSI-Bench dataset paths
WSI_BENCH_TRAIN="$DATASET_DIR/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
WSI_BENCH_TEST="$DATASET_DIR/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

KNOWLEDGE_TRAIN_JSON="$DATASET_DIR/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="$DATASET_DIR/WSI-Bench-test_Report_knowledge.json"

DEFAULT_LLM_PATH="$CHECKPOINTS_DIR/Qwen2.5-7B-Instruct"
UMLS_LLM_PATH="$CHECKPOINTS_DIR/Qwen-UMLS-7B-Instruct"

# Configuration
OUTPUT_DIR="$CHECKPOINTS_DIR/stage2_wsi_bench_knowledge"
STAGE1_CHECKPOINT="$CHECKPOINTS_DIR/stage1_wsi_bench_knowledge/2026-02-24_10-55_vt128_ml768_e10_bs4/best_model.pt"
STAGE2_RESULTS_DIR="$RESULTS_DIR/stage2_wsi_bench"

# Retry logic: repeat if error occurs
python train_stage2.py \
    --train_json "$KNOWLEDGE_TRAIN_JSON" \
    --test_json  "$KNOWLEDGE_TEST_JSON" \
    --llm_path   "$UMLS_LLM_PATH" \
    --dataset_format wsi-bench \
    --use_knowledge_guidance \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --output_results_dir "$STAGE2_RESULTS_DIR" \
    --compute_metrics_during_training \
    --use_bleu_penalty \
    --bleu_metric bleu4 \
    --label_smoothing 0.1 \
    --warmup_ratio 0.1 \
    --eval_num_beams 4 \
    --eval_length_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --projection_lr 1e-5 \
    --weight_decay 0.01 \
    --early_stopping \
    --patience 1 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 4 \
    --output_dir "$STAGE2_OUTPUT_DIR"



if [ $? -eq 0 ]; then
    echo ""
    echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"
else
    echo ""
    echo "Error occurred during training."
fi
