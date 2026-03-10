#!/bin/bash
source "$(dirname "$0")/../paths_config.sh"

# Configuration
WSI_DIR="$DATA_DIR/temp_wsis"
JOB_DIR="$DATA_DIR/trident_processed"
CSV_PATH="$WSI_DIR/slide_list.csv"
LOG_FILE="$JOB_DIR/pipeline_status.log"

# Initialize or clear previous log
echo "--- TRIDENT EXECUTION LOG ($(date)) ---" > "$LOG_FILE"

echo "Starting TRIDENT Pipeline with Manual MPP Mapping..." | tee -a "$LOG_FILE"

# 1. UNI v1 Patch Encoding
echo "Running Stage 1: UNI v1..." | tee -a "$LOG_FILE"
python run_batch_of_slides.py \
    --task all \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --custom_list_of_wsis "$CSV_PATH" \
    --max_workers 8 \
    --patch_encoder conch_v15 \
    --mag 20 \
    --patch_size 512 2>&1 | tee -a "$LOG_FILE"

# Check if Stage 1 succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[SUCCESS] Stage 1: UNI v1 completed at $(date)" >> "$LOG_FILE"
else
    echo "[FAILURE] Stage 1: UNI v1 failed at $(date). Terminating pipeline." | tee -a "$LOG_FILE"
    exit 1
fi

echo "--------------------------------------" | tee -a "$LOG_FILE"

# 2. TITAN Slide Encoding
echo "Running Stage 2: TITAN..." | tee -a "$LOG_FILE"
python run_batch_of_slides.py \
    --task feat \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --custom_list_of_wsis "$CSV_PATH" \
    --max_workers 4 \
    --slide_encoder titan \
    --mag 20 \
    --patch_size 512 2>&1 | tee -a "$LOG_FILE"

# Check if Stage 2 succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[SUCCESS] Stage 2: TITAN completed at $(date)" | tee -a "$LOG_FILE"
    echo "======================================" >> "$LOG_FILE"
    echo "FULL PIPELINE COMPLETED SUCCESSFULLY" | tee -a "$LOG_FILE"
else
    echo "[FAILURE] Stage 2: TITAN failed at $(date)." | tee -a "$LOG_FILE"
    exit 1
fi