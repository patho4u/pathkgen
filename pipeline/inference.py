"""
Inference script for generating pathology reports from WSI slides and computing NLU metrics.

Supports Stage 1 (projection-only) and Stage 2 (projection + QLoRA) checkpoints.

Usage:
    # Stage 2 checkpoint (projection + LoRA)
    python pipeline/inference.py \\
        --stage2_checkpoint /path/to/stage2/best_model.pt \\
        --output_dir /path/to/outputs

    # Stage 1 checkpoint (projection only)
    python pipeline/inference.py \\
        --stage1_checkpoint /path/to/stage1/best_model.pt \\
        --output_dir /path/to/outputs

    # Custom dataset and generation settings
    python pipeline/inference.py \\
        --stage2_checkpoint /path/to/stage2/best_model.pt \\
        --data_json /path/to/test.json \\
        --output_dir /path/to/outputs \\
        --num_beams 4 \\
        --use_knowledge_guidance
"""

import os
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local imports
from model import VisionLanguageModel
from dataset import WSI_Report_Dataset
from data_collator import VLMDataCollator
from metrics import compute_metrics
from config import (
    IMAGE_START_TOKEN, IMAGE_END_TOKEN,
    DEFAULT_LLM_PATH,
    DEFAULT_VISUAL_DIM,
    DEFAULT_NUM_VISUAL_TOKENS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    extract_report_from_labels,
    DATASET_DIR,
)

# Default test dataset path
DEFAULT_TEST_JSON = os.path.join(
    DATASET_DIR,
    "WSI-Bench-test_Report_knowledge.json"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM Inference: generate reports and compute NLU metrics"
    )

    # ── Checkpoint (exactly one required) ───────────────────────────────────
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--stage1_checkpoint", type=str, default=None,
        help="Path to Stage 1 best_model.pt (projection weights only, no LoRA)."
    )
    ckpt_group.add_argument(
        "--stage2_checkpoint", type=str, default=None,
        help="Path to Stage 2 best_model.pt (projection + QLoRA weights)."
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--llm_path", type=str, default=DEFAULT_LLM_PATH,
        help="Path to the base LLM (same one used during training)."
    )
    parser.add_argument(
        "--visual_dim", type=int, default=DEFAULT_VISUAL_DIM,
        help="Visual feature dimension (default: 768 for TITAN)."
    )
    parser.add_argument(
        "--num_visual_tokens", type=int, default=DEFAULT_NUM_VISUAL_TOKENS,
        help="Number of visual tokens fed to the LLM."
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_json", type=str, default=DEFAULT_TEST_JSON,
        help="Path to the test dataset JSON file."
    )
    parser.add_argument(
        "--dataset_format", type=str, default="wsi-bench",
        choices=["histgen", "wsi-bench"],
        help="Dataset format."
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Dataset split to use (HistGen format only, e.g. 'test')."
    )
    parser.add_argument(
        "--feature_key", type=str, default="tokens",
        help="Key in H5 feature files for slide embeddings."
    )
    parser.add_argument(
        "--use_knowledge_guidance", action="store_true",
        help="Pass knowledge text to the model during generation."
    )
    parser.add_argument(
        "--max_length", type=int, default=DEFAULT_MAX_LENGTH,
        help="Maximum tokenised sequence length for the data collator."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=4,
        help="Batch size for the inference DataLoader."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes."
    )

    # ── Generation ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per sample."
    )
    parser.add_argument(
        "--num_beams", type=int, default=1,
        help="Number of beams for beam search (1 = greedy)."
    )
    parser.add_argument(
        "--length_penalty", type=float, default=1.0,
        help="Length penalty for beam search (>1 favours longer outputs)."
    )
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=3,
        help="Block repeated n-grams of this size during generation."
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (ignored when num_beams > 1)."
    )
    parser.add_argument(
        "--top_p", type=float, default=DEFAULT_TOP_P,
        help="Nucleus sampling p (ignored when num_beams > 1)."
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY,
        help="Repetition penalty."
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help=(
            "Directory to save inference outputs. "
            "Creates: predictions.json, metrics.json"
        )
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, device):
    """
    Load the VisionLanguageModel and the requested checkpoint.

    Stage 1 checkpoint  → use_qlora=False, load 'projection_state_dict' only.
    Stage 2 checkpoint  → use_qlora=True,  load 'projection_state_dict' + 'lora_state_dict'.

    Model is initialised on CPU first, then moved to *device* after weight loading
    (mirrors evaluate_best_model in train_stage2.py to avoid GPU OOM during load).
    """
    is_stage2 = args.stage2_checkpoint is not None
    checkpoint_path = args.stage2_checkpoint if is_stage2 else args.stage1_checkpoint
    stage_label = "Stage 2 (QLoRA)" if is_stage2 else "Stage 1 (projection-only)"

    print(f"\n{'='*60}")
    print(f"Loading model for inference  [{stage_label}]")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  LLM        : {args.llm_path}")
    print(f"{'='*60}")

    # Initialise model entirely on CPU first to keep GPU headroom for weights
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        slidechat_checkpoint=None,       # Not needed — weights are in checkpoint
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        use_qlora=is_stage2,
        freeze_llm=True,                 # Inference only
        gradient_checkpointing=False,
        device="cpu",
    )

    # Load checkpoint to CPU, then selectively restore weights
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Projection weights (always present)
    model.projection.load_state_dict(checkpoint["projection_state_dict"])
    print("  ✓ Projection weights loaded")

    # LoRA weights (Stage 2 only)
    if is_stage2:
        if "lora_state_dict" in checkpoint:
            model.llm.load_state_dict(checkpoint["lora_state_dict"], strict=False)
            print("  ✓ LoRA weights loaded")
        else:
            print("  ! Warning: 'lora_state_dict' not found in Stage 2 checkpoint")

    # Log checkpoint metadata if available
    if "epoch" in checkpoint:
        print(f"  Checkpoint epoch : {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
        print(f"  Checkpoint val loss : {checkpoint['val_loss']:.4f}")

    # Free checkpoint memory before moving model to GPU
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    # Move model to target device
    print(f"\nMoving model to {device} ...")
    model = model.to(device)
    model.eval()

    print("✓ Model ready.\n")
    return model


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, dataloader, args, device, tokenizer):
    """
    Iterate over the dataset, generate reports, collect predictions and references.

    Mirrors the compute_metrics=True branch of validate() in train_stage2.py.
    Returns lists of (ids, predictions, references).
    """
    all_ids         = []
    all_predictions = []
    all_references  = []

    pbar = tqdm(dataloader, desc="Generating reports")

    for batch in pbar:
        visual_features = batch["visual_features"]   # [B, N, D]
        labels          = batch["labels"]             # [B, seq_len]
        batch_size      = visual_features.shape[0]

        for i in range(batch_size):
            single_visual = visual_features[i:i+1].to(device)

            # Per-sample question / knowledge text
            question_text  = batch["questions"][i]          if "questions"      in batch else None
            knowledge_text = batch["knowledge_texts"][i]    if (
                "knowledge_texts" in batch and args.use_knowledge_guidance
            ) else None

            # Generate — settings exactly match validate() in train_stage2.py:
            #   temperature=0.0  (greedy / beam; never sample during eval)
            #   top_p=1.0        (no nucleus filtering)
            #   repetition_penalty not passed (matches training eval default)
            pred_text = model.generate(
                visual_features=single_visual,
                prompt=None,            # auto-selects prompt from config
                question=question_text,
                knowledge_text=knowledge_text,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,        # always greedy/beam — matches training eval
                top_p=1.0,             # no nucleus filtering — matches training eval
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            all_predictions.append(pred_text)

            # Ground-truth reference from labels
            label_ids = labels[i][labels[i] != -100]
            ref_text  = extract_report_from_labels(tokenizer, label_ids)
            all_references.append(ref_text)

            # Sample ID
            sample_id = batch["ids"][i] if "ids" in batch else str(len(all_ids))
            all_ids.append(sample_id)

            # Free per-sample GPU allocation
            del single_visual
            torch.cuda.empty_cache()

        pbar.set_postfix({"generated": len(all_ids)})

    return all_ids, all_predictions, all_references


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Output directory ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add image marker special tokens (must match model.py)
    special_tokens_dict = {
        "additional_special_tokens": [IMAGE_START_TOKEN, IMAGE_END_TOKEN]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    print(f"  ✓ Tokenizer loaded, special tokens added: {IMAGE_START_TOKEN}, {IMAGE_END_TOKEN}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.data_json}")
    split = args.split if args.dataset_format == "histgen" else None

    dataset = WSI_Report_Dataset(
        json_path=args.data_json,
        dataset_format=args.dataset_format,
        split=split,
        feature_key=args.feature_key,
        use_knowledge_guidance=args.use_knowledge_guidance,
    )
    print(f"  ✓ {len(dataset)} samples loaded")

    collator = VLMDataCollator(
        tokenizer=tokenizer,
        dataset_format=args.dataset_format,
        max_length=args.max_length,
        use_knowledge_guidance=args.use_knowledge_guidance,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model(args, device)

    # ── Inference ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Running inference ...")
    print("=" * 60)

    all_ids, all_predictions, all_references = run_inference(
        model, dataloader, args, device, tokenizer
    )

    print(f"\n✓ Generated {len(all_predictions)} reports")

    # ── Save predictions ─────────────────────────────────────────────────────
    records = [
        {"id": sid, "prediction": pred, "reference": ref}
        for sid, pred, ref in zip(all_ids, all_predictions, all_references)
    ]

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(records, f, indent=4)
    print(f"✓ Predictions saved → {predictions_path}")

    # ── Compute NLU metrics ──────────────────────────────────────────────────
    if len(all_predictions) > 0 and len(all_references) > 0:
        print("\nComputing NLU metrics ...")
        metrics = compute_metrics(all_predictions, all_references)

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(
            f"  BLEU-1 : {metrics.get('bleu1', 0):.4f}   "
            f"BLEU-2 : {metrics.get('bleu2', 0):.4f}   "
            f"BLEU-3 : {metrics.get('bleu3', 0):.4f}   "
            f"BLEU-4 : {metrics.get('bleu4', 0):.4f}"
        )
        print(
            f"  METEOR : {metrics.get('meteor', 0):.4f}   "
            f"ROUGE-L: {metrics.get('rouge_l', 0):.4f}"
        )
        print("=" * 60)

        # Save metrics
        metrics_payload = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.stage2_checkpoint or args.stage1_checkpoint,
            "data_json": args.data_json,
            "num_samples": len(all_predictions),
            "generation_config": {
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "length_penalty": args.length_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "temperature": args.temperature,
                "use_knowledge_guidance": args.use_knowledge_guidance,
            },
            "metrics": metrics,
        }

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=4)
        print(f"✓ Metrics saved      → {metrics_path}")
    else:
        print("No predictions to evaluate.")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
