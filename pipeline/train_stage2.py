"""
Stage 2 Training: Projection + LLM with QLoRA

Fine-tunes both the projection layer and LLM using QLoRA for memory efficiency.

Usage:
    python pipeline/train_stage2.py \
        --stage1_checkpoint /path/to/stage1/best_model.pt \
        --epochs 5 \
        --batch_size 2 \
        --lr 2e-5
"""

import os
import math
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import AutoTokenizer

# Local imports
from model import VisionLanguageModel
from dataset import WSI_Report_Dataset
from data_collator import VLMDataCollator
from config import (GENERATION_PROMPT, TRAINING_PROMPT_TEMPLATE,
                    DEFAULT_LLM_PATH, DEFAULT_SLIDECHAT_CHECKPOINT,
                    DEFAULT_DATA_JSON, DEFAULT_STAGE2_OUTPUT,
                    DEFAULT_VISUAL_DIM, DEFAULT_NUM_VISUAL_TOKENS,
                    DEFAULT_MAX_LENGTH, DEFAULT_MAX_NEW_TOKENS,
                    extract_report_from_labels)
from metrics import compute_bleu_penalty

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Training: Projection + QLoRA")
    
    # Data arguments
    parser.add_argument("--data_json", type=str, 
                       default=DEFAULT_DATA_JSON,
                       help="Path to dataset JSON (HistGen format)")
    parser.add_argument("--train_json", type=str, default=None,
                       help="Path to training JSON (WSI-Bench format)")
    parser.add_argument("--test_json", type=str, default=None,
                       help="Path to test JSON (WSI-Bench format)")
    parser.add_argument("--dataset_format", type=str, default="histgen",
                       choices=["histgen", "wsi-bench"],
                       help="Dataset format: 'histgen' (nested dict with splits) or 'wsi-bench' (flat list with questions)")
    parser.add_argument("--train_split", type=str, default="train,val",
                       help="Split(s) to use for training (HistGen only). Can be 'train' or 'train,val'")
    parser.add_argument("--val_split", type=str, default="test",
                       help="Split to use for validation (HistGen only)")
    parser.add_argument("--feature_key", type=str, default="tokens",
                       help="Key in H5 files for slide features (use 'tokens' for spatial features)")
    
    # Model arguments
    parser.add_argument("--llm_path", type=str,
                       default=DEFAULT_LLM_PATH,
                       help="Path to base LLM")
    parser.add_argument("--slidechat_checkpoint", type=str,
                       default=DEFAULT_SLIDECHAT_CHECKPOINT,
                       help="Path to SlideChat weights")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                       help="Path to Stage 1 checkpoint (for projection weights). "
                            "Required unless --init_from is provided.")
    parser.add_argument("--visual_dim", type=int, default=DEFAULT_VISUAL_DIM,
                       help="Dimension of slide features")
    parser.add_argument("--num_visual_tokens", type=int, default=DEFAULT_NUM_VISUAL_TOKENS,
                       help="Number of visual tokens")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size (small for QLoRA)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                       help="Validation batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate for LLM LoRA")
    parser.add_argument("--projection_lr", type=float, default=1e-4,
                       help="Learning rate for projection layer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--use_bleu_penalty", action="store_true",
                       help="Enable BLEU-based penalty loss during training")
    parser.add_argument("--bleu_penalty_weight", type=float, default=2,
                       help="Weight for BLEU penalty loss (total_loss = ce_loss + bleu_penalty_weight * bleu_penalty)")
    parser.add_argument("--bleu_metric", type=str, default="bleu2",
                       choices=["bleu1", "bleu2", "bleu3", "bleu4"],
                       help="Which BLEU metric to use for penalty")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="DataLoader workers")
    
    # Knowledge guidance arguments
    parser.add_argument("--use_knowledge_guidance", action="store_true",
                       help="Enable knowledge graph guidance for report generation")
    parser.add_argument("--max_knowledge_tokens", type=int, default=64,
                       help="Maximum length for knowledge text tokens")
    
    # QLoRA arguments (match WSI-LLaVA configuration)
    parser.add_argument("--lora_rank", type=int, default=128,
                       help="LoRA rank (WSI-LLaVA uses 128)")
    parser.add_argument("--lora_alpha", type=int, default=256,
                       help="LoRA alpha (WSI-LLaVA uses 256)")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, 
                       default=DEFAULT_STAGE2_OUTPUT,
                       help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=1,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from Stage 2 checkpoint (restores epoch + optimizer)")
    parser.add_argument("--init_from", type=str, default=None,
                       help="Warm-start: load projection + LoRA weights from a previous Stage 2 checkpoint. "
                            "Epoch resets to 0, optimizer is fresh. Skips --stage1_checkpoint load. "
                            "Useful for rapid prototyping with new data/hyperparams.")
    
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping based on validation loss")
    parser.add_argument("--patience", type=int, default=3,
                       help="Number of epochs with no improvement before stopping (if early_stopping enabled)")
    parser.add_argument("--min_delta", type=float, default=0.001,
                       help="Minimum change in validation loss to qualify as improvement")
    
    # Misc
    parser.add_argument("--compute_metrics_during_training", action="store_true",
                       help="Compute NLP metrics during validation (slow). If False, only compute AR loss and evaluate best model at end.")
    parser.add_argument("--skip_final_metrics", action="store_true",
                       help="Skip computing NLP metrics at end of training (only use if metrics computed during training).")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log_every", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--output_results_dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                       help="Directory to save per-epoch prediction/reference JSONL files")
    
    
    # NLU Score Improvement Arguments
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                       help="Label smoothing factor (0.0 = no smoothing, 0.1 = recommended)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                       help="Ratio of total steps for linear warmup (e.g., 0.1 = 10%% warmup)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of warmup steps (overrides warmup_ratio if > 0)")
    
    # Generation improvements for evaluation
    parser.add_argument("--eval_num_beams", type=int, default=1,
                       help="Number of beams for beam search during evaluation (1 = greedy)")
    parser.add_argument("--eval_length_penalty", type=float, default=1.0,
                       help="Length penalty for beam search (>1 favors longer, <1 favors shorter)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                       help="Size of n-grams that cannot repeat during generation")   
    return parser.parse_args()



def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def compute_bleu_penalty_loss(
    model,
    visual_features,
    labels,
    tokenizer,
    input_ids,
    attention_mask,
    bleu_metric='bleu4',
    max_gen_tokens=128,
    baseline_bleu=None,
    questions=None,       # Bug 1 fix: per-sample question strings
    knowledge_texts=None, # Bug 1 fix: per-sample knowledge strings
):
    """
    Compute BLEU-weighted CE loss (hard-example mining style).

    Architecture:
        - Generate greedily for each sample (matches eval)
        - Compute BLEU(generated, reference)
        - Weight = 0.5 + sigmoid(−5 × (bleu − baseline))) → [0.5, 1.5]
          (high BLEU → weight < 1; low BLEU → weight > 1, harder sample gets more CE)
        - Return mean(weight_i × CE_i) — replaces normal CE for this batch

    Args:
        model:           VisionLanguageModel
        visual_features: [batch, num_tokens, visual_dim]
        labels:          [batch, seq_len] — -100 masked to CE targets
        tokenizer:       for decoding references
        input_ids:       [batch, seq_len] prompt token ids
        attention_mask:  [batch, seq_len]
        bleu_metric:     'bleu1' .. 'bleu4'
        max_gen_tokens:  generation budget
        baseline_bleu:   EMA baseline BLEU; None → initialise at 0.2
        questions:       list[str | None] — per-sample question text
        knowledge_texts: list[str | None] — per-sample knowledge text

    Returns:
        (bleu_loss, avg_bleu) or (None, 0.0) when all samples are skipped
    """
    batch_size = visual_features.shape[0]
    bleu_scores = []
    weighted_losses = []
    current_baseline = baseline_bleu if baseline_bleu is not None else 0.2

    model.eval()
    for i in range(batch_size):
        single_visual = visual_features[i:i+1]
        single_labels = labels[i]
        single_input_ids = input_ids[i:i+1]
        single_attention_mask = attention_mask[i:i+1]

        # Decode reference
        label_ids = single_labels[single_labels != -100]
        reference_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        if len(reference_text.split()) < 3:
            continue  # too short to compute meaningful BLEU

        # BUG 1 FIX: use the actual per-sample question/knowledge so the
        # generation matches the evaluation prompt exactly.
        q_text = questions[i] if questions is not None else None
        k_text = knowledge_texts[i] if knowledge_texts is not None else None

        with torch.no_grad():
            generated_text = model.generate(
                visual_features=single_visual,
                prompt=None,             # auto-select based on dataset
                question=q_text,         # per-sample question
                knowledge_text=k_text,   # per-sample knowledge
                max_new_tokens=max_gen_tokens,
                temperature=0.0,         # greedy — matches eval
                num_beams=1,
            )

        bleu_score = compute_bleu_penalty(generated_text, reference_text, metric=bleu_metric)
        bleu_scores.append(bleu_score)

        # Hard-example weight: inverse sigmoid of (bleu - baseline)
        # reward < 0  →  weight closer to 1.5  (focus on this sample)
        # reward > 0  →  weight closer to 0.5  (sample is already good)
        reward = bleu_score - current_baseline
        weight = 0.5 + 1.0 / (1.0 + math.exp(5.0 * reward))  # → [0.5, 1.5]

        model.train()
        outputs = model(
            visual_features=single_visual,
            input_ids=single_input_ids,
            attention_mask=single_attention_mask,
            labels=single_labels.unsqueeze(0),
        )
        model.eval()

        weighted_losses.append(weight * outputs['loss'])
        del outputs

    model.train()  # ensure training mode before returning

    if not weighted_losses:
        return None, 0.0   # signal: no valid samples → fall back to ce_loss

    bleu_loss = torch.stack(weighted_losses).mean()
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return bleu_loss, avg_bleu


def train_epoch(model, dataloader, optimizer, scheduler, epoch, args, device, tokenizer=None, running_bleu_baseline=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_bleu_penalty = 0.0
    bleu_scores_epoch = []
    optimizer.zero_grad()
    
    # Dynamic baseline for BLEU penalty (exponential moving average)
    if running_bleu_baseline is None:
        running_bleu_baseline = 0.2  # Initial estimate
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        visual_features = batch['visual_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Extract knowledge tokens if present
        knowledge_input_ids = batch.get('knowledge_input_ids')
        knowledge_attention_mask = batch.get('knowledge_attention_mask')
        if knowledge_input_ids is not None:
            knowledge_input_ids = knowledge_input_ids.to(device)
            knowledge_attention_mask = knowledge_attention_mask.to(device)
        
        # Standard forward pass
        outputs = model(
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            knowledge_input_ids=knowledge_input_ids,
            knowledge_attention_mask=knowledge_attention_mask
        )
        ce_loss = outputs['loss']
        del outputs
        
        # Compute BLEU-weighted CE penalty if enabled (hard-example mining).
        # Only evaluated every gradient_accumulation_steps to amortise generate() cost.
        bleu_computed = False   # Bug 2 fix: explicit flag instead of requires_grad check
        bleu_penalty = None
        if args.use_bleu_penalty and tokenizer is not None:
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == 0:
                # Bug 1 fix: pass per-sample question/knowledge to generate()
                questions_batch = batch.get('questions', None)
                knowledge_batch = batch.get('knowledge_texts', None)
                bleu_penalty, avg_bleu = compute_bleu_penalty_loss(
                    model=model,
                    visual_features=visual_features,
                    labels=labels,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bleu_metric=args.bleu_metric,
                    max_gen_tokens=128,
                    baseline_bleu=running_bleu_baseline,
                    questions=questions_batch,
                    knowledge_texts=knowledge_batch,
                )
                if bleu_penalty is not None:   # None when all samples skipped
                    bleu_computed = True
                    bleu_scores_epoch.append(avg_bleu)
                    running_bleu_baseline = 0.9 * running_bleu_baseline + 0.1 * avg_bleu
                    total_bleu_penalty += bleu_penalty.detach().item()
                    tqdm.write(f"  [Step {step}] BLEU: {avg_bleu:.4f}, baseline: {running_bleu_baseline:.4f}")

        # Loss selection:
        #   BLEU computed → use BLEU-weighted CE scaled by bleu_penalty_weight
        #   otherwise     → use standard CE (Bug 3 fix: apply weight; Bug 2 fix: no req_grad)
        if bleu_computed:
            loss = args.bleu_penalty_weight * bleu_penalty  # Bug 3 fix: weight now applied
        else:
            loss = ce_loss
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        # Free memory from forward pass immediately (already cleaned up above)
        
        # Optimizer step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Clear cache after optimizer step
            torch.cuda.empty_cache()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Update progress bar
        if step % args.log_every == 0:
            postfix = {
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            }
            if args.use_bleu_penalty and len(bleu_scores_epoch) > 0:
                postfix['bleu'] = f'{bleu_scores_epoch[-1]:.3f}'
            pbar.set_postfix(postfix)
    
    avg_loss = total_loss / len(dataloader)
    
    # Log BLEU penalty info if enabled
    if args.use_bleu_penalty and len(bleu_scores_epoch) > 0:
        avg_bleu = sum(bleu_scores_epoch) / len(bleu_scores_epoch)
        avg_bleu_penalty = total_bleu_penalty / len(bleu_scores_epoch)
        return avg_loss, {'avg_bleu': avg_bleu, 'avg_bleu_penalty': avg_bleu_penalty, 'running_baseline': running_bleu_baseline}
    
    return avg_loss, {'running_baseline': running_bleu_baseline if args.use_bleu_penalty else None}


@torch.no_grad()
def validate(model, dataloader, epoch, args, device, tokenizer=None, compute_metrics=False):
    """Validate the model with loss and optionally generation metrics.
    
    Args:
        compute_metrics: If True, generate text and compute NLP metrics (slow).
                        If False, only compute AR loss (fast, default during training).
    """
    model.eval()
    total_loss = 0
    num_samples = 0
    all_predictions = []
    all_references = []
    all_ids = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
    
    for batch in pbar:
        # Process each sample individually to avoid OOM during loss computation
        visual_features = batch['visual_features']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        batch_size = visual_features.shape[0]
        batch_loss_sum = 0
        
        # Process samples one at a time
        for i in range(batch_size):
            # Move single sample to device
            single_visual = visual_features[i:i+1].to(device)
            single_input = input_ids[i:i+1].to(device)
            single_mask = attention_mask[i:i+1].to(device)
            single_labels = labels[i:i+1].to(device)
            
            # Extract knowledge tokens if present
            single_knowledge_ids = None
            single_knowledge_mask = None
            if 'knowledge_input_ids' in batch:
                single_knowledge_ids = batch['knowledge_input_ids'][i:i+1].to(device)
                single_knowledge_mask = batch['knowledge_attention_mask'][i:i+1].to(device)
            
            # Forward pass for single sample
            outputs = model(
                visual_features=single_visual,
                input_ids=single_input,
                attention_mask=single_mask,
                labels=single_labels,
                knowledge_input_ids=single_knowledge_ids,
                knowledge_attention_mask=single_knowledge_mask
            )
            
            sample_loss = outputs['loss'].item()
            batch_loss_sum += sample_loss
            
            # Free memory immediately
            del outputs, single_input, single_mask, single_labels
            torch.cuda.empty_cache()
            
            # Generate prediction for metrics (only if requested)
            if compute_metrics and tokenizer is not None:
                # Temporarily disable gradient checkpointing for generation
                if hasattr(model.llm, 'gradient_checkpointing_disable'):
                    model.llm.gradient_checkpointing_disable()
                
                # Get knowledge text and question if available
                knowledge_text = None
                question_text = None
                if 'knowledge_texts' in batch and args.use_knowledge_guidance:
                    knowledge_text = batch['knowledge_texts'][i]
                if 'questions' in batch:
                    question_text = batch['questions'][i]
                
                # Use beam search if configured for better NLU metrics
                pred_text = model.generate(
                    visual_features=single_visual,
                    prompt=None,  # Let model auto-select based on knowledge_text
                    question=question_text,
                    knowledge_text=knowledge_text,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    temperature=0.0,  # Greedy/beam search for evaluation
                    top_p=1.0,
                    num_beams=getattr(args, 'eval_num_beams', 1),
                    length_penalty=getattr(args, 'eval_length_penalty', 1.0),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 3)
                )
                all_predictions.append(pred_text)

                # Extract ground truth report from labels
                label_ids = labels[i][labels[i] != -100]
                ref_text = extract_report_from_labels(tokenizer, label_ids)
                all_references.append(ref_text)

                # Collect sample ID (tcga_id)
                sample_id = batch['ids'][i] if 'ids' in batch else str(len(all_ids))
                all_ids.append(sample_id)
                
                # Re-enable gradient checkpointing if needed
                if hasattr(model.llm, 'gradient_checkpointing_enable') and args.gradient_checkpointing:
                    model.llm.gradient_checkpointing_enable()
            
            # Clear cache after each sample
            del single_visual
            torch.cuda.empty_cache()
        
        avg_batch_loss = batch_loss_sum / batch_size
        total_loss += batch_loss_sum
        num_samples += batch_size
        
        pbar.set_postfix({'val_loss': f'{avg_batch_loss:.4f}'})
    
    avg_loss = total_loss / num_samples

    metrics = {}
    if len(all_predictions) > 0:
        from metrics import compute_metrics as calc_metrics
        metrics = calc_metrics(all_predictions, all_references)

        # Save model.generate() outputs as formatted JSON for easy comparison
        import json
        results_dir = args.output_results_dir
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'stage2_val_epoch{epoch+1}.json')
        records = [
            {
                'tcga_id':    tcga_id,
                'prediction': pred,
                'reference':  ref,
            }
            for tcga_id, pred, ref in zip(all_ids, all_predictions, all_references)
        ]
        with open(results_path, 'w') as f:
            json.dump(records, f, indent=4)
        print(f"  [Results] Saved {len(records)} entries → {results_path}")

    return avg_loss, metrics


@torch.no_grad()
def evaluate_best_model(checkpoint_path, args, device):
    """Evaluate the best model with full NLP metrics after training completes."""
    print("\n" + "="*60)
    print("Evaluating Best Model with Full NLP Metrics")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL: Add image marker tokens (must match model.py)
    from config import IMAGE_START_TOKEN, IMAGE_END_TOKEN
    special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Create validation dataset with correct format
    # Use same logic as main training to determine paths and format
    if args.dataset_format == 'wsi-bench':
        val_path = args.test_json if args.test_json else args.data_json
        val_split = None
    else:
        val_path = args.data_json
        val_split = args.val_split
    
    val_dataset = WSI_Report_Dataset(
        json_path=val_path,
        dataset_format=args.dataset_format,
        split=val_split,
        feature_key=args.feature_key,
        use_knowledge_guidance=args.use_knowledge_guidance
    )
    
    # Create data collator with dataset format
    data_collator = VLMDataCollator(
        tokenizer=tokenizer,
        dataset_format=args.dataset_format,
        max_length=args.max_length,
        use_knowledge_guidance=args.use_knowledge_guidance
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    # Load model (on CPU first to save GPU memory)
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        slidechat_checkpoint=args.slidechat_checkpoint,
        use_qlora=True,
        freeze_llm=False,
        gradient_checkpointing=False,  # Disable for inference
        device='cpu'  # Initialize on CPU first
    )
    
    # Load checkpoint from disk to CPU
    print(f"\nLoading checkpoint to CPU: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load weights using model's method (handles projection + LoRA separately)
    model.projection.load_state_dict(checkpoint['projection_state_dict'])
    if 'lora_state_dict' in checkpoint:
        model.llm.load_state_dict(checkpoint['lora_state_dict'], strict=False)
    
    # Ensure projection is on CPU after loading (explicit device placement)
    model.projection = model.projection.to('cpu')
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']+1}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    
    # Store epoch before deleting checkpoint
    checkpoint_epoch = checkpoint['epoch']
    
    # Free checkpoint memory immediately
    del checkpoint
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Now move model to GPU (the new to() method will properly handle all components)
    print(f"Moving model to {device}...")
    model = model.to(device)
    
    # Evaluate with full metrics
    val_loss, metrics = validate(
        model, val_loader, checkpoint_epoch, args, device,
        tokenizer=tokenizer, compute_metrics=True
    )
    
    print("\nFinal Evaluation Results:")
    print(f"  Val Loss: {val_loss:.4f}")
    if metrics:
        print(f"  BLEU-1: {metrics.get('bleu1', 0):.2f}  "
              f"BLEU-2: {metrics.get('bleu2', 0):.2f}  "
              f"BLEU-3: {metrics.get('bleu3', 0):.2f}  "
              f"BLEU-4: {metrics.get('bleu4', 0):.2f}")
        print(f"  METEOR: {metrics.get('meteor', 0):.2f}  "
              f"ROUGE-L: {metrics.get('rouge_l', 0):.2f}")
    
    return val_loss, metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = f"{timestamp}_vt{args.num_visual_tokens}_ml{args.max_length}_e{args.epochs}_bs{args.batch_size}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Mirror experiment name in results dir so each run gets its own folder
    args.output_results_dir = os.path.join(args.output_results_dir, exp_name)
    os.makedirs(args.output_results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment directory: {exp_dir}")
    print(f"{'='*60}")
    
    # Save args
    with open(os.path.join(exp_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL: Add image marker tokens (must match model.py)
    from config import IMAGE_START_TOKEN, IMAGE_END_TOKEN
    special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_tokens} special tokens to tokenizer: {IMAGE_START_TOKEN}, {IMAGE_END_TOKEN}")
    
    # Initialize datasets
    print("Loading datasets...")
    
    # Determine paths and format based on arguments
    if args.dataset_format == 'wsi-bench':
        # WSI-Bench format: separate train and test JSONs (flat lists)
        train_path = args.train_json if args.train_json else args.data_json
        test_path = args.test_json if args.test_json else args.data_json
        train_split = None  # Not used for wsi-bench
        val_split = None
        print(f"Using WSI-Bench dataset format")
        print(f"  Train JSON: {train_path}")
        print(f"  Test JSON: {test_path}")
    else:
        # HistGen format: single JSON with nested splits
        train_path = args.data_json
        test_path = args.data_json
        train_split = args.train_split.split(',') if ',' in args.train_split else args.train_split
        val_split = args.val_split
        print(f"Using HistGen dataset format")
        print(f"  Data JSON: {args.data_json}")
        print(f"  Train split(s): {args.train_split}")
        print(f"  Val split: {args.val_split}")

    if args.use_knowledge_guidance:
        print("Using knowledge guidance")
    else:
        print("Not using knowledge guidance")
    
    train_dataset = WSI_Report_Dataset(
        json_path=train_path,
        dataset_format=args.dataset_format,
        split=train_split,
        feature_key=args.feature_key,
        use_knowledge_guidance=args.use_knowledge_guidance
    )
    
    val_dataset = WSI_Report_Dataset(
        json_path=test_path,
        dataset_format=args.dataset_format,
        split=val_split,
        feature_key=args.feature_key,
        use_knowledge_guidance=args.use_knowledge_guidance
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize collator with dataset format
    collator = VLMDataCollator(
        tokenizer=tokenizer,
        dataset_format=args.dataset_format,
        max_length=args.max_length,
        use_knowledge_guidance=args.use_knowledge_guidance
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    # Initialize model with QLoRA
    print("\n" + "="*60)
    print("Initializing VLM for Stage 2 Training (QLoRA)")
    print("="*60)
    
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        slidechat_checkpoint=args.slidechat_checkpoint,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        use_qlora=True,    # Stage 2: Use QLoRA
        freeze_llm=False,  # Stage 2: Train LLM with LoRA
        gradient_checkpointing=args.gradient_checkpointing,
        label_smoothing=args.label_smoothing,
        device=device
    )
    
    # ── Load initial weights for Stage 2 ────────────────────────────────
    if args.init_from is not None:
        # Warm-start from a previous Stage 2 checkpoint (projection + LoRA)
        print(f"\n[Warm-start] Loading Stage 2 weights from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location='cpu')
        if 'projection_state_dict' in ckpt:
            model.projection.load_state_dict(ckpt['projection_state_dict'])
            print("  ✓ Projection weights loaded")
        else:
            raise KeyError(f"'projection_state_dict' not found. Keys: {list(ckpt.keys())}")
        if 'lora_state_dict' in ckpt:
            model.llm.load_state_dict(ckpt['lora_state_dict'], strict=False)
            print("  ✓ LoRA weights loaded")
        else:
            print("  ! No 'lora_state_dict' found — LoRA starts from random init")
        print("  [Epoch resets to 0, optimizer is fresh]")
        del ckpt
        torch.cuda.empty_cache()
    elif args.stage1_checkpoint is not None:
        # Standard: load only projection from Stage 1 checkpoint
        print(f"\nLoading Stage 1 projection weights from: {args.stage1_checkpoint}")
        checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
        model.projection.load_state_dict(checkpoint['projection_state_dict'])
        print("✓ Projection weights loaded from Stage 1")
        del checkpoint
        torch.cuda.empty_cache()
    else:
        raise ValueError("Either --stage1_checkpoint or --init_from must be provided.")
    
    total_params, trainable_params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer with separate learning rates
    # LoRA parameters: lower LR
    # Projection parameters: higher LR
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and 'projection' in n],
            'lr': args.projection_lr,
            'name': 'projection'
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and 'projection' not in n],
            'lr': args.lr,
            'name': 'lora'
        }
    ]
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    print(f"\nOptimizer parameter groups:")
    for group in optimizer.param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params:,} params, lr={group['lr']:.2e}")
    
    # Learning rate scheduler with optional warmup
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    # Compute warmup steps
    if args.warmup_steps > 0:
        num_warmup_steps = args.warmup_steps
    elif args.warmup_ratio > 0:
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    else:
        num_warmup_steps = 0
    
    if num_warmup_steps > 0:
        # Warmup + Cosine decay scheduler
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay after warmup
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"Using warmup + cosine decay scheduler: {num_warmup_steps} warmup steps out of {num_training_steps} total")
    else:
        # Standard cosine annealing without warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-7)
        print(f"Using cosine annealing scheduler: {num_training_steps} total steps")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    
    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_epoch, optimizer_state = model.load_checkpoint(args.resume_from, load_optimizer=True)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Stage 2 Training")
    if args.early_stopping:
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    print("="*60)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Only initialize NLU metric lists if computing during training
    if args.compute_metrics_during_training:
        history.update({
            'bleu1': [],
            'bleu2': [],
            'bleu3': [],
            'bleu4': [],
            'meteor': [],
            'rouge_l': []
        })
    
    # Early stopping variables
    epochs_without_improvement = 0
    early_stopped = False
    
    # Track last completed epoch (initialize to start_epoch - 1 in case no epochs run)
    epoch = start_epoch - 1 if start_epoch > 0 else 0
    
    # Running BLEU baseline for dynamic hard example mining
    running_bleu_baseline = None
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_bleu_info = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, args, device, 
            tokenizer=tokenizer if args.use_bleu_penalty else None,
            running_bleu_baseline=running_bleu_baseline
        )
        history['train_loss'].append(train_loss)
        
        # Update running baseline for next epoch
        if train_bleu_info and 'running_baseline' in train_bleu_info:
            running_bleu_baseline = train_bleu_info['running_baseline']
        
        # Store BLEU penalty info if available
        if train_bleu_info:
            if 'train_bleu' not in history:
                history['train_bleu'] = []
            history['train_bleu'].append(train_bleu_info.get('avg_bleu', 0.0))
        
        # Aggressive memory cleanup between training and validation
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Validate (fast mode: only AR loss unless user requested full metrics)
        compute_metrics_now = args.compute_metrics_during_training
        val_loss, val_metrics = validate(
            model, val_loader, epoch, args, device, tokenizer,
            compute_metrics=compute_metrics_now
        )
        history['val_loss'].append(val_loss)
        
        # Store metrics only if computed during training
        if args.compute_metrics_during_training:
            for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge_l']:
                history[key].append(val_metrics.get(key, 0.0))
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        if train_bleu_info:
            print(f"  Train BLEU ({args.bleu_metric}): {train_bleu_info.get('avg_bleu', 0):.3f}  "
                  f"BLEU Penalty: {train_bleu_info.get('avg_bleu_penalty', 0):.3f}")
        print(f"  Val Loss: {val_loss:.4f}")
        if val_metrics:
            print(f"  BLEU-1: {val_metrics.get('bleu1', 0):.2f}  "
                  f"BLEU-2: {val_metrics.get('bleu2', 0):.2f}  "
                  f"BLEU-3: {val_metrics.get('bleu3', 0):.2f}  "
                  f"BLEU-4: {val_metrics.get('bleu4', 0):.2f}")
            print(f"  METEOR: {val_metrics.get('meteor', 0):.2f}  "
                  f"ROUGE-L: {val_metrics.get('rouge_l', 0):.2f}")
        elif not args.compute_metrics_during_training:
            print("  (NLP metrics will be computed for best model at end of training)")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pt')
            model.save_checkpoint(save_path, epoch, optimizer.state_dict(), val_loss=val_loss)
        
        # Save best model
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0  # Reset counter
            best_path = os.path.join(exp_dir, 'best_model.pt')
            model.save_checkpoint(best_path, epoch, optimizer.state_dict(), val_loss=val_loss)
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if args.early_stopping:
                print(f"  No improvement for {epochs_without_improvement}/{args.patience} epochs")
        
        # Check early stopping
        if args.early_stopping and epochs_without_improvement >= args.patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"No improvement in validation loss for {args.patience} consecutive epochs")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            print(f"{'='*60}")
            early_stopped = True
            break
    
    # Save final model
    final_path = os.path.join(exp_dir, 'final_model.pt')
    model.save_checkpoint(final_path, epoch, optimizer.state_dict(), val_loss=best_val_loss)
    
    # Save final training history
    history['early_stopped'] = early_stopped
    history['epochs_completed'] = epoch + 1
    with open(os.path.join(exp_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    if early_stopped:
        print(f"Stopped early at epoch {epoch + 1}/{args.epochs}")
    print(f"Best model saved at epoch {best_epoch+1} with val_loss: {best_val_loss:.4f}")
    print(f"All checkpoints saved to: {exp_dir}")
    
    # Evaluate best model with full metrics if not computed during training
    should_evaluate = not args.compute_metrics_during_training and not args.skip_final_metrics
    if should_evaluate:
        # IMPORTANT: Free GPU memory before loading evaluation model
        print("\nFreeing GPU memory before final evaluation...")
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Give CUDA time to fully free memory
        import time
        time.sleep(2)
        print("GPU memory freed. Loading evaluation model...")
        
        best_checkpoint = os.path.join(exp_dir, 'best_model.pt')
        final_val_loss, final_metrics = evaluate_best_model(best_checkpoint, args, device)
        
        # Update history with final metrics
        history['final_evaluation'] = {
            'val_loss': final_val_loss,
            'metrics': final_metrics,
            'best_epoch': best_epoch + 1
        }
        
        # Re-save history with final metrics
        with open(os.path.join(exp_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    print("="*60)


if __name__ == "__main__":
    main()
