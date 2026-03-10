"""
Stage 1 Training: Projection Layer Only

Trains only the projection layer while keeping the LLM frozen.
This stage aligns the visual features with the LLM's embedding space.

Usage:
    python pipeline/train_stage1.py --epochs 10 --batch_size 8 --lr 1e-4
"""

import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

# Local imports
from model import VisionLanguageModel
from dataset import WSI_Report_Dataset
from data_collator import VLMDataCollator
from config import (GENERATION_PROMPT, TRAINING_PROMPT_TEMPLATE,
                    DEFAULT_LLM_PATH, DEFAULT_SLIDECHAT_CHECKPOINT,
                    DEFAULT_DATA_JSON, DEFAULT_STAGE1_OUTPUT,
                    DEFAULT_VISUAL_DIM, DEFAULT_NUM_VISUAL_TOKENS,
                    DEFAULT_MAX_LENGTH, DEFAULT_MAX_NEW_TOKENS,
                    extract_report_from_labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Training: Projection Layer")
    
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
    parser.add_argument("--visual_dim", type=int, default=DEFAULT_VISUAL_DIM,
                       help="Dimension of slide features (TITAN encoder)")
    parser.add_argument("--num_visual_tokens", type=int, default=DEFAULT_NUM_VISUAL_TOKENS,
                       help="Number of visual tokens to generate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size (reduced for memory)")
    parser.add_argument("--val_batch_size", type=int, default=2,
                       help="Validation batch size (smaller to avoid OOM)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--use_bleu_penalty", action="store_true",
                       help="Enable BLEU-based penalty loss during training")
    parser.add_argument("--bleu_penalty_weight", type=float, default=0.1,
                       help="Weight for BLEU penalty loss (total_loss = ce_loss + bleu_penalty_weight * bleu_penalty)")
    parser.add_argument("--bleu_metric", type=str, default="bleu4",
                       choices=["bleu1", "bleu2", "bleu3", "bleu4"],
                       help="Which BLEU metric to use for penalty")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio for scheduler")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")
    
    # Knowledge guidance arguments
    parser.add_argument("--use_knowledge_guidance", action="store_true",
                       help="Enable knowledge graph guidance for report generation (recommended if Stage 2 will use it)")
    parser.add_argument("--max_knowledge_tokens", type=int, default=64,
                       help="Maximum length for knowledge text tokens")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, 
                       default=DEFAULT_STAGE1_OUTPUT,
                       help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=2,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint (restores epoch + optimizer)")
    parser.add_argument("--init_from", type=str, default=None,
                       help="Warm-start: load projection weights from a previous Stage 1 checkpoint "
                            "(epoch resets to 0, optimizer is fresh). Useful for rapid prototyping.")
    
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping based on validation loss")
    parser.add_argument("--patience", type=int, default=5,
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
    
    return parser.parse_args()



def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(model, dataloader, optimizer, scheduler, epoch, args, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        visual_features = batch['visual_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        # Free memory from forward pass immediately
        del outputs
        
        # Optimizer step at the end of the gradient accumulation
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
            pbar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


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
            
            # Forward pass for single sample
            outputs = model(
                visual_features=single_visual,
                input_ids=single_input,
                attention_mask=single_mask,
                labels=single_labels
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
                
                pred_text = model.generate(
                    visual_features=single_visual,
                    prompt=None,
                    question=question_text,
                    knowledge_text=knowledge_text,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    temperature=0.0,
                    top_p=1.0
                )
                all_predictions.append(pred_text)
                
                # Extract ground truth report from labels
                label_ids = labels[i][labels[i] != -100]
                ref_text = extract_report_from_labels(tokenizer, label_ids)
                all_references.append(ref_text)
                
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
    
    # Create data collator
    data_collator = VLMDataCollator(
        tokenizer=tokenizer,
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
        use_qlora=False,  # Stage 1: no QLoRA
        freeze_llm=True,  # Stage 1: LLM is frozen
        gradient_checkpointing=False,  # Disable for inference
        device='cpu'  # Initialize on CPU first
    )
    
    # Load checkpoint from disk to CPU
    print(f"\nLoading checkpoint to CPU: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load projection weights
    model.projection.load_state_dict(checkpoint['projection_state_dict'])
    
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
    
    # Now move model to GPU (the to() method will properly handle all components)
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
    
    # CRITICAL: Add image marker tokens as special tokens (must match model.py)
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
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing VLM for Stage 1 Training")
    print("="*60)
    
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        slidechat_checkpoint=args.slidechat_checkpoint,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        use_qlora=False,  # Stage 1: No QLoRA
        freeze_llm=True,   # Stage 1: Freeze LLM
        gradient_checkpointing=args.gradient_checkpointing,  # Memory optimization
        device=device
    )
    
    total_params, trainable_params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer (only projection parameters)
    trainable_params_list = [p for p in model.projection.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params_list,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    
    # ── Warm-start: load projection weights only (epoch + optimizer reset) ──────
    if args.init_from is not None:
        print(f"\n[Warm-start] Loading projection weights from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location='cpu')
        if 'projection_state_dict' in ckpt:
            model.projection.load_state_dict(ckpt['projection_state_dict'])
            print("  ✓ Projection weights loaded (epoch resets to 0, optimizer is fresh)")
        else:
            raise KeyError(f"'projection_state_dict' not found in {args.init_from}. "
                           f"Keys: {list(ckpt.keys())}")
        del ckpt
        torch.cuda.empty_cache()

    # ── Resume from checkpoint (restores epoch + optimizer) ──────────────────
    # Note: resume_from takes priority over init_from for epoch/optimizer state
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1

    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_epoch, optimizer_state = model.load_checkpoint(args.resume_from, load_optimizer=True)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Stage 1 Training")
    if args.early_stopping:
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    print("="*60)
    
    # Initialize history (only track NLU metrics if computed during training)
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
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, args, device)
        history['train_loss'].append(train_loss)
        
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
        print("\nFreeing GPU memory before final NLU evaluation...")
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
