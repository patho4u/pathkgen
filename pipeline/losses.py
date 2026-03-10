"""
losses.py — Custom loss functions for the VLM training pipeline.

Contains pure tensor operations with no model dependencies, importable
by both model.py and training scripts.

Functions:
    label_smoothing_loss  — Cross-entropy with label smoothing
    compute_rdrop_loss    — R-Drop consistency regularization (Wu et al., 2021)
"""

import math
import torch
import torch.nn.functional as F


def label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident by
    distributing some probability mass to non-target classes.

    The caller is responsible for shifting logits/labels for causal LM:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

    Args:
        logits:       [batch, seq_len, vocab_size] — already shifted if needed
        labels:       [batch, seq_len] — already shifted; -100 positions are ignored
        smoothing:    Label smoothing factor (0.0 = standard cross-entropy)
        ignore_index: Token ID to ignore in the loss (default -100)

    Returns:
        Scalar loss tensor (differentiable)
    """
    if smoothing <= 0.0:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
        )

    vocab_size = logits.size(-1)

    # Flatten
    logits_flat = logits.view(-1, vocab_size)   # [N, vocab]
    labels_flat = labels.view(-1)               # [N]

    # Mask padding/ignored positions
    valid_mask = labels_flat != ignore_index
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits = logits_flat[valid_mask]      # [V, vocab]
    valid_labels = labels_flat[valid_mask]      # [V]

    # Log-probabilities (numerically stable)
    log_probs = F.log_softmax(valid_logits, dim=-1)     # [V, vocab]

    # Smoothed target distribution:
    #   p_true  = 1 - smoothing
    #   p_other = smoothing / (vocab_size - 1)
    smooth_target = torch.full_like(log_probs, smoothing / (vocab_size - 1))
    smooth_target.scatter_(1, valid_labels.unsqueeze(1), 1.0 - smoothing)

    # KL-divergence / cross-entropy with soft targets
    loss = -(smooth_target * log_probs).sum() / valid_mask.sum()
    return loss
