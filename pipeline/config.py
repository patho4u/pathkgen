"""
Configuration file for VLM training and inference.

Single source of truth for prompts and other shared settings.
Change the prompt here once, and it updates everywhere.
"""
import os
import sys
from dataclasses import dataclass

# Import base directories from root-level config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import VLM_ROOT, DATA_DIR, DATASET_DIR, CHECKPOINTS_DIR, RESULTS_DIR

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Image marker tokens (will be added as special tokens to tokenizer)
IMAGE_START_TOKEN = "<image_start>"
IMAGE_END_TOKEN = "<image_end>"

# ============================================================================
# PROMPT TEMPLATES - HISTGEN FORMAT (Legacy)
# ============================================================================

# Training prompt without knowledge guidance (HistGen format)
TRAINING_PROMPT_NO_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Please analyze this whole slide image and provide a detailed diagnostic report describing the key histopathological features and diagnosis.<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""

# Training prompt with knowledge guidance (HistGen format)
TRAINING_PROMPT_WITH_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Generic knowledge regarding histology type (96.5% accurate). The site is exact.
{knowledge}
Use to help, but verify visually.

Please analyze this whole slide image and provide a detailed diagnostic report describing the key histopathological features and diagnosis.<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""

# ============================================================================
# PROMPT TEMPLATES - WSI-BENCH FORMAT (Recommended)
# ============================================================================

# WSI-Bench format - uses dataset's question field
TRAINING_PROMPT_WSI_BENCH = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

# WSI-Bench with knowledge guidance
TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Generic knowledge regarding histology type (97.52% accurate). The site is exact.
{knowledge}
Use to help, but verify visually.

{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

# ============================================================================
# DATASET PATHS
# ============================================================================

# WSI-Bench dataset paths (pre-split into train/test)
WSI_BENCH_TRAIN = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", "wsi_bench_Report_train.json")
WSI_BENCH_TEST = os.path.join(DATASET_DIR, "wsi_bench_Report_with_features", "wsi_bench_Report_test.json")

# WSI-Bench dataset paths WITH KNOWLEDGE
WSI_BENCH_TRAIN_KNOWLEDGE = os.path.join(DATASET_DIR, "WSI-Bench-train_Report_knowledge.json")
WSI_BENCH_TEST_KNOWLEDGE = os.path.join(DATASET_DIR, "WSI-Bench-test_Report_knowledge.json")

# HistGen dataset path (contains splits internally)
HISTGEN_DATASET = os.path.join(DATASET_DIR, "histgen_dataset.json")

# ============================================================================
# INFERENCE/GENERATION PROMPTS
# ============================================================================

# Generation/Inference prompts (no {report} placeholder)

# Without knowledge guidance
GENERATION_PROMPT_NO_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
{question}<|im_end|>
<|im_start|>assistant
"""

# With knowledge guidance (includes {knowledge} placeholder)
GENERATION_PROMPT_WITH_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Generic knowledge regarding histology type (97.52% accurate). The site is exact.
{knowledge}
Use to help, but verify visually.

{question}<|im_end|>
<|im_start|>assistant
"""

# Legacy template for backward compatibility
TRAINING_PROMPT_TEMPLATE = TRAINING_PROMPT_NO_KNOWLEDGE
GENERATION_PROMPT = GENERATION_PROMPT_NO_KNOWLEDGE

# ============================================================================
# HELPER FUNCTIONS FOR EVALUATION
# ============================================================================

def extract_report_from_labels(tokenizer, label_ids):
    """
    Extract the ground truth report text from tokenized labels.

    After label masking in VLMDataCollator, label_ids passed here contains
    ONLY the answer (T-answer) tokens because the caller filters:
        label_ids = labels[i][labels[i] != -100]
    Everything before <|im_start|>assistant is masked to -100, so the
    primary decode path is simply a clean skip_special_tokens decode.

    Args:
        tokenizer: HuggingFace tokenizer
        label_ids: 1-D Tensor of non-masked label token IDs (-100 already removed)

    Returns:
        str: Clean ground truth report text (T-answer only)
    """
    # Safety net: decode keeping special tokens so we can detect ChatML markers.
    # This handles the edge case where label masking wasn't applied and the full
    # prompt survives. We look for <|im_start|>assistant and extract only what
    # follows it (up to the closing <|im_end|>).
    full_text = tokenizer.decode(label_ids, skip_special_tokens=False)

    assistant_marker = "<|im_start|>assistant"
    end_marker = "<|im_end|>"

    if assistant_marker in full_text:
        # Full prompt is present — extract the assistant portion only
        after_assistant = full_text.split(assistant_marker, 1)[-1]
        if end_marker in after_assistant:
            report = after_assistant.split(end_marker, 1)[0]
        else:
            report = after_assistant
        return report.strip()

    # Primary path (normal case): label masking already removed the prompt.
    # skip_special_tokens=True strips any remaining <|im_end|> cleanly.
    return tokenizer.decode(label_ids, skip_special_tokens=True).strip()



# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Visual embedding settings
DEFAULT_NUM_VISUAL_TOKENS = 128  # Default number of visual tokens
DEFAULT_VISUAL_DIM = 768  # TITAN encoder feature dimension per token
DEFAULT_LLM_HIDDEN_DIM = 3584  # Qwen 2.5 7B hidden dimension

# NOTE: Number of visual tokens is now DYNAMIC based on encoder output
# Typical values: 128 (TITAN), 256, 576 (Gigapath)
# Determined automatically from feature file shape [num_tokens, 768]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Sequence length settings
DEFAULT_MAX_LENGTH = 768  # Maximum input sequence length (prompt + report) - increased for knowledge context
DEFAULT_MAX_NEW_TOKENS = 512  # Maximum tokens to generate during validation (increased for safety)
INFERENCE_MAX_NEW_TOKENS = 512  # Maximum tokens for standalone inference

# Generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2

# Knowledge guidance settings
DEFAULT_USE_KNOWLEDGE_GUIDANCE = False
# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Default paths (can be overridden by command-line arguments)
DEFAULT_LLM_PATH = os.path.join(CHECKPOINTS_DIR, "Qwen-UMLS-7B-Instruct")
DEFAULT_SLIDECHAT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "SlideChat_Weight", "stage2_pth", "mp_rank_00_model_states.pt")
DEFAULT_DATA_JSON = os.path.join(DATASET_DIR, "tcga_report_histgen", "primary_site_splits", "breast_dataset_split_0.7_0.1_0.2.json")
DEFAULT_STAGE1_OUTPUT = os.path.join(CHECKPOINTS_DIR, "stage1")
DEFAULT_STAGE2_OUTPUT = os.path.join(CHECKPOINTS_DIR, "stage2")



@dataclass
class Stage1Config:
    """Configuration for Stage 1 contrastive alignment training."""
    
    # ---- Data ----
    slide_features_dir: str = os.path.join(DATA_DIR, "slide_features")  # Has train/ and test/ subdirs
    text_embeddings_dir: str = os.path.join(DATA_DIR, "text_embeddings")  # Has train/ and test/ subdirs
    
    
    # ---- TITAN Model ----
    titan_model_name: str = "MahmoodLab/TITAN"
    titan_dim: int = 768  # TITAN token dimension (128 tokens × 768-dim each)
    
    # ---- Projection Heads ----
    hidden_dim: int = 1536        # Hidden dimension of MLP projectors
    alignment_dim: int = 1536     # Output dimension of shared alignment space
    
    # ---- Contrastive Loss (WSI-LLaVA paper) ----
    temperature: float = 0.02   # InfoNCE temperature (paper: τ=0.02)
    
    # ---- Training (WSI-LLaVA paper) ----
    lr: float = 1e-3            # Learning rate (paper: 0.001)
    weight_decay: float = 0.01
    batch_size: int = 64        # Batch size (paper: 64)
    epochs: int = 50            # Total epochs (paper: 50)
    warmup_ratio: float = 0.03  # Warmup fraction of total steps
    
    # ---- Text Encoding ----
    context_length: int = 128   # TITAN text context length
    
    # ---- Checkpointing ----
    checkpoint_dir: str = os.path.join(CHECKPOINTS_DIR, "stage1")
    save_every_n_epochs: int = 5
    
    # ---- System ----
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    log_interval: int = 10      # Log every N batches
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
