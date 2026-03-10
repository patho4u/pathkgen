import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Tuple
import os
import warnings
from config import (DEFAULT_VISUAL_DIM, DEFAULT_NUM_VISUAL_TOKENS,
                    DEFAULT_LLM_HIDDEN_DIM, DEFAULT_LLM_PATH,
                    DEFAULT_SLIDECHAT_CHECKPOINT, IMAGE_START_TOKEN, IMAGE_END_TOKEN)
from losses import label_smoothing_loss




class ProjectionLayer(nn.Module):
    """
    Projects spatial WSI slide features into LLM embedding space.
    
    WSI-LLaVA Style - Token-wise projection:
        - Input: [batch, num_tokens, 768] spatial slide embeddings
        - Project each token independently: [768] → [3584]
        - Output: [batch, num_tokens, 3584]
    
    Note: nn.Linear automatically handles 3D inputs by operating on last dimension.
    No flattening/reshaping needed - preserves spatial structure!
    """
    def __init__(
        self, 
        visual_dim: int = DEFAULT_VISUAL_DIM,  # 768 (TITAN feature dim)
        llm_hidden_dim: int = DEFAULT_LLM_HIDDEN_DIM  # 3584 (Qwen 2.5 7B)
    ):
        super().__init__()
        
        # Token-wise MLP projection (WSI-LLaVA mlp2x_gelu style)
        # hidden_dim = llm_hidden_dim (3584) to match WSI-LLaVA exactly
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, llm_hidden_dim),      # Per-token: [768] → [3584]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)   # Per-token: [3584] → [3584]
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, num_tokens, 768] spatial features
                            (e.g., [batch, 128, 768] from TITAN)
            
        Returns:
            visual_tokens: [batch, num_tokens, 3584] projected tokens
        """
        # Validate input shape
        if visual_features.dim() != 3:
            raise ValueError(
                f"Expected 3D input [batch, num_tokens, 768], "
                f"got shape {visual_features.shape}"
            )
        
        # Token-wise projection (nn.Linear handles 3D automatically!)
        # Input:  [batch, num_tokens, 768]
        # Output: [batch, num_tokens, 3584]
        visual_tokens = self.projector(visual_features)
        
        return visual_tokens

class VisionLanguageModel(nn.Module):
    """
    Complete VLM for histopathology report generation.
    
    Components:
        - Projection layer: WSI features → visual tokens
        - Language model: Qwen 2.5 7B with SlideChat weights
        - QLoRA support for efficient fine-tuning
    """
    def __init__(
        self,
        llm_path: str = DEFAULT_LLM_PATH,
        slidechat_checkpoint: str = DEFAULT_SLIDECHAT_CHECKPOINT,
        visual_dim: int = DEFAULT_VISUAL_DIM,
        llm_hidden_dim: int = DEFAULT_LLM_HIDDEN_DIM,
        num_visual_tokens: int = DEFAULT_NUM_VISUAL_TOKENS,
        use_qlora: bool = False,
        freeze_llm: bool = True,
        gradient_checkpointing: bool = False,
        label_smoothing: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.use_qlora = use_qlora
        self.freeze_llm = freeze_llm
        self.label_smoothing = label_smoothing
        
        # 1. Initialize tokenizer (use same path as LLM for consistency)
        print(f"Loading tokenizer from {llm_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add image marker tokens as special tokens
        special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_tokens} special tokens: {IMAGE_START_TOKEN}, {IMAGE_END_TOKEN}")
        
        # Store token IDs for later use
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_START_TOKEN)
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_END_TOKEN)
            
        # 2. Initialize projection layer
        llm_hidden_dim = llm_hidden_dim  # Qwen 2.5 7B
        self.projection = ProjectionLayer(
            visual_dim=visual_dim,
            llm_hidden_dim=llm_hidden_dim
        )
        
        # 3. Load language model
        if use_qlora:
            print("Loading LLM with 4-bit quantization for QLoRA...")
            self.llm = self._load_llm_with_qlora(llm_path)
        else:
            print(f"Loading LLM from {llm_path}")
            # Determine device_map based on target device
            # If initializing on CPU (for checkpoint loading), load on CPU
            # If initializing on CUDA, use device_map='auto' for multi-GPU support
            llm_device_map = "cpu" if device == "cpu" else "auto"
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                device_map=llm_device_map,
                trust_remote_code=True
            )
        
        # 4. Load SlideChat weights if provided
        if slidechat_checkpoint is not None:
            print(f"Loading SlideChat pathology weights from {slidechat_checkpoint}")
            self._load_slidechat_weights(slidechat_checkpoint)
        
        # 5. Enable gradient checkpointing if requested (saves ~40% activation memory)
        if gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency...")
            self.llm.gradient_checkpointing_enable()
        
        # 6. Freeze LLM if specified
        if freeze_llm and not use_qlora:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
                
        # 7. Move projection to device and match LLM dtype
        self.projection = self.projection.to(device)
        # Convert projection to bfloat16 to match LLM
        self.projection = self.projection.to(torch.bfloat16)
        
        # Store the LLM's dtype for consistency
        self.llm_dtype = torch.bfloat16
        
        print(f"VLM initialized successfully!")
        print(f"  - Projection: Token-wise {visual_dim}d → {llm_hidden_dim}d (WSI-LLaVA style)")
        print(f"  - Visual tokens: Dynamic (determined by encoder output)")
        print(f"  - LLM frozen: {freeze_llm}")
        print(f"  - Using QLoRA: {use_qlora}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        
    def _load_llm_with_qlora(self, llm_path: str) -> nn.Module:
        """Load LLM with 4-bit quantization and LoRA adapters."""
        # Clear GPU memory before loading
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
        )
        
        # Clear cache again before prepare_model_for_kbit_training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        # LoRA configuration (optimized for fast convergence without knowledge)
        lora_config = LoraConfig(
            r=128,  # Rank (higher = more capacity)
            lora_alpha=256,  # Scaling factor (2x rank = aggressive updates for faster learning)
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.02,  # Reduced from 0.05 (less regularization for faster learning)
            bias="all",  # Enable bias adaptation (was "none", adds capacity)
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _load_slidechat_weights(self, checkpoint_path: str):
        """Load SlideChat pathology-specific weights into the LLM using memory-mapped file."""
        try:
            print(f"Loading SlideChat weights (this may take a moment)...")
            
            # Use mmap_mode='r' to memory-map the file instead of loading it all into RAM
            # This reads weights directly from disk as needed, avoiding OOM
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print("  Memory-mapping checkpoint file (avoids loading entire file into RAM)...")
            # CRITICAL: Load to CPU first to avoid OOM (Qwen already on GPU)
            state_dict = torch.load(
                checkpoint_path, 
                map_location='cpu',  # Load to CPU first, then selectively move to GPU
                weights_only=False,
                mmap=True  # Memory-map the file
            )
            
            # Clean state dict and immediately move to device to minimize CPU RAM usage
            new_state_dict = {}
            print(f"  Filtering {len(state_dict)} keys for LLM parameters...")
            
            for key, value in state_dict.items():
                if key.startswith("model."):
                    clean_key = key.replace("model.", "")
                    # Skip vision-only parameters
                    if "mm_projector" not in clean_key and "slide_encoder" not in clean_key:
                        # Value is already on target device from map_location
                        new_state_dict[clean_key] = value
                        
                # Free memory every 100 keys
                if len(new_state_dict) % 100 == 0:
                    gc.collect()
            
            # Delete original state_dict to free RAM immediately
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"  Loading {len(new_state_dict)} parameters into LLM...")
            # Load into LLM (strict=False to allow missing vision keys)
            if self.use_qlora:
                # For PEFT models, load into base_model
                msg = self.llm.base_model.model.load_state_dict(new_state_dict, strict=False)
            else:
                msg = self.llm.load_state_dict(new_state_dict, strict=False)
            
            # Clean up
            del new_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"✓ SlideChat weights loaded successfully")
            print(f"  - Missing keys: {len(msg.missing_keys)} (expected for vision components)")
            print(f"  - Unexpected keys: {len(msg.unexpected_keys)}")
            
        except Exception as e:
            warnings.warn(f"Failed to load SlideChat weights: {e}")
            print("Continuing with base Qwen 2.5 weights...")
            import traceback
            traceback.print_exc()
    
    def get_trainable_params(self) -> Dict[str, torch.nn.Parameter]:
        """Get dictionary of trainable parameters for optimizer."""
        trainable = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable[name] = param
        return trainable
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def forward(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        knowledge_input_ids: Optional[torch.Tensor] = None,
        knowledge_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with image marker token injection.
        
        Args:
            visual_features: [batch, 768] slide embeddings
            input_ids: [batch, seq_len] tokenized text (includes <image_start> and <image_end>)
            attention_mask: [batch, seq_len] attention mask
            labels: [batch, seq_len] labels for language modeling loss
            
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        batch_size = visual_features.shape[0]
        
        # 1. Project visual features to tokens (ensure dtype matches LLM)
        # Input: [batch, N, 768] spatial features (e.g., N=128 from TITAN)
        # Output: [batch, N, 3584] visual tokens
        visual_features = visual_features.to(self.llm_dtype)
        visual_tokens = self.projection(visual_features)  # Token-wise projection
        
        # Get actual number of visual tokens from the input (dynamic)
        num_visual_tokens = visual_tokens.shape[1]  # e.g., 128 for TITAN
        
        # 2. Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden_dim]
        
        # 3. Find image marker positions (assume same across batch since same template)
        sample_input_ids = input_ids[0]
        marker_start_positions = (sample_input_ids == self.image_start_token_id).nonzero(as_tuple=True)[0]
        marker_end_positions = (sample_input_ids == self.image_end_token_id).nonzero(as_tuple=True)[0]
        
        if len(marker_start_positions) == 0 or len(marker_end_positions) == 0:
            # Detailed debugging info
            import numpy as np
            print(f"\n{'='*60}")
            print("ERROR: Image markers not found in input_ids")
            print(f"{'='*60}")
            print(f"Expected start token ID: {self.image_start_token_id}")
            print(f"Expected end token ID: {self.image_end_token_id}")
            print(f"Start marker found: {len(marker_start_positions) > 0}")
            print(f"End marker found: {len(marker_end_positions) > 0}")
            print(f"Unique token IDs in sequence: {torch.unique(sample_input_ids).tolist()}")
            print(f"Sequence length: {len(sample_input_ids)}")
            print(f"First 20 tokens: {sample_input_ids[:20].tolist()}")
            print(f"Last 20 tokens: {sample_input_ids[-20:].tolist()}")
            print(f"\nPossible causes:")
            print(f"1. Sequence was truncated and lost the </image_end> marker")
            print(f"2. Tokenizer was not initialized with image marker special tokens")
            print(f"3. Knowledge text is too long, pushing markers beyond max_length")
            print(f"{'='*60}\n")
            raise ValueError("Image markers <image_start> and <image_end> not found in input_ids")
        
        visual_start_idx = marker_start_positions[0].item()
        visual_end_idx = marker_end_positions[0].item()
        
        # Split embeddings: [before_markers] + [visual_tokens] + [after_markers]
        before_embeds = inputs_embeds[:, :visual_start_idx+1, :]  # Everything before with <image_start>
        after_embeds = inputs_embeds[:, visual_end_idx:, :]   # Everything after with <image_end>
        
        # Concatenate: before + visual + after
        combined_embeds = torch.cat([before_embeds, visual_tokens, after_embeds], dim=1)
        
        # 6. VECTORIZED: Handle attention mask
        if attention_mask is not None:
            before_attn = attention_mask[:, :visual_start_idx]
            after_attn = attention_mask[:, visual_end_idx+1:]
            # Attention for: start_marker (1) + visual_tokens (N) + end_marker (1)
            marker_visual_attn = torch.ones(
                batch_size, 
                1 + num_visual_tokens + 1, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            combined_attention_mask = torch.cat([before_attn, marker_visual_attn, after_attn], dim=1)
        else:
            combined_attention_mask = None
        
        # 7. VECTORIZED: Handle labels
        if labels is not None:
            before_labels = labels[:, :visual_start_idx]
            after_labels = labels[:, visual_end_idx+1:]
            # Labels for: start_marker (1) + visual_tokens (N) + end_marker (1) = all -100 (ignore)
            marker_visual_labels = torch.full(
                (batch_size, 1 + num_visual_tokens + 1), 
                -100, 
                dtype=labels.dtype, 
                device=labels.device
            )
            combined_labels = torch.cat([before_labels, marker_visual_labels, after_labels], dim=1)
        else:
            combined_labels = None
        
        # 8. Forward through LLM
        # If using label smoothing, don't pass labels to LLM (compute custom loss)
        if self.label_smoothing > 0 and combined_labels is not None:
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=None,  # Don't compute loss in LLM
                return_dict=True
            )
            
            # Compute label smoothing loss manually
            # Shift logits and labels for causal LM (predict next token)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = combined_labels[..., 1:].contiguous()
            
            loss = label_smoothing_loss(
                shift_logits,
                shift_labels,
                smoothing=self.label_smoothing,
                ignore_index=-100
            )
        else:
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=combined_labels,
                return_dict=True
            )
            loss = outputs.loss
        
        return {
            "loss": loss,
            "logits": outputs.logits
        }
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        prompt: str = None,
        question: str = None,
        knowledge_text: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        diversity_penalty: float = 0.0,
        num_beam_groups: int = 1
    ) -> str:
        """
        Generate report for a single WSI slide with image marker tokens.
        
        Supports both sampling and beam search with various enhancements for
        improved NLU metrics.
        
        Args:
            visual_features: [1, num_tokens, 768] or [num_tokens, 768] slide embedding
            prompt: Text prompt (if None, uses default from config)
            question: Per-sample question string (WSI-Bench). If None, falls back to
                      a generic instruction in the generation prompt.
            knowledge_text: Optional knowledge context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored if num_beams > 1)
            top_p: Nucleus sampling parameter (ignored if num_beams > 1)
            repetition_penalty: Penalty for repetition
            num_beams: Number of beams for beam search (1 = greedy/sampling)
            length_penalty: Exponential penalty for length (>1 = longer, <1 = shorter)
            no_repeat_ngram_size: Size of n-grams that cannot repeat
            early_stopping: Whether to stop when all beams finish
            diversity_penalty: Penalty for diverse beam search (requires num_beam_groups > 1)
            num_beam_groups: Number of beam groups for diverse decoding
            
        Returns:
            Generated report text
        """
        from config import GENERATION_PROMPT_NO_KNOWLEDGE, GENERATION_PROMPT_WITH_KNOWLEDGE
        
        self.eval()

        # Suppress use_cache + gradient_checkpointing incompatibility warning.
        # gradient_checkpointing is only active during training; during .eval()
        # generation we can safely re-enable the KV cache.
        llm_config_use_cache = getattr(self.llm.config, 'use_cache', True)
        self.llm.config.use_cache = True
        # Ensure batch dimension
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        
        # Select appropriate prompt if not provided
        if prompt is None:
            # Fallback question if none provided
            q = question if question else "Analyze the whole slide histopathology image and provide a concise pathology diagnosis."
            if knowledge_text is not None:
                prompt = GENERATION_PROMPT_WITH_KNOWLEDGE.format(
                    knowledge=knowledge_text,
                    question=q
                )
            else:
                prompt = GENERATION_PROMPT_NO_KNOWLEDGE.format(question=q)

        
        # Project visual features (ensure dtype matches)
        visual_features = visual_features.to(self.device).to(self.llm_dtype)
        visual_tokens = self.projection(visual_features)  # [1, num_visual_tokens, hidden_dim]
        
        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = prompt_ids.input_ids.to(self.device)
        
        # Get prompt embeddings
        prompt_embeds = self.llm.get_input_embeddings()(input_ids)  # [1, seq_len, hidden_dim]
        
        # Find image marker positions and inject visual tokens (KEEP the markers)
        image_start_pos = (input_ids[0] == self.image_start_token_id).nonzero(as_tuple=True)[0]
        image_end_pos = (input_ids[0] == self.image_end_token_id).nonzero(as_tuple=True)[0]
        
        if len(image_start_pos) > 0 and len(image_end_pos) > 0:
            start_idx = image_start_pos[0].item()
            end_idx = image_end_pos[0].item()
            
            # Split: [before_markers] + [visual_tokens] + [after_markers]
            before_embeds = prompt_embeds[0, :start_idx+1]
            after_embeds = prompt_embeds[0, end_idx:]
            
            # Concatenate
            combined_embeds = torch.cat([before_embeds.unsqueeze(0), visual_tokens, after_embeds.unsqueeze(0)], dim=1)
        else:
            # Fallback: prepend visual tokens if no markers found
            combined_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)
        
        # Build generation config based on parameters
        gen_kwargs = {
            'inputs_embeds': combined_embeds,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'no_repeat_ngram_size': no_repeat_ngram_size,
        }
        
        if num_beams > 1:
            # Beam search mode - better for NLU metrics
            gen_kwargs.update({
                'do_sample': False,
                'num_beams': num_beams,
                'length_penalty': length_penalty,
                'early_stopping': early_stopping,
            })
            
            # Diverse beam search if requested
            if diversity_penalty > 0 and num_beam_groups > 1:
                gen_kwargs.update({
                    'num_beam_groups': num_beam_groups,
                    'diversity_penalty': diversity_penalty,
                })
        else:
            # Greedy or sampling mode
            do_sample = temperature > 0
            gen_kwargs.update({
                'do_sample': do_sample,
                'temperature': temperature if do_sample else 1.0,
            })
            # Only pass top_p when actually sampling — these flags are invalid
            # for greedy decoding and produce a HuggingFace warning.
            if do_sample:
                gen_kwargs['top_p'] = top_p
        
        # Generate
        outputs = self.llm.generate(**gen_kwargs)
        
        # Decode output tokens.
        # NOTE: when using inputs_embeds, llm.generate() returns ONLY the newly
        # generated tokens (not the prompt), so the output is already the
        # assistant response only. skip_special_tokens removes <|im_start|> etc.
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Safety strip: in case the model echoes any ChatML wrapper despite the
        # above (e.g. when called via input_ids instead of inputs_embeds),
        # extract only the assistant portion.
        if "<|im_start|>assistant" in generated_text:
            generated_text = generated_text.split("<|im_start|>assistant")[-1]
        elif "<|im_end|>" in generated_text:
            # Strip any trailing end marker
            generated_text = generated_text.split("<|im_end|>")[0]

        # Restore use_cache to its pre-generation state so training is unaffected.
        self.llm.config.use_cache = llm_config_use_cache

        return generated_text.strip()
    
    def to(self, device):
        """
        Override to() method to properly move all model components to target device.
        
        This is critical when loading checkpoints on CPU and then moving to CUDA,
        as the LLM may have device_map='cpu' set.
        """
        # Update internal device tracker
        self.device = device if isinstance(device, str) else str(device)
        
        # Move projection layer
        self.projection = self.projection.to(device)
        
        # Move LLM - need to handle device_map models specially
        # For models loaded with device_map, we need to move them differently
        if hasattr(self.llm, 'hf_device_map'):
            # Model was loaded with device_map, need to reload or use accelerate
            # For now, just move it normally and clear the device_map
            self.llm.hf_device_map = None
        
        # Move LLM to target device
        self.llm = self.llm.to(device)
        
        return self
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: dict = None, val_loss: float = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'projection_state_dict': self.projection.state_dict(),
            'val_loss': val_loss,
        }
        
        # CRITICAL: Save LoRA weights for Stage 2 (QLoRA) models
        if self.use_qlora:
            checkpoint['lora_state_dict'] = self.llm.state_dict()
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """
        Load checkpoint and ensure all components are on correct device.
        
        Returns:
            epoch: The epoch number from checkpoint
            optimizer_state: Optimizer state dict if load_optimizer=True, else None
        """
        print(f"Loading checkpoint from: {path}")
        
        # Load checkpoint to current device (not CPU!)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load projection weights
        self.projection.load_state_dict(checkpoint['projection_state_dict'])
        
        # Load LoRA weights if present (Stage 2 models)
        if 'lora_state_dict' in checkpoint and self.use_qlora:
            print("  Loading LoRA adapter weights...")
            self.llm.load_state_dict(checkpoint['lora_state_dict'], strict=False)
            print("  ✓ LoRA weights loaded")
        
        # CRITICAL: Explicitly ensure projection is on the correct device and dtype
        self.projection = self.projection.to(self.device)
        if hasattr(self, 'llm_dtype'):
            self.projection = self.projection.to(self.llm_dtype)
        
        print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']+1}")
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        
        # Verify device placement
        sample_param = next(self.projection.parameters())
        print(f"  Projection device: {sample_param.device}, dtype: {sample_param.dtype}")
        
        optimizer_state = None
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
        
        epoch = checkpoint['epoch']
        
        # Clean up
        del checkpoint
        torch.cuda.empty_cache()
        
        return epoch, optimizer_state
    
    def to(self, device):
        """
        Override to() to ensure ALL components are moved to the correct device.
        
        IMPORTANT: This must handle:
        1. LLM (AutoModelForCausalLM)
        2. Projection layer (nn.Module)
        3. Any intermediate buffers
        """
        # Move base class first
        super().to(device)
        
        # Explicitly move LLM
        self.llm = self.llm.to(device)
        
        # Explicitly move projection layer and all its submodules
        self.projection = self.projection.to(device)
        
        # Ensure projection matches LLM dtype (bfloat16)
        if hasattr(self, 'llm_dtype'):
            self.projection = self.projection.to(self.llm_dtype)
        
        # Update stored device
        self.device = device
        
        return self


if __name__ == "__main__":
    # Test model initialization
    print("="*60)
    print("Testing VLM Model")
    print("="*60)
    
    # Stage 1: Projection only
    print("\n[Stage 1 Test] Projection layer training")
    model_stage1 = VisionLanguageModel(
        visual_dim=768,
        num_visual_tokens=64,
        use_qlora=False,
        freeze_llm=True
    )
    
    total, trainable = model_stage1.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {100*trainable/total:.2f}%")
    
    # Test forward pass
    dummy_visual = torch.randn(2, 768).to(model_stage1.device)
    dummy_input_ids = torch.randint(0, 1000, (2, 50)).to(model_stage1.device)
    dummy_labels = dummy_input_ids.clone()
    
    outputs = model_stage1(
        visual_features=dummy_visual,
        input_ids=dummy_input_ids,
        labels=dummy_labels
    )
    print(f"\nForward pass successful!")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test generation
    print("\n[Testing Generation]")
    report = model_stage1.generate(
        visual_features=dummy_visual[0],
        prompt="Provide a concise single-paragraph pathology diagnosis:",
        max_new_tokens=50
    )
    print(f"Generated report: {report[:200]}...")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)