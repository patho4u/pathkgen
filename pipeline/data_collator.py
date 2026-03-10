"""
Data collator for Vision-Language Model training.

Handles batching of slide features and text prompt_message with proper padding.
"""

import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer
from config import (
    TRAINING_PROMPT_NO_KNOWLEDGE,
    TRAINING_PROMPT_WITH_KNOWLEDGE,
    TRAINING_PROMPT_WSI_BENCH,
    TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    CHECKPOINTS_DIR
)


class VLMDataCollator:
    """
    Collates batches for VLM training.
    
    Handles:
    - Variable-length text sequences
    - Proper padding and attention masks
    - Label preparation for causal LM loss
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_format: str = 'wsi-bench',
        max_length: int = 512,
        padding: str = "longest",
        use_knowledge_guidance: bool = False,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            dataset_format: Dataset format ('wsi-bench' or 'histgen')
            max_length: Maximum sequence length
            padding: Padding strategy ('longest' or 'max_length')
            use_knowledge_guidance: Whether to use knowledge-guided prompts
        """
        self.tokenizer = tokenizer
        self.dataset_format = dataset_format
        self.max_length = max_length
        self.padding = padding
        self.use_knowledge_guidance = use_knowledge_guidance
        
        # Select appropriate prompt template based on dataset format and knowledge guidance
        if dataset_format == 'wsi-bench':
            if use_knowledge_guidance:
                self.prompt_template = TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE
            else:
                self.prompt_template = TRAINING_PROMPT_WSI_BENCH
        else:  # histgen
            if use_knowledge_guidance:
                self.prompt_template = TRAINING_PROMPT_WITH_KNOWLEDGE
            else:
                self.prompt_template = TRAINING_PROMPT_NO_KNOWLEDGE
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dicts from dataset with keys:
                - 'slide_embedding': [768] tensor
                - 'report': str
                - 'id': str
                - 'organ': str
                
        Returns:
            Dictionary with:
                - 'visual_features': [batch, 768]
                - 'input_ids': [batch, seq_len]
                - 'attention_mask': [batch, seq_len]
                - 'labels': [batch, seq_len]
                - 'ids': List[str]
        """
        # Extract slide features
        visual_features = torch.stack([item['slide_embedding'] for item in batch])
        
        # Format prompts with prompt template based on dataset format
        if self.dataset_format == 'wsi-bench':
            # WSI-Bench: use question and answer fields
            if self.use_knowledge_guidance:
                prompt_message = [
                    self.prompt_template.format(
                        knowledge=item['knowledge_text'],
                        question=item['question'],
                        answer=item['answer']
                    ) for item in batch
                ]
            else:
                prompt_message = [
                    self.prompt_template.format(
                        question=item['question'],
                        answer=item['answer']
                    ) for item in batch
                ]
        else:  # histgen
            # HistGen: use report field with fixed prompt
            if self.use_knowledge_guidance:
                prompt_message = [
                    self.prompt_template.format(
                        knowledge=item['knowledge_text'],
                        report=item['answer']  # Uses 'answer' which maps to 'report' in HistGen
                    ) for item in batch
                ]
            else:
                prompt_message = [
                    self.prompt_template.format(
                        report=item['answer']  # Uses 'answer' which maps to 'report' in HistGen
                    ) for item in batch
                ]
        
        # Tokenize prompt_message with truncation strategy that preserves image markers
        tokenized = self.tokenizer(
            prompt_message,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Prepare labels for causal LM:
        # Only the ANSWER tokens should have active labels (standard instruction tuning).
        # Everything before <|im_start|>assistant (system prompt, knowledge, question)
        # is masked to -100 so the model learns to USE knowledge, not PREDICT it.
        labels = tokenized['input_ids'].clone()

        # Find the token ids for "<|im_start|>assistant" to locate answer start
        assistant_marker_ids = self.tokenizer(
            "<|im_start|>assistant",
            add_special_tokens=False
        ).input_ids  # typically 2 tokens: <|im_start|> and "assistant"

        input_ids_np = tokenized['input_ids']
        for i in range(input_ids_np.shape[0]):
            seq = input_ids_np[i].tolist()
            # Find the last occurrence of the assistant marker
            # (last because the prompt may contain multi-turn examples)
            marker_pos = -1
            marker_len = len(assistant_marker_ids)
            for j in range(len(seq) - marker_len, -1, -1):
                if seq[j:j + marker_len] == assistant_marker_ids:
                    marker_pos = j
                    break

            if marker_pos != -1:
                # Mask everything up to and including <|im_start|>assistant\n
                # (+1 for the newline token that follows the marker)
                mask_until = marker_pos + marker_len
                labels[i, :mask_until] = -100
            else:
                # Fallback: mask everything (should not happen with correct prompts)
                labels[i, :] = -100

        # Also mask pad tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Collect metadata
        ids = [item['id'] for item in batch]
        
        # Prepare result dictionary
        result = {
            'visual_features': visual_features,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'ids': ids
        }
        
        # Include knowledge_text for generation (if using knowledge guidance)
        if self.use_knowledge_guidance:
            knowledge_texts = [item.get('knowledge_text', '') for item in batch]
            result['knowledge_texts'] = knowledge_texts
        
        # Always include questions for generation (WSI-Bench has per-sample questions)
        result['questions'] = [item.get('question', '') or '' for item in batch]
        
        return result



if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    from config import IMAGE_START_TOKEN, IMAGE_END_TOKEN

    LLM_PATH = os.path.join(CHECKPOINTS_DIR, "Qwen2.5-7B-Instruct")
    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    errors = []

    def check(name, condition, detail=""):
        if condition:
            print(f"  {PASS} {name}")
        else:
            print(f"  {FAIL} {name}" + (f": {detail}" if detail else ""))
            errors.append(name)

    print("=" * 60)
    print("VLMDataCollator Tests")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    ANSWER_1 = "Invasive ductal carcinoma, grade 2."
    ANSWER_2 = "Follicular thyroid adenoma with focal atypia."
    QUESTION_1 = "Describe the key diagnostic features of this slide."
    QUESTION_2 = "What is the primary diagnosis for this tissue?"
    KNOWLEDGE_1 = "Breast adenocarcinoma. Site: breast."
    KNOWLEDGE_2 = "Thyroid follicular neoplasm. Site: thyroid."

    # ── TEST 1: WSI-Bench without knowledge ──────────────────────────
    print("\nTest 1: WSI-Bench, no knowledge — label masking")
    collator = VLMDataCollator(tokenizer=tokenizer, dataset_format='wsi-bench',
                               max_length=512, use_knowledge_guidance=False)
    wsi_batch = [
        {'id': 's1', 'organ': 'breast', 'slide_embedding': torch.randn(128, 768),
         'question': QUESTION_1, 'answer': ANSWER_1},
        {'id': 's2', 'organ': 'thyroid', 'slide_embedding': torch.randn(128, 768),
         'question': QUESTION_2, 'answer': ANSWER_2},
    ]
    out = collator(wsi_batch)

    for i, (answer, item) in enumerate(zip([ANSWER_1, ANSWER_2], wsi_batch)):
        ids = out['input_ids'][i]
        lbls = out['labels'][i]
        active_mask = (lbls != -100)
        active_ids = ids[active_mask]
        decoded_active = tokenizer.decode(active_ids, skip_special_tokens=True).strip()
        check(f"sample {i+1}: active labels contain answer",
              answer in decoded_active, f"got: '{decoded_active[:80]}'")
        first_active = active_mask.nonzero(as_tuple=True)[0]
        if len(first_active) > 0:
            first_pos = first_active[0].item()
            check(f"sample {i+1}: all prompt tokens are -100",
                  (lbls[:first_pos] == -100).all().item())
        check(f"sample {i+1}: question in batch['questions']",
              out['questions'][i] == item['question'])

    # ── TEST 2: WSI-Bench WITH knowledge ──────────────────────────────
    print("\nTest 2: WSI-Bench + knowledge — knowledge tokens masked, answer active")
    collator_k = VLMDataCollator(tokenizer=tokenizer, dataset_format='wsi-bench',
                                 max_length=512, use_knowledge_guidance=True)
    batch_k = [
        {'id': 's1', 'organ': 'breast', 'slide_embedding': torch.randn(128, 768),
         'question': QUESTION_1, 'answer': ANSWER_1, 'knowledge_text': KNOWLEDGE_1},
        {'id': 's2', 'organ': 'thyroid', 'slide_embedding': torch.randn(128, 768),
         'question': QUESTION_2, 'answer': ANSWER_2, 'knowledge_text': KNOWLEDGE_2},
    ]
    out_k = collator_k(batch_k)

    check("knowledge_texts present in batch", 'knowledge_texts' in out_k)
    check("knowledge_text[0] correct", out_k['knowledge_texts'][0] == KNOWLEDGE_1)
    check("knowledge_text[1] correct", out_k['knowledge_texts'][1] == KNOWLEDGE_2)
    check("questions present in batch", 'questions' in out_k)

    for i, (answer, know) in enumerate(zip([ANSWER_1, ANSWER_2], [KNOWLEDGE_1, KNOWLEDGE_2])):
        ids = out_k['input_ids'][i]
        lbls = out_k['labels'][i]
        # First token of knowledge text must be masked
        know_ids = tokenizer(know, add_special_tokens=False).input_ids
        know_tok = know_ids[0]
        positions = (ids == know_tok).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            pos = positions[0].item()
            check(f"sample {i+1}: first knowledge token at pos {pos} is -100",
                  lbls[pos].item() == -100)
        active_ids = ids[lbls != -100]
        decoded_active = tokenizer.decode(active_ids, skip_special_tokens=True).strip()
        check(f"sample {i+1}: active labels still contain answer",
              answer in decoded_active, f"got: '{decoded_active[:80]}'")

    # ── TEST 3: Per-sample questions are independent ───────────────────
    print("\nTest 3: Per-sample questions are independent")
    check("question[0] != question[1]", out_k['questions'][0] != out_k['questions'][1])
    check("question[0] == QUESTION_1", out_k['questions'][0] == QUESTION_1)
    check("question[1] == QUESTION_2", out_k['questions'][1] == QUESTION_2)

    # ── TEST 4: HistGen format (uses {report} not {question}) ──────────
    print("\nTest 4: HistGen format — questions key still emitted")
    collator_hg = VLMDataCollator(tokenizer=tokenizer, dataset_format='histgen',
                                  max_length=512, use_knowledge_guidance=False)
    batch_hg = [
        {'id': 's1', 'organ': 'breast', 'slide_embedding': torch.randn(128, 768),
         'answer': ANSWER_1},
    ]
    out_hg = collator_hg(batch_hg)
    check("questions key present for histgen", 'questions' in out_hg)
    ids_hg = out_hg['input_ids'][0]
    lbls_hg = out_hg['labels'][0]
    active_hg = tokenizer.decode(ids_hg[lbls_hg != -100], skip_special_tokens=True).strip()
    check("histgen active labels contain answer",
          ANSWER_1 in active_hg, f"got: '{active_hg[:80]}'")

    # ── SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if errors:
        print(f"\033[91mFAILED ({len(errors)}): {errors}\033[0m")
        sys.exit(1)
    else:
        print("\033[92mAll collator tests passed!\033[0m")
    print("=" * 60)

