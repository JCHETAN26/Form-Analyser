"""
Unsloth Fine-Tuning Script for FitnessGPT

Uses Unsloth AI for 4x faster LoRA fine-tuning on fitness biomechanics data.
Trains a GPT-class model to understand pose time series tokens and provide
expert coaching feedback.

Supports:
  - Llama 3.1/3.2, Mistral, Phi-3, Gemma 2, Qwen 2.5
  - 4-bit QLoRA for memory-efficient training
  - Alpaca and ShareGPT chat formats
  - GGUF/VLLM export for deployment

Usage:
    python train_unsloth.py --data ./outputs/train_alpaca.json
    python train_unsloth.py --data ./outputs/train_sharegpt.json --format sharegpt
    python train_unsloth.py --data ./outputs/train_alpaca.json --model unsloth/Llama-3.2-3B-Instruct

Author: FitnessAQA Capstone
"""

import argparse
import json
import os

# Limit to 1 GPU to avoid multi-device conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on fitness biomechanics data using Unsloth")
    
    # Data
    parser.add_argument("--data", type=str, default="./outputs/train_alpaca.json",
                        help="Path to training data JSON")
    parser.add_argument("--format", type=str, default="alpaca", choices=["alpaca", "sharegpt"],
                        help="Dataset format")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Model
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                        help="Base model from Unsloth hub")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/fitness_gpt",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_method", type=str, default="lora",
                        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
                        help="Model save method")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HF Hub model ID to push to")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Build dataset only, don't train")
    
    return parser.parse_args()


# ───────────────── Alpaca Prompt Template ─────────────────

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
{system}

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
{system}

### Instruction:
{instruction}

### Response:
{output}"""


def formatting_prompts_func(examples):
    """Format examples into Alpaca-style prompts for training."""
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    systems = examples.get("system", [""] * len(instructions))
    
    texts = []
    for instruction, inp, output, system in zip(instructions, inputs, outputs, systems):
        if inp and inp.strip():
            text = ALPACA_TEMPLATE.format(
                system=system, instruction=instruction,
                input=inp, output=output
            )
        else:
            text = ALPACA_TEMPLATE_NO_INPUT.format(
                system=system, instruction=instruction, output=output
            )
        texts.append(text)
    
    return {"text": texts}


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  FitnessGPT — Unsloth Fine-Tuning Pipeline")
    print("=" * 60)
    
    # ─────────── Step 1: Prepare Dataset ───────────
    print(f"\n[1/5] Loading dataset from {args.data}...")
    
    if not os.path.exists(args.data):
        print(f"  Dataset not found at {args.data}")
        print(f"  Run `python dataset_builder.py` first to generate the training data.")
        sys.exit(1)
    
    with open(args.data, 'r') as f:
        raw_data = json.load(f)
    
    print(f"  Loaded {len(raw_data)} examples")
    
    if args.dry_run:
        print("\n  [DRY RUN] Dataset loaded successfully. Exiting.")
        print(f"  Sample instruction: {raw_data[0]['instruction'][:100]}...")
        print(f"  Sample output: {raw_data[0]['output'][:100]}...")
        return
    
    # ─────────── Step 2: Load Model with Unsloth ───────────
    print(f"\n[2/5] Loading model: {args.model}")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.float16,
    )
    
    print(f"  Model loaded: {model.config._name_or_path}")
    print(f"  Parameters: {model.num_parameters():,}")
    
    # ─────────── Step 3: Apply LoRA ───────────
    print(f"\n[3/5] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=args.seed,
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    # ─────────── Step 4: Prepare Training Data ───────────
    print(f"\n[4/5] Preparing training data ({args.format} format)...")
    
    from datasets import Dataset
    
    if args.format == "sharegpt":
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
        
        from unsloth.chat_templates import standardize_sharegpt
        dataset = standardize_sharegpt(Dataset.from_list(raw_data))
        
        from unsloth.chat_templates import train_on_responses_only
        
        def sharegpt_format(examples):
            convos = examples["conversations"]
            texts = [
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                for convo in convos
            ]
            return {"text": texts}
        
        dataset = dataset.map(sharegpt_format, batched=True)
    else:
        # Alpaca format
        dataset = Dataset.from_list(raw_data)
        dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Train/val split
    if args.val_split > 0:
        split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train: {len(train_dataset)} (no validation split)")
    
    # ─────────── Step 5: Train ───────────
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",  # Use "wandb" for experiment tracking
        
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Eval
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        
        # Performance
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        
        seed=args.seed,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,  # Pack short examples together for efficiency
        args=training_args,
    )
    
    # Show memory stats
    gpu_stats = None
    try:
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
            print(f"  GPU: {gpu_stats.name} ({gpu_stats.total_mem / 1024**3:.1f} GB)")
            print(f"  Reserved VRAM: {reserved} GB")
    except Exception:
        pass
    
    # Train
    trainer_stats = trainer.train()
    
    print(f"\n  Training complete!")
    print(f"  Loss: {trainer_stats.training_loss:.4f}")
    print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.0f}s")
    if gpu_stats:
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"  Peak VRAM: {peak_mem} GB")
    
    # ─────────── Save Model ───────────
    print(f"\n  Saving model ({args.save_method})...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_method == "lora":
        model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
        print(f"  LoRA adapter saved to {args.output_dir}/lora_adapter")
        
    elif args.save_method == "merged_16bit":
        model.save_pretrained_merged(
            os.path.join(args.output_dir, "merged_16bit"),
            tokenizer, save_method="merged_16bit"
        )
        print(f"  Merged 16-bit model saved.")
        
    elif args.save_method == "merged_4bit":
        model.save_pretrained_merged(
            os.path.join(args.output_dir, "merged_4bit"),
            tokenizer, save_method="merged_4bit_forced"
        )
        print(f"  Merged 4-bit model saved.")
        
    elif args.save_method == "gguf":
        # Export to GGUF for llama.cpp / Ollama
        model.save_pretrained_gguf(
            os.path.join(args.output_dir, "gguf"),
            tokenizer,
            quantization_method="q4_k_m",  # Good balance of quality/size
        )
        print(f"  GGUF model saved (q4_k_m quantization).")
    
    # Optional: push to HF Hub
    if args.push_to_hub and args.hub_model_id:
        print(f"\n  Pushing to HF Hub: {args.hub_model_id}")
        model.push_to_hub(args.hub_model_id, tokenizer=tokenizer)
        print(f"  Push complete.")
    
    print("\n" + "=" * 60)
    print("  Training pipeline complete!")
    print(f"  Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
