"""
Inference Module for FitnessGPT

Load a fine-tuned model and run inference on new exercise data.
Supports both LoRA adapter loading and merged model loading.

Usage:
    python inference.py --model ./outputs/fitness_gpt/lora_adapter --json ./data/pullup_analysis.json
    python inference.py --model ./outputs/fitness_gpt/lora_adapter --prompt "Analyze this pull-up set..."

Author: FitnessAQA Capstone
"""

import argparse
import json
import sys
from typing import Optional

from time_series_encoder import TimeSeriesEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Run FitnessGPT inference")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to LoRA adapter or merged model")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                        help="Base model (needed for LoRA loading)")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to pullup_analysis.json for automatic encoding")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (instead of auto-encoding from JSON)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p nucleus sampling")
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame stride for encoding")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Run in interactive mode")
    
    return parser.parse_args()


SYSTEM_PROMPT = (
    "You are FitnessGPT, an expert AI biomechanics coach. You analyze exercise "
    "form using pose estimation data encoded as structured tokens. Provide precise, "
    "actionable coaching feedback grounded in the data."
)


def load_model(model_path: str, base_model: str):
    """Load the fine-tuned model."""
    from unsloth import FastLanguageModel
    import os
    
    # Check if this is a LoRA adapter or a merged model
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        print(f"Loading base model: {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        print(f"Loading LoRA adapter: {model_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print(f"Loading merged model: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
    
    # Switch to inference mode (2x faster)
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def build_prompt(instruction: str, system: str = SYSTEM_PROMPT) -> str:
    """Build an Alpaca-format prompt."""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
{system}

### Instruction:
{instruction}

### Response:
"""


def encode_exercise_data(json_path: str, stride: int = 5) -> str:
    """Encode exercise JSON into an analysis prompt."""
    encoder = TimeSeriesEncoder()
    result = encoder.encode_from_json(json_path, stride=stride)
    
    prompt = (
        f"Analyze the following pull-up exercise data and provide a comprehensive "
        f"quality assessment with coaching feedback.\n\n"
        f"Pose Data Summary:\n{result['summary_tokens']}\n\n"
        f"Quality Indicators: {result['quality_tokens']}\n\n"
        f"Encoded Sequence (excerpt):\n{result['frame_tokens'][:1500]}"
    )
    
    return prompt


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate a response from the model."""
    full_prompt = build_prompt(prompt)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        use_cache=True,
    )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return response.strip()


def interactive_mode(model, tokenizer, args, encoder: Optional[TimeSeriesEncoder] = None):
    """Run interactive inference loop."""
    print("\n" + "=" * 50)
    print("  FitnessGPT Interactive Mode")
    print("  Type 'quit' to exit, 'load <path>' to analyze a JSON file")
    print("=" * 50 + "\n")
    
    current_context = None
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower().startswith("load "):
            json_path = user_input[5:].strip()
            try:
                prompt = encode_exercise_data(json_path, args.stride)
                current_context = prompt
                print(f"\n[Loaded and encoded {json_path}]")
                print(f"[Now generating analysis...]\n")
                
                response = generate(
                    model, tokenizer, prompt,
                    args.max_new_tokens, args.temperature, args.top_p
                )
                print(f"FitnessGPT: {response}\n")
            except Exception as e:
                print(f"Error loading file: {e}\n")
            continue
        
        # Use current context if available
        if current_context:
            prompt = f"{user_input}\n\nContext from loaded exercise data:\n{current_context[:1000]}"
        else:
            prompt = user_input
        
        response = generate(
            model, tokenizer, prompt,
            args.max_new_tokens, args.temperature, args.top_p
        )
        print(f"FitnessGPT: {response}\n")


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model, args.base_model)
    
    if args.interactive:
        interactive_mode(model, tokenizer, args)
        return
    
    # Determine prompt
    if args.json:
        print(f"Encoding exercise data from {args.json}...")
        prompt = encode_exercise_data(args.json, args.stride)
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: Provide either --json or --prompt")
        sys.exit(1)
    
    print(f"\nGenerating response (max {args.max_new_tokens} tokens)...\n")
    
    response = generate(
        model, tokenizer, prompt,
        args.max_new_tokens, args.temperature, args.top_p
    )
    
    print("=" * 50)
    print("FitnessGPT Response:")
    print("=" * 50)
    print(response)


if __name__ == "__main__":
    main()
