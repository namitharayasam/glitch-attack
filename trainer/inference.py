"""
This script generates responses from language models with injected glitch tokens
to analyze their behavior. It uses chat templates and supports batch processing.
"""

import os
import time
import argparse
import gc
import pandas as pd
import torch
import torch._dynamo
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from google.cloud import storage

torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

GLITCH_TOKENS = ['your_tokens']


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download a file from Google Cloud Storage."""
    print(f"Downloading {source_blob_name} from GCS bucket {bucket_name}...", flush=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded to {destination_file_name}", flush=True)


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to Google Cloud Storage."""
    print(f"Uploading {source_file_name} to GCS bucket {bucket_name}...", flush=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded to {destination_blob_name}", flush=True)


def insert_token(prompt, token, position):
    """
    Insert a glitch token into a prompt at the specified position.
    
    Args:
        prompt: Original prompt text
        token: Token to insert
        position: Where to insert ('prefix', 'middle', or 'suffix')
    
    Returns:
        Modified prompt with inserted token
    """
    words = prompt.split()
    if not words:
        return prompt
    
    if position == "prefix":
        return token + " " + prompt
    elif position == "middle":
        mid = len(words) // 2
        return " ".join(words[:mid] + [token] + words[mid:])
    elif position == "suffix":
        return prompt + " " + token
    
    return prompt


def load_model_and_tokenizer(model_name, hf_token=None):
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name or path of the model
        hf_token: HuggingFace API token (optional)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading tokenizer: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token", flush=True)
    
    print(f"Loading model: {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token
    )
    
    # Check for chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        print("Model has built-in chat template", flush=True)
    else:
        print("⚠️ No chat template found", flush=True)
    
    model.eval()
    print(f"Model loaded on device: {model.device}", flush=True)
    
    return model, tokenizer


def batch_generate(model, tokenizer, prompts_batch, max_new_tokens=100, max_length=512):
    """
    Generate responses for a batch of prompts using chat template formatting.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        prompts_batch: List of prompts
        max_new_tokens: Maximum number of new tokens to generate
        max_length: Maximum input sequence length
    
    Returns:
        List of generated responses
    """
    # Format prompts using chat template
    formatted_prompts = []
    for prompt in prompts_batch:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    responses = []
    for i, output in enumerate(outputs):
        input_length = inputs['input_ids'][i].shape[0]
        new_tokens = output[input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses


def test_chat_template(tokenizer, test_prompt):
    """Test and display chat template formatting."""
    print("\n" + "="*50)
    print("TESTING CHAT TEMPLATE FORMATTING")
    print("="*50)
    print(f"Original prompt: {test_prompt[:100]}...")
    
    test_messages = [{"role": "user", "content": test_prompt}]
    test_formatted = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"Formatted prompt: {test_formatted[:200]}...")
    print("="*50 + "\n")


def run_glitch_analysis(
    prompts,
    model,
    tokenizer,
    glitch_tokens,
    positions,
    batch_size=32,
    max_length=256,
    max_new_tokens=100
):
    """
    Run glitch token analysis on prompts.
    
    Args:
        prompts: List of original prompts
        model: Loaded language model
        tokenizer: Loaded tokenizer
        glitch_tokens: List of glitch tokens to test
        positions: List of positions to test ('prefix', 'middle', 'suffix')
        batch_size: Batch size for processing
        max_length: Maximum input sequence length
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Test chat template formatting
    if prompts:
        test_chat_template(tokenizer, prompts[0])

    print("\nProcessing pure prompts (no glitch tokens)...", flush=True)
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in range(0, len(prompts), batch_size):
        start_time = time.time()
        batch_prompts = prompts[i:i + batch_size]
        
        try:
            batch_responses = batch_generate(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=max_new_tokens,
                max_length=max_length
            )
            
            for j, response in enumerate(batch_responses):
                results.append({
                    "prompt_id": i + j,
                    "prompt": prompts[i + j],
                    "glitch_token": None,
                    "position": "pure",
                    "response": response
                })
            
            elapsed = time.time() - start_time
            print(
                f"Pure batch {i // batch_size + 1}/{total_batches} "
                f"completed in {elapsed:.2f}s",
                flush=True
            )
        
        except Exception as e:
            print(f"Error in pure batch {i // batch_size + 1}: {e}", flush=True)
            for j in range(len(batch_prompts)):
                results.append({
                    "prompt_id": i + j,
                    "prompt": prompts[i + j],
                    "glitch_token": None,
                    "position": "pure",
                    "response": "[ERROR]"
                })
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Process each glitch token and position combination
    for token in glitch_tokens:
        for position in positions:
            print(f"\nProcessing: token='{token}', position='{position}'", flush=True)
            
            # Create modified prompts
            modified_prompts = [insert_token(p, token, position) for p in prompts]
            total_batches = (len(modified_prompts) + batch_size - 1) // batch_size
            
            # Process in batches
            for i in range(0, len(modified_prompts), batch_size):
                start_time = time.time()
                batch_prompts = modified_prompts[i:i + batch_size]
                
                try:
                    batch_responses = batch_generate(
                        model,
                        tokenizer,
                        batch_prompts,
                        max_new_tokens=max_new_tokens,
                        max_length=max_length
                    )
                    
                    # Store results
                    for j, response in enumerate(batch_responses):
                        results.append({
                            "prompt_id": i + j,
                            "prompt": prompts[i + j],
                            "glitch_token": token,
                            "position": position,
                            "response": response
                        })
                    
                    elapsed = time.time() - start_time
                    print(
                        f"Batch {i // batch_size + 1}/{total_batches} "
                        f"completed in {elapsed:.2f}s",
                        flush=True
                    )
                
                except Exception as e:
                    print(f"Error in batch {i // batch_size + 1}: {e}", flush=True)
                    # Store error results
                    for j in range(len(batch_prompts)):
                        results.append({
                            "prompt_id": i + j,
                            "prompt": prompts[i + j],
                            "glitch_token": token,
                            "position": position,
                            "response": "[ERROR]"
                        })
                
                # Clean up
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Glitch Token Analysis")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model name or path")
    parser.add_argument("--bucket", type=str, default="glitch-bucket",
                        help="GCS bucket name")
    parser.add_argument("--input-gcs", type=str, default="harmbench_behaviors_text_test.csv",
                        help="Input CSV path in GCS")
    parser.add_argument("--output-gcs", type=str, default="outputs/glitch_analysis.csv",
                        help="Output CSV path in GCS")
    parser.add_argument("--input-local", type=str, default="/tmp/input.csv",
                        help="Local path for downloaded input")
    parser.add_argument("--output-local", type=str, default="/tmp/output.csv",
                        help="Local path for output")
    parser.add_argument("--glitch-tokens", type=str, nargs='+', default=GLITCH_TOKENS,
                        help="List of glitch tokens to test")
    parser.add_argument("--positions", type=str, nargs='+', default=['suffix'],
                        help="Positions to test: prefix, middle, suffix")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of examples to process (None for all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum input sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token")
    
    args = parser.parse_args()
    
    # Use environment variable for HF token if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    print("="*50)
    print("GLITCH TOKEN ANALYSIS PIPELINE")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Input: gs://{args.bucket}/{args.input_gcs}")
    print(f"Output: gs://{args.bucket}/{args.output_gcs}")
    print(f"Glitch tokens: {args.glitch_tokens}")
    print(f"Positions: {args.positions}")
    print(f"Batch size: {args.batch_size}")
    print("="*50 + "\n")
    
    # Check CUDA
    print("Checking CUDA availability...", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print()
    
    # Download input data
    download_from_gcs(args.bucket, args.input_gcs, args.input_local)
    
    # Load prompts
    print("Loading prompts from CSV...", flush=True)
    df = pd.read_csv(args.input_local)
    prompts = df["Behavior"].dropna().tolist()
    
    if args.num_examples:
        prompts = prompts[:args.num_examples]
    
    print(f"Loaded {len(prompts)} prompts", flush=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, hf_token)
    
    # Run analysis
    results = run_glitch_analysis(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        glitch_tokens=args.glitch_tokens,
        positions=args.positions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save results
    print(f"\nSaving results to {args.output_local}...", flush=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_local, index=False)
    print(f"Saved {len(results)} results", flush=True)
    
    # Upload to GCS
    upload_to_gcs(args.bucket, args.output_local, args.output_gcs)
    
    print("\nPipeline completed successfully!", flush=True)


if __name__ == "__main__":
    main()