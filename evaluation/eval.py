import argparse
import os
os.environ["VLLM_CACHE_ROOT"] = "/tmp/vllm_cache"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import csv  # Add CSV import
from tqdm import tqdm
from datetime import datetime

import openai
import google.generativeai as genai
import anthropic
import concurrent.futures  # Add this import for parallel processing

# Update this line to import all methods from the methods package
from methods import base, tool_augmentation, self_correction, self_consistency, rag
from evaluation.auto_judge import judge_responses


# Define multimodal and text-only model lists
MULTIMODAL_MODELS = [
    "gpt-4o", 
    "gpt-4.1",
    "o4-mini",
    "o3",
    "gpt-5",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",

    "claude-3-7-sonnet", 
    "claude-3-5", 
    "claude-sonnet-4",
    
    "qwen2.5-vl-32b-instruct",

    "llama-4-maverick-17b"
]

# Other models are considered text-only by default

def parse_args():
    parser = argparse.ArgumentParser(description="Run QA evaluation.")
    parser.add_argument("--input", type=str, default="datasets/MatSciBench/qa.csv",
                        help="Path to the input CSV file containing the QA dataset.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Model name to use with vLLM.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling (use 0.0 for deterministic output).")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--method", type=str, default="base",
                        choices=["base", "tool", "correction", "consistency", "rag"], 
                        help="Methods to use for solving questions.")
    parser.add_argument("--gpt_api_key", type=str, default=None,
                        help="OpenAI API key (will use OPENAI_API_KEY env var if not provided).")
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Gemini API key (will use GEMINI_API_KEY env var if not provided).")
    parser.add_argument("--claude_api_key", type=str, default=None,
                        help="Claude API key (will use ANTHROPIC_API_KEY env var if not provided).")
    parser.add_argument("--deepseek_api_key", type=str, default=None,
                        help="DeepSeek API key (will use DEEPSEEK_API_KEY env var if not provided).")
    parser.add_argument("--qwen_api_key", type=str, default=None,
                        help="Qwen API key (will use QWEN_API_KEY env var if not provided).")
    parser.add_argument("--lambda_api_key", type=str, default=None,
                        help="Lambda API key (will use LAMBDA_API_KEY env var if not provided).")
    parser.add_argument("--llm_judge", action="store_true",
                        help="Use LLM to judge the correctness of the answer.")
    parser.add_argument("--rule_judge", action="store_true",
                        help="Use rule-based judging for correctness of the answer.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers to use for parallel processing.")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Output directory for results.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate.")
    parser.add_argument("--use_batch", action="store_true",
                        help="Use batch processing for OpenAI and Anthropic models (50% cost reduction).")
    return parser.parse_args()





def main():
    args = parse_args()

    # Validate that at least one judge is enabled
    if not args.llm_judge and not args.rule_judge:
        print("Error: At least one of --llm_judge or --rule_judge must be specified.")
        return

    # Load the QA dataset from CSV.
    try:
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"Error loading file {args.input}: {e}")
        return

    print(f"Loaded {len(data)} questions.")
    if data:
        # Print first row safely using dictionary access
        print(f"Sample: {data[0].get('question', 'N/A')} - {data[0].get('answer', 'N/A')}")

    # Set API keys from environment variables with fallback to command line arguments
    openai_api_key = os.environ.get("OPENAI_API_KEY", args.gpt_api_key)
    gemini_api_key = os.environ.get("GEMINI_API_KEY", args.gemini_api_key)
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY", args.claude_api_key)
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", args.deepseek_api_key)
    qwen_api_key = os.environ.get("QWEN_API_KEY", args.qwen_api_key)
    lambda_api_key = os.environ.get("LAMBDA_API_KEY", args.lambda_api_key)
    # Determine model type
    is_openai_model = args.model.startswith("gpt") or args.model.startswith("o4") or args.model.startswith("o3") or args.model.startswith("o1")
    is_gemini_model = args.model.startswith("gemini")
    is_claude_model = args.model.startswith("claude")
    is_deepseek_model = args.model.startswith("deepseek")
    is_qwen_model = args.model.startswith("qwen3") or args.model.startswith("qwq")
    is_llama_model = args.model.startswith("llama")
    model_type = "vllm"  # default
    
    # Initialize model and API keys
    llm = None
    sampling_params = None

    if is_openai_model:
        if not openai_api_key:
            print("Error: A valid OpenAI API key is required via OPENAI_API_KEY environment variable or --gpt_api_key argument.")
            return
        print(f"Using OpenAI model: {args.model}")
        openai.api_key = openai_api_key
        model_type = "openai"
    elif is_gemini_model:
        if not gemini_api_key:
            print("Error: A valid Gemini API key is required via GEMINI_API_KEY environment variable or --gemini_api_key argument.")
            return
        print(f"Using Gemini model: {args.model}")
        genai.configure(api_key=gemini_api_key)
        model_type = "gemini"
    elif is_claude_model:
        if not claude_api_key:
            print("Error: A valid Claude API key is required via ANTHROPIC_API_KEY environment variable or --claude_api_key argument.")
            return
        print(f"Using Claude model: {args.model}")
        anthropic.api_key = claude_api_key
        model_type = "claude"
    elif is_deepseek_model:
        if not deepseek_api_key:
            print("Error: A valid DeepSeek API key is required via DEEPSEEK_API_KEY environment variable or --deepseek_api_key argument.")
            return
        print(f"Using DeepSeek model: {args.model}")
        # Configure OpenAI client for DeepSeek
        openai.api_key = deepseek_api_key
        openai.base_url = "https://api.deepseek.com"
        model_type = "deepseek"
    elif is_qwen_model:
        if not qwen_api_key:
            print("Error: A valid Qwen API key is required via QWEN_API_KEY environment variable or --qwen_api_key argument.")
            return
        print(f"Using Qwen model: {args.model}")
        # Configure OpenAI client for Qwen
        openai.api_key = qwen_api_key
        openai.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model_type = "qwen"
    elif is_llama_model:
        if not lambda_api_key:
            print("Error: A valid Lambda API key is required via LAMBDA_API_KEY environment variable or --lambda_api_key argument.")
            return
        print(f"Using Llama model: {args.model}")
        # Configure OpenAI client for Lambda
        openai.api_key = lambda_api_key
        openai.base_url = "https://api.lambda.ai/v1"
        model_type = "llama"
    else:
        from vllm import LLM, SamplingParams
        import torch
        # Initialize the vLLM engine for the specified model.
        print(f"Using vLLM model: {args.model}")
        
        # Use all available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Using all {gpu_count} GPUs with tensor parallelism")
        
        llm = LLM(
            model=args.model, 
            dtype='bfloat16', 
            max_model_len=20000,
            tensor_parallel_size=gpu_count,
            trust_remote_code=True
        )
        # Set sampling parameters.
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_tokens
        )


    is_multimodal = any(mm_model in args.model.lower() for mm_model in MULTIMODAL_MODELS)
    
    # Filter questions based on model capabilities
    filtered_data = data
    if not is_multimodal:
        # For non-multimodal models, only evaluate questions without images
        filtered_data = [entry for entry in data if not entry.get('image') or not entry['image'].strip()]
        print(f"Non-multimodal model: filtering to {len(filtered_data)}/{len(data)} questions without images")
    else:
        print(f"Multimodal model: evaluating all {len(data)} questions")
    
    if args.sample_size:
        filtered_data = filtered_data[:args.sample_size]
    
    # Process all models using parallel execution or batch processing for vLLM
    responses = []
    
    temperature = args.temperature
    if args.method == "base":
        method = base
    elif args.method == "tool":
        method = tool_augmentation
    elif args.method == "correction":
        method = self_correction
    elif args.method == "rag":
        method = rag
    elif args.method == "consistency":
        method = self_consistency
        temperature = 0.6
    
    print("Using method: ", args.method)

    if model_type == "vllm":
        # For vLLM, use batch processing for better throughput
        print("Processing all requests in batch with vLLM...")
        from utils.vllm_batch_processor import process_batch_vllm
        responses = process_batch_vllm(
            filtered_data, 
            method,
            args.model, 
            args.max_tokens, 
            temperature,
            llm,
            sampling_params,
            is_multimodal
        )
    else:
        # For API-based models, use batch processing if enabled
        if args.use_batch and (is_openai_model or is_claude_model):
            print("Using batch processing for cost-effective evaluation...")

            try:
                # Special handling for tool augmentation method
                if args.method == "tool":
                    print("Using multi-round batch processing for tool augmentation...")
                    from methods.tool_augmentation import tool_augmentation_batch

                    if is_openai_model:
                        model_type_for_batch = "openai"
                    elif is_claude_model:
                        model_type_for_batch = "anthropic"
                    else:
                        raise ValueError(f"Unsupported model type for batch tool augmentation: {args.model}")

                    responses = tool_augmentation_batch(
                        filtered_data,
                        args.model,
                        args.max_tokens,
                        temperature,
                        model_type_for_batch,
                        is_multimodal
                    )
                else:
                    # Use regular batch processing for other methods
                    from utils.batch_processor import process_batch_openai, process_batch_claude

                    if is_openai_model:
                        responses = process_batch_openai(
                            filtered_data,
                            method,
                            args.model,
                            args.max_tokens,
                            temperature,
                            is_multimodal
                        )
                    elif is_claude_model:
                        responses = process_batch_claude(
                            filtered_data,
                            method,
                            args.model,
                            args.max_tokens,
                            temperature,
                            is_multimodal
                        )

                # Add image information to responses
                for response in responses:
                    if 'has_image' not in response:
                        response["has_image"] = bool((response.get("image_path") and response["image_path"].strip()) or
                                                   (response.get("image") and response["image"].strip()))

            except Exception as e:
                print(f"Batch processing failed: {e}")
                return  # Exit instead of falling back

        if not args.use_batch or not (is_openai_model or is_claude_model):
            # Use parallel processing for non-batch modes or unsupported models
            if not args.num_workers:
                if is_openai_model:
                    max_workers = 4
                elif is_claude_model:
                    max_workers = 2
                elif is_deepseek_model:
                    max_workers = 32
                else:
                    max_workers = 8
            else:
                max_workers = args.num_workers

            print(f"Processing requests in parallel with {max_workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a list of future tasks
                future_to_entry = {
                    executor.submit(
                        method,
                        entry,
                        args.model,
                        args.max_tokens,
                        temperature,
                        model_type,
                        llm,
                        sampling_params,
                        is_multimodal
                    ): entry for entry in filtered_data
                }

                # Process results as they complete with tqdm for progress
                for future in tqdm(concurrent.futures.as_completed(future_to_entry), total=len(filtered_data)):
                    result = future.result()
                    if result:
                        # Add information about whether this question has an image
                        result["has_image"] = bool((result.get("image_path") and result["image_path"].strip()) or
                                                  (result.get("image") and result["image"].strip()))
                        responses.append(result)

    # Now handle the LLM-based judging in parallel for free response questions
    # print(predictions)

    decisions = judge_responses(responses, max_workers=32, use_llm=args.llm_judge, use_rule=args.rule_judge)
    
    

    model_name = args.model.split('/')[-1].lower()
    # Create a results directory in the same directory as the input file
    results_dir = args.output_dir

    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    csv_output_path = os.path.join(results_dir, f"{model_name}_{args.method}_{timestamp}.csv")

    
    # Before writing to CSV, ensure all dictionaries have a consistent set of keys
    if decisions:
        # Get a union of all keys present in any dictionary
        all_keys = set()
        for decision in decisions:
            all_keys.update(decision.keys())
        
        # Ensure all dictionaries have all keys
        for decision in decisions:
            for key in all_keys:
                if key not in decision:
                    decision[key] = None  # Add missing key with None value
        
        # Now write to CSV
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_keys))
            writer.writeheader()
            writer.writerows(decisions)
        print(f"Saved responses and decisions to {csv_output_path}")
    else:
        # No decisions were produced, so no file will be created.
        # Provide a helpful message to the user instead of a misleading save path.
        num_responses = len(responses) if isinstance(responses, list) else 0
        print(
            "Warning: No decisions were generated; CSV not written. "
            f"responses={num_responses}, decisions=0. "
            "If you used batch mode, results may not have been retrieved correctly. "
            "Try running without --use_batch, verify API keys, or check batch result fetching."
        )
    


if __name__ == "__main__":
    main()
