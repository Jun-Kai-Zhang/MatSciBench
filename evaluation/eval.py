import argparse
import os
import re
import csv  # Add CSV import
from tqdm import tqdm

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

    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",

    "claude-3-7", 
    "claude-3-5", 
    
    "qwen2.5-vl-32b-instruct",

    "llama-4-maverick"
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
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers to use for parallel processing.")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Output directory for results.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate.")
    return parser.parse_args()





def main():
    args = parse_args()

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
    is_qwen_model = args.model.startswith("qwen") or args.model.startswith("qwq")
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
        # Initialize the vLLM engine for the specified model.
        print(f"Using vLLM model: {args.model}")
        llm = LLM(model=args.model)
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
    
    # Process all models using parallel execution
    # Set the number of workers based on model type - 4 for GPT, 8 for others
    # Set workers based on model type - 4 for OpenAI, 1 for vLLM, 8 for others
    responses = []
    
    if not args.num_workers:
        if is_openai_model:
            max_workers = 4
        elif is_claude_model:
            max_workers = 2
        elif is_deepseek_model:
            max_workers = 32
        elif model_type == "vllm":  # vLLM case (when model_type is not set)
            max_workers = 1
        else:
            max_workers = 8
    else:
        max_workers = args.num_workers
    
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

    decisions = judge_responses(responses, max_workers=8, use_llm=args.llm_judge)
    
    

    model_name = args.model.split('/')[-1].lower()
    # Create a results directory in the same directory as the input file
    results_dir = args.output_dir

    
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    csv_output_path = os.path.join(results_dir, f"{model_name}_{args.method}.csv")

    
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
    


if __name__ == "__main__":
    main()
