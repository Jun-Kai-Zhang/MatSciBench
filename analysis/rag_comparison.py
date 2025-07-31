"""RAG comparison script that analyzes error types across different methods for a single model.
Unlike error_categorization.py, this randomly samples questions rather than focusing
only on mistakes, and generates bar graphs grouped by error categories.

Run example:
    python analysis/rag_comparison.py \
        --model gemini-2.5-pro-preview-03-25 \
        --methods base,rag,tool,correction \
        --sample_size 50
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Project‑local imports
from analysis.prompts import (
    ERROR_CATEGORIES,
    CATEGORIZATION_SYSTEM_PROMPT,
    categorization_user_prompt,
)
from analysis.data_processor import load_evaluation_data
from utils import generate_with_api, extract_final_answer

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"  # fixed model used for error categorisation
MAX_TOKENS = 4096  # thinking budget / output limit for Gemini calls
CACHE_DIR = Path("results/analysis/rag_comparison/cache")  # Directory to store cached results
OUTPUT_DIR = Path("results/analysis/rag_comparison")  # Directory to store outputs

# Morandi color palette
MORANDI_COLORS = ['#E0D3C3','#5B7493']

# Simplified category names for plotting
SIMPLIFIED_CATEGORIES = {
    1: "Comprehension",
    2: "Knowledge",
    3: "Strategy",
    4: "Calculation", 
    5: "Hallucination",
    6: "Code Conversion"
}

# ────────────────────────────────────────────────────────────────────────────────
# Gemini call wrapper via shared utils/api.py
# ────────────────────────────────────────────────────────────────────────────────

def _categorize_error_with_gemini(
    question: str,
    correct_solution: str,
    model_solution: str,
    temperature: float = 0.0,
) -> Optional[int]:
    """Call Gemini through the project‑wide helper and extract the category id."""
    conversation = [
        {"role": "system", "content": CATEGORIZATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": categorization_user_prompt(question, correct_solution, model_solution),
        },
    ]

    try:
        full_output, _ = generate_with_api(
            model_type="gemini",
            model=GEMINI_MODEL,
            conversation=conversation,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            image_paths=None,
        )
        
        # Handle case where API returns None
        if full_output is None:
            print(f"Warning: API returned None for categorization")
            return None
            
        text = full_output.strip()
        
        # Use the imported extract_final_answer function to get the category
        extracted_answer = extract_final_answer(text)
        if extracted_answer and extracted_answer.isdigit():
            category_idx = int(extracted_answer)
            if 1 <= category_idx <= 6:
                return category_idx
        
        # Fallback to searching for a standalone digit
        for tok in text.split():
            if tok.isdigit() and 1 <= int(tok) <= 6:
                return int(tok)
                
    except Exception as e:
        print(f"Error in API call to categorize error: {e}")
        return None
            
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Row‑level processor
# ────────────────────────────────────────────────────────────────────────────────

def _process_row(row: pd.Series) -> Dict[str, Any]:
    category_idx = _categorize_error_with_gemini(
        row.get("question", ""),
        row.get("correct_solution", ""),
        row.get("full_output", ""),
    )
    return {
        "qid": row.get("qid"),
        "method": row.get("method"),
        "is_correct": row.get("is_correct"),
        "category_idx": category_idx,
        "category": ERROR_CATEGORIES.get(category_idx, "Unclassified"),
        "question": row.get("question", ""),
        "correct_solution": row.get("correct_solution", ""),
        "full_output": row.get("full_output", ""),
    }

# ────────────────────────────────────────────────────────────────────────────────
# Statistics / plotting helpers
# ────────────────────────────────────────────────────────────────────────────────

def _plot_category_by_method_bars(results_df: pd.DataFrame, model_name: str, out_dir: Path, method_total_counts: Dict[str, int]) -> None:
    """Generate bar graph showing error distributions by method, grouped by error category."""
    # Filter to include only valid categorizations
    valid_df = results_df[results_df["category_idx"].notna()]
    
    # Filter out "Error Conversion Code" category
    valid_df = valid_df[valid_df["category"] != "Error Conversion Code"]
    
    if valid_df.empty:
        print("No valid categorizations to plot")
        return
    
    # Get all methods in the data
    methods = valid_df["method"].unique()
    
    # Group by category and method to get counts
    category_method_counts = valid_df.groupby(["category_idx", "method"]).size().reset_index(name="count")
    
    # Pivot the data to have categories as rows and methods as columns
    pivot_df = category_method_counts.pivot(index="category_idx", columns="method", values="count").fillna(0)
    
    # Ensure all methods have a column
    for method in methods:
        if method not in pivot_df.columns:
            pivot_df[method] = 0
    
    # Set the width of each bar group
    width = 0.8 / len(methods)
    
    # Set the style similar to multimodal_drop.py
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Plot each method as a group of bars - no borders around bars
    for i, method in enumerate(methods):
        x = np.arange(len(pivot_df.index))
        offset = (i - len(methods)/2 + 0.5) * width
        color = MORANDI_COLORS[i % len(MORANDI_COLORS)]
        plt.bar(x + offset, pivot_df[method], width, label=method, color=color)
    
    # Customize the plot
    plt.xlabel("Error Categories", fontsize=26)
    plt.ylabel("Error Count", fontsize=26)
    plt.xticks(fontsize=26)
    
    # Use simplified category names for x-axis labels
    category_indices = [idx if not pd.isna(idx) else 0 for idx in pivot_df.index]
    
    # Map the indices to simplified names
    simplified_labels = []
    for idx in category_indices:
        if pd.isna(idx) or idx not in SIMPLIFIED_CATEGORIES:
            simplified_labels.append("Other")
        else:
            simplified_labels.append(SIMPLIFIED_CATEGORIES[idx])
    
    plt.xticks(np.arange(len(pivot_df.index)), simplified_labels)
    
    # Create a proper legend with a title
    plt.legend(title="Methods", title_fontsize=26, fontsize=26)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"rag_comparison_{model_name}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# Caching utilities
# ────────────────────────────────────────────────────────────────────────────────

def _get_cache_path(model: str) -> Path:
    """Generate a cache file path for a specific model."""
    # Create sanitized filenames (remove special characters)
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    return CACHE_DIR / f"cache_{safe_model}.csv"

def _load_cached_results(model: str) -> Optional[pd.DataFrame]:
    """Attempt to load cached results for a model."""
    cache_path = _get_cache_path(model)
    if cache_path.exists():
        try:
            print(f"Loading cached results from {cache_path}")
            return pd.read_csv(cache_path)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None

def _save_to_cache(df: pd.DataFrame, model: str) -> None:
    """Save categorization results to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(model)
    try:
        df.to_csv(cache_path, index=False)
        print(f"Saved results to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving to cache: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# CLI + main logic
# ────────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare error types across different methods for a single model.")
    p.add_argument("--model", default="deepseek-chat", help="Model name to analyse (e.g. gemini-2.5-pro-preview)")
    p.add_argument("--methods", default="base,rag", help="Comma-separated list of methods to compare")
    p.add_argument("--sample_size", default="50", help="How many questions to randomly sample (use 'all' for all questions)")
    p.add_argument("--force_recalculate", action="store_true", help="Force recalculation even if cache exists")
    p.add_argument("--clean_cache", action="store_true", help="Clean cache before running")
    return p.parse_args()

def _clean_model_cache(model: str) -> None:
    """Remove cache files for a specific model."""
    if not CACHE_DIR.exists():
        return
        
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    cache_file = CACHE_DIR / f"cache_{safe_model}.csv"
    
    if cache_file.exists():
        try:
            cache_file.unlink()
            print(f"Removed cache file for model {model}")
        except Exception as e:
            print(f"Error removing {cache_file}: {e}")

def main() -> None:
    args = _parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle cache cleaning if requested
    if args.clean_cache:
        _clean_model_cache(args.model)

    # Parse methods list
    methods = [m.strip() for m in args.methods.split(",")]
    
    # Handle sample size (allow 'all' option)
    use_all_questions = False
    try:
        if args.sample_size.lower() == 'all':
            use_all_questions = True
            sample_size = None
        else:
            sample_size = int(args.sample_size)
    except (ValueError, AttributeError):
        # Default to 50 if there's any issue with the input
        sample_size = 50
    
    # 1. Load full evaluation dataframe
    print("Loading evaluation data...")
    eval_df, _ = load_evaluation_data(base_dir="results")
    if eval_df.empty:
        print("No evaluation data available.")
        return

    # 2. Filter data for the specified model
    model_df = eval_df[eval_df["model"] == args.model]
    if model_df.empty:
        print(f"No data found for model {args.model}")
        return
    
    # 3. Find questions that are common across all methods
    print(f"Finding questions common across all methods: {', '.join(methods)}")
    method_qids = {}
    for method in methods:
        method_df = model_df[model_df["method"] == method]
        if method_df.empty:
            print(f"No data found for method {method}")
            return
        method_qids[method] = set(method_df["qid"].unique())
        print(f"Method {method} has {len(method_qids[method])} questions")
    
    # Find the intersection of all QID sets
    common_qids = set.intersection(*method_qids.values())
    print(f"Found {len(common_qids)} questions common across all methods")
    
    if not common_qids:
        print("No common questions found across all methods. Cannot proceed.")
        return
    
    # Keep track of total question counts per method for normalization
    method_total_counts = {}
    for method in methods:
        method_total_counts[method] = len(common_qids)
    
    # Check if we have cached results
    cached_results = None
    if not args.force_recalculate:
        cached_results = _load_cached_results(args.model)
    
    # Filter cached results to only include common questions if cache exists
    if cached_results is not None:
        cached_results = cached_results[cached_results["qid"].isin(common_qids)]
        cached_qids = set(cached_results["qid"].unique())
        print(f"Found {len(cached_qids)} cached common questions")
    
    # Determine if we need to process new questions
    if cached_results is not None and not use_all_questions and len(cached_qids) >= sample_size:
        # For sample_size mode: if we have enough cached results, just use those
        sampled_qids = list(cached_qids)
        if len(sampled_qids) > sample_size:
            sampled_qids = np.random.choice(sampled_qids, size=sample_size, replace=False)
        results_df = cached_results[cached_results["qid"].isin(sampled_qids)]
        print(f"Using {len(results_df)} cached results")
    else:
        # Either processing all questions or we don't have enough in cache
        # 4. Sample from common questions or use all
        if use_all_questions:
            print(f"Using all {len(common_qids)} common questions")
            sampled_qids = list(common_qids)
        else:
            sample_size_to_use = min(sample_size, len(common_qids))
            print(f"Randomly sampling {sample_size_to_use} out of {len(common_qids)} common questions")
            sampled_qids = np.random.choice(list(common_qids), size=sample_size_to_use, replace=False)
        
        # 5. For each method, get data for these sampled questions
        all_results = []
        already_processed_qids = set()
        
        # If we have cached results, track already processed qids
        if cached_results is not None and not cached_results.empty:
            already_processed_qids = set(cached_results["qid"].tolist())
            all_results.append(cached_results)
        
        # Filter out qids that have already been processed
        new_qids = [qid for qid in sampled_qids if qid not in already_processed_qids]
        
        if not new_qids:
            print("No new questions to process.")
            results_df = cached_results
        else:
            print(f"Processing {len(new_qids)} new questions...")
            
            # Process each method
            for method in methods:
                print(f"\nProcessing method: {method}")
                
                # Get data for this method and the sampled qids
                method_df = model_df[(model_df["method"] == method) & (model_df["qid"].isin(new_qids))]
                
                if method_df.empty:
                    print(f"No data for method {method}")
                    continue
                
                # Filter for incorrect answers with valid correct solutions
                wrong_df = method_df[
                    (~method_df["is_correct"]) & 
                    method_df["correct_solution"].notna() & 
                    (method_df["correct_solution"] != "")
                ]
                
                if wrong_df.empty:
                    print(f"No incorrect answers with valid correct solutions for method {method}")
                    continue
                
                print(f"Found {len(wrong_df)} incorrect answers to analyze for method {method}")
                
                # Add method column to make sure it's preserved
                wrong_df["method"] = method
                
                # Process the wrong answers to categorize errors
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
                    futs = [ex.submit(_process_row, row) for _, row in wrong_df.iterrows()]
                    for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
                        results.append(f.result())
                
                if results:
                    method_results_df = pd.DataFrame(results)
                    all_results.append(method_results_df)
            
            # Combine all results
            if all_results:
                results_df = pd.concat(all_results, ignore_index=True)
                
                # Save to cache
                _save_to_cache(results_df, args.model)
            else:
                print("No results to analyze.")
                return
    
    # Print method error counts with category breakdown
    print("\nMethod error counts with category breakdown:")
    valid_df = results_df[results_df["category_idx"].notna()]
    if not valid_df.empty:
        methods = valid_df["method"].unique()
        for method in methods:
            method_df = valid_df[valid_df["method"] == method]
            total_count = len(method_df)
            print(f"\n{method}: {total_count} total errors")
            
            # Get category counts for this method
            category_counts = method_df["category"].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count}")
        
        print(f"\nTotal categorized errors across all methods: {len(valid_df)}")
    else:
        print("  No valid categorizations found")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    _plot_category_by_method_bars(results_df, args.model, OUTPUT_DIR, method_total_counts)
    print(f"Saved plots to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
