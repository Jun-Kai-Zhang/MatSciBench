"""Error categorization script that loads evaluation data with data_processor.py
and classifies the error type for a chosen model–method pair using **Gemini 2.0 Pro**
through the shared `utils/api.generate_with_api` helper.

Run example:
    python analysis/error_categorization_by_model.py \
        --datasets_dir datasets \
        --model gemini-2.5-pro-preview-03-25 \
        --method vanilla \
        --sample_size 50
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Ensure consistent color cycles
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Project‑local imports
from analysis.prompts import (
    ERROR_CATEGORIES,
    CATEGORIZATION_SYSTEM_PROMPT,
    categorization_user_prompt,
)
from analysis.data_processor import load_evaluation_data
from utils import generate_with_api, extract_final_answer  # <‑‑ unified API wrapper

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"  # fixed model used for error categorisation
MAX_TOKENS   = 1024                # thinking budget / output limit for Gemini calls
CACHE_DIR    = Path("analysis/model_error_analysis/cache")  # Directory to store cached results

# Model name abbreviations for plotting
MODEL_ABBREVIATIONS = {
    'claude-3-7-sonnet-20250219': 'Claude-3-7-Sonnet',
    'deepseek-chat': 'DeepSeek-V3',
    'gemini-2.0-flash': 'Gemini-2.0-Flash',
    'gpt-4.1-2025-04-14': 'GPT-4.1',
    'llama-4-maverick-17b-128e-instruct-fp8': 'Llama-4-Maverick',
    'deepseek-reasoner': 'DeepSeek-R1',
    'gemini-2.5-pro-preview-05-06': 'Gemini-2.5-Pro',
    'o4-mini-2025-04-16': 'o4-mini',
    'qwen3-235b-a22b': 'Qwen-3-235B'
}

# Simplified category names for radar charts
SIMPLIFIED_CATEGORIES = {
    1: "Comprehension",
    2: "Knowledge",
    3: "Strategy",
    4: "Calculation",
    5: "Hallucination",
    6: "Code Conversion"
}

# Abbreviations for error categories
CATEGORY_ABBREVIATIONS = {
    1: "Comp.",    # Comprehension
    2: "Know.",    # Knowledge
    3: "Strat.",   # Strategy
    4: "Calc.",    # Calculation
    5: "Hall.",    # Hallucination
    6: "Code"     # Code Conversion
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

    full_output, _ = generate_with_api(
        model_type="gemini",
        model=GEMINI_MODEL,
        conversation=conversation,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        image_paths=None,
    )

    text = full_output.strip()
    # print("text: ", text)
    
    # Use the imported extract_final_answer function to get the category
    extracted_answer = extract_final_answer(text)
    if extracted_answer and extracted_answer.isdigit():
        category_idx = int(extracted_answer)
        if 1 <= category_idx <= 5:
            return category_idx
    
    # Fallback to searching for a standalone digit
    for tok in text.split():
        # Skip tokens that contain non-digit characters (except for decimal points)
        if not all(c.isdigit() or c == '.' for c in tok):
            continue
        try:
            num = int(float(tok))
            if 1 <= num <= 5:
                return num
        except (ValueError, TypeError):
            continue
            
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
        "is_correct": row.get("is_correct"),
        "category_idx": category_idx,
        "category": ERROR_CATEGORIES.get(category_idx, "Unclassified"),
    }

# ────────────────────────────────────────────────────────────────────────────────
# Combined radar chart
# ────────────────────────────────────────────────────────────────────────────────

def _plot_multi_model_comparison(all_results: Dict[str, Dict[str, pd.DataFrame]], out_dir: Path, total_samples: int) -> None:
    """Create a row of radar charts comparing error categories across multiple models."""
    # Clean up any existing figures
    plt.close("all")
    
    # Prepare color palette with more distinct, vibrant colors for better contrast
    vibrant_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22'   # Olive
    ]
    
    # Fixed color mapping for methods to ensure consistency
    method_colors = {
        'base': vibrant_colors[0],      # Blue
        'tool': vibrant_colors[1],      # Orange
        'correction': vibrant_colors[2]  # Green
    }
    
    # Get all unique methods across all models and sort them for consistency
    all_methods = set()
    for model_results in all_results.values():
        all_methods.update(model_results.keys())
    all_methods = sorted(all_methods)  # Sort methods for consistent ordering
    
    # Sort models for consistent ordering
    sorted_models = sorted(all_results.keys())
    n_models = len(sorted_models)
    
    # Create a figure with subplots in a single row
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6), subplot_kw={'polar': True})
    
    # If only one model, axes won't be an array
    if n_models == 1:
        axes = [axes]
    
    # Prepare data - exclude "Other" category (index 6)
    categories = [cat for idx, cat in sorted(ERROR_CATEGORIES.items()) if idx < 6]
    # Use simplified category names for display
    simplified_names = [CATEGORY_ABBREVIATIONS.get(i+1, cat) for i, cat in enumerate(categories)]
    
    # Close the loop for the radar chart
    simplified_names_closed = simplified_names + [simplified_names[0]]
    
    # Calculate angle for each category
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles_closed = angles + [angles[0]]
    
    # Keep track of global max value for consistent scaling
    global_max_value = 0
    
    # First pass: compute max value across all models for consistent scaling
    for model_name in sorted_models:
        method_results = all_results[model_name]
        for method in sorted(method_results.keys()):
            df = method_results[method]
            valid_df = df[df["category_idx"].notna()]
            if len(valid_df) == 0:
                continue
                
            # Sort categories by index for consistent ordering
            error_categories_list = sorted(ERROR_CATEGORIES.items(), key=lambda x: x[0])
            error_categories_values = [v for _, v in error_categories_list]
            
            # Get counts with consistent ordering
            counts = valid_df["category"].value_counts().reindex(error_categories_values, fill_value=0)
            values = [counts[cat] / total_samples for cat in categories]
            global_max_value = max(global_max_value, max(values) if values else 0)
    
    # Add 20% margin to max value for better visualization
    y_limit = min(1.0, global_max_value * 1.05 if global_max_value > 0 else 0.1)
    
    # Second pass: plot radar charts
    for i, model_name in enumerate(sorted_models):
        method_results = all_results[model_name]
        ax = axes[i]
        ax.set_facecolor('white')
        
        # Setup this subplot
        ax.set_ylim(0, y_limit)
        
        # Set category labels with better positioning
        ax.set_xticks(angles)
        label_position = y_limit
        
        for angle, label in zip(angles, simplified_names):
            ha = "center"
            if angle == 0:
                ha = "left"
            elif 0 < angle < 0.5 * np.pi:
                ha = "left"
            elif 0.5 * np.pi < angle < 1.5 * np.pi:
                ha = "right"
            elif angle > 1.5 * np.pi:
                ha = "left"
                
            ax.text(angle, label_position, label, 
                   horizontalalignment=ha, verticalalignment="center", 
                   size=30)  # Removed fontweight='bold'
        
        # Hide default category labels and set title using abbreviated model name
        ax.set_xticklabels([])
        # Get abbreviated model name or use original if not found
        plot_title = MODEL_ABBREVIATIONS.get(model_name, model_name)
        ax.set_title(plot_title, size=30, pad=20)  # Removed fontweight='bold'
        
        # Customize grid lines
        ax.grid(True, color='gray', alpha=0.3, linestyle='--')
        
        # Set y-axis ticks but hide labels
        num_ticks = 5
        yticks = [y_limit * i/num_ticks for i in range(1, num_ticks+1)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(["" for _ in yticks])
        
        # Add a circular border
        circle = plt.Circle((0, 0), y_limit, transform=ax.transData._b, fill=False, 
                           edgecolor='black', linewidth=1.5, alpha=0.5)  # Darker border
        ax.add_artist(circle)
        
        # Enhance the spokes (lines from center to edge)
        for angle in angles:
            ax.plot([angle, angle], [0, y_limit], color='gray', alpha=0.3, linewidth=0.8)
        
        # Plot each method in this model - using sorted order for consistency
        for method in sorted(method_results.keys()):
            df = method_results[method]
            valid_df = df[df["category_idx"].notna()]
            if len(valid_df) == 0:
                continue
                
            # Sort categories by index for consistent ordering
            error_categories_list = sorted(ERROR_CATEGORIES.items(), key=lambda x: x[0])
            error_categories_values = [v for _, v in error_categories_list]
            
            # Get counts with consistent ordering
            counts = valid_df["category"].value_counts().reindex(error_categories_values, fill_value=0)
            
            # Normalize values by total samples
            values = [counts[cat] / total_samples for cat in categories]
            values_closed = values + [values[0]]
            
            # Use consistent colors for methods across models
            # First try the fixed mapping, then fall back to a deterministic color based on method name
            if method in method_colors:
                color = method_colors[method]
            else:
                # Use a deterministic method to assign colors
                method_hash = sum(ord(c) for c in method) % len(vibrant_colors)
                color = vibrant_colors[method_hash]
            
            # Plot the data with higher contrast
            ax.plot(angles_closed, values_closed, linewidth=3.5, linestyle='solid', color=color, label=f"{method}")
            ax.fill(angles_closed, values_closed, alpha=0.4, color=color)  # Slightly higher alpha for better visibility
    
    # Create a single legend for all plots - sort labels for consistency
    handles, labels = axes[0].get_legend_handles_labels()
    # Sort handles and labels together by label
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: x[1])
    handles = [h for h, _ in sorted_pairs]
    labels = [l for _, l in sorted_pairs]
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=30, title_fontsize=30,
               title="Methods", ncol=min(5, len(all_methods)))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust to make room for legend
    
    # Save figure with a hash of the data to see if it changes
    output_path = out_dir / "error_categorization_multi_model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # If the file exists, compute its checksum
    if output_path.exists():
        plot_checksum = hashlib.md5(output_path.read_bytes()).hexdigest()
        print(f"Plot file checksum: {plot_checksum}")
    
    plt.close("all")  # Ensure figure is closed

# ────────────────────────────────────────────────────────────────────────────────
# Caching utilities
# ────────────────────────────────────────────────────────────────────────────────

def _get_cache_path(model: str, method: str) -> Path:
    """Generate a cache file path for a specific model-method pair."""
    # Create sanitized filenames (remove special characters)
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    safe_method = "".join(c if c.isalnum() else "_" for c in method)
    return CACHE_DIR / f"cache_{safe_model}_{safe_method}.csv"

def _load_cached_results(model: str, method: str) -> Optional[pd.DataFrame]:
    """Attempt to load cached results for a model-method pair."""
    cache_path = _get_cache_path(model, method)
    if cache_path.exists():
        try:
            # Compute a hash of the cache file to detect changes
            checksum = hashlib.md5(cache_path.read_bytes()).hexdigest()
            print(f"Loading cached results from {cache_path}")
            print(f"Cache file checksum: {checksum}")
            
            # Load the cached data and sort it consistently
            df = pd.read_csv(cache_path)
            if not df.empty:
                # Sort by qid to ensure consistent ordering
                df = df.sort_values(by="qid")
            return df
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None

def _save_to_cache(df: pd.DataFrame, model: str, method: str) -> None:
    """Save categorization results to cache, appending to existing cache if present."""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(model, method)
    
    try:
        # Read existing cache if it exists
        if cache_path.exists():
            # Compute hash of existing cache file
            old_checksum = hashlib.md5(cache_path.read_bytes()).hexdigest()
            print(f"Existing cache file checksum: {old_checksum}")
            
            existing_df = pd.read_csv(cache_path)
            # Combine existing and new results
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on qid, keeping the most recent entry
            combined_df = combined_df.drop_duplicates(subset=['qid'], keep='last')
            # Sort by qid for consistent ordering
            combined_df = combined_df.sort_values(by="qid")
            # Save combined results
            combined_df.to_csv(cache_path, index=False)
            
            # Compute hash of new cache file
            new_checksum = hashlib.md5(cache_path.read_bytes()).hexdigest()
            print(f"Updated cache file checksum: {new_checksum}")
            print(f"Appended and saved results to cache: {cache_path}")
        else:
            # If no existing cache, just save the new results
            # Sort by qid for consistent ordering
            sorted_df = df.sort_values(by="qid")
            sorted_df.to_csv(cache_path, index=False)
            
            # Compute hash of new cache file
            new_checksum = hashlib.md5(cache_path.read_bytes()).hexdigest()
            print(f"New cache file checksum: {new_checksum}")
            print(f"Created new cache file: {cache_path}")
    except Exception as e:
        print(f"Error saving to cache: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# CLI + main logic
# ────────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Categorise errors for a model–method pair.")
    p.add_argument("--model", default="gemini-2.0-flash", help="Model name to analyse (e.g. gemini-2.5-…)")
    p.add_argument("--models", default=["gemini-2.0-flash", "claude-3-7-sonnet-20250219","deepseek-chat","gpt-4.1-2025-04-14", "llama-4-maverick-17b-128e-instruct-fp8"], help="List of models to analyze (overrides --model)")
    p.add_argument("--method", default="base", help="Method name to analyse (vanilla, tool, …)")
    p.add_argument("--methods", default=["base", "correction","tool"], help="List of methods to compare (overrides --method)")
    p.add_argument("--sample_size", type=str, default="100", help="How many questions to sample (use 'all' for all common questions)")
    p.add_argument("--output_dir", default="analysis/model_error_analysis", help="Where to store outputs")
    p.add_argument("--force_recalculate", action="store_true", help="Force recalculation even if cache exists")
    p.add_argument("--clean_cache", action="store_true", help="Clean cache before running")
    p.add_argument("--use_cache_only", action="store_true", help="Only use cached results, skip any new API calls")
    return p.parse_args()

def _clean_all_cache() -> None:
    """Remove all cache files."""
    if not CACHE_DIR.exists():
        return
        
    count = 0
    for cache_file in CACHE_DIR.glob("cache_*.csv"):
        try:
            cache_file.unlink()
            count += 1
        except Exception as e:
            print(f"Error removing {cache_file}: {e}")
    
    print(f"Removed {count} cache files")

def _clean_model_cache(model: str) -> None:
    """Remove cache files for a specific model."""
    if not CACHE_DIR.exists():
        return
        
    safe_model = "".join(c if c.isalnum() else "_" for c in model)
    count = 0
    for cache_file in CACHE_DIR.glob(f"cache_{safe_model}_*.csv"):
        try:
            cache_file.unlink()
            count += 1
        except Exception as e:
            print(f"Error removing {cache_file}: {e}")
    
    print(f"Removed {count} cache files for model {model}")

def _find_common_questions(eval_df: pd.DataFrame, models: List[str], methods: List[str]) -> Set[str]:
    """Find questions that are common across all models and methods and have correct solutions."""
    # Get all questions with valid correct solutions
    valid_questions = eval_df[eval_df["correct_solution"].notna() & (eval_df["correct_solution"] != "")]["qid"].unique()
    print(f"Total questions with valid correct solutions: {len(valid_questions)}")
    # Print all unique question IDs
    all_qids = eval_df["qid"].unique()
    print(f"Total unique questions in dataset: {len(all_qids)}")
    
    # Print first 10 question IDs as a sample
    print("Sample of question IDs:")
    for qid in list(all_qids)[:10]:
        print(f"  - {qid}")
    # Find questions common to all model-method pairs
    common_qids = set(valid_questions)
    
    # Check for each model and method if it has answers for these questions
    for model in models:
        for method in methods:
            # Find questions this model-method pair has answers for
            mask = (eval_df["model"] == model) & (eval_df["method"] == method)
            pair_df = eval_df.loc[mask]
            
            if pair_df.empty:
                print(f"WARNING: No data for model={model}, method={method}")
                return set()  # If any pair has no data, no common questions exist
                
            model_method_qids = set(pair_df["qid"].unique())
            print(f"Model={model}, Method={method}: {len(model_method_qids)} questions")
            
            # Keep only common questions
            common_qids = common_qids.intersection(model_method_qids)
    
    print(f"Questions common to ALL model-method pairs: {len(common_qids)}")
    return common_qids

def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing figures
    plt.close("all")
    
    # Handle cache cleaning if requested
    if args.clean_cache:
        if args.model and args.model != "all" and not args.models:
            _clean_model_cache(args.model)
        elif args.models:
            for model in args.models:
                _clean_model_cache(model)
        else:
            _clean_all_cache()

    # 1. Load full evaluation dataframe
    print("Loading evaluation data …")
    eval_df, difficulty_df = load_evaluation_data(base_dir="results")
    if eval_df.empty:
        print("No evaluation data available.")
        return

    # Determine which methods to analyze
    methods = args.methods if args.methods else [args.method]
    
    # Determine which models to analyze
    models = args.models if args.models else [args.model]
    
    # Find questions common to all models with valid correct solutions
    print("Finding common questions across all models and methods...")
    common_qids = _find_common_questions(eval_df, models, methods)
    
    if not common_qids:
        print("No common questions found across all specified models and methods.")
        return
    
    print(f"Found {len(common_qids)} questions common across all models and methods.")
    
    # Sample the questions if needed
    if args.sample_size.lower() == 'all':
        sampled_qids = sorted(list(common_qids))  # Sort for consistency
        print(f"Using all {len(sampled_qids)} common questions.")
    else:
        try:
            sample_size = int(args.sample_size)
            if sample_size >= len(common_qids):
                sampled_qids = sorted(list(common_qids))  # Sort for consistency
                print(f"Sample size {sample_size} is larger than available questions. Using all {len(sampled_qids)} common questions.")
            else:
                # Use a fixed random seed for consistent sampling
                rng = np.random.RandomState(42)
                sampled_qids = rng.choice(sorted(list(common_qids)), size=sample_size, replace=False).tolist()
                sampled_qids = sorted(sampled_qids)  # Sort for consistency
                print(f"Sampled {len(sampled_qids)} questions from {len(common_qids)} common questions.")
        except ValueError:
            print(f"Invalid sample size: {args.sample_size}. Using 50 questions.")
            rng = np.random.RandomState(42)
            sampled_qids = rng.choice(sorted(list(common_qids)), size=min(50, len(common_qids)), replace=False).tolist()
            sampled_qids = sorted(sampled_qids)  # Sort for consistency
    
    # Filter evaluation data to include only the sampled questions
    sampled_eval_df = eval_df[eval_df["qid"].isin(sampled_qids)]
    
    # Total number of sampled questions - used for normalization
    total_sampled = len(sampled_qids)
    
    # Dictionary to store results by model and method
    all_results = {}
    
    # Process models in sorted order for consistency
    for model in sorted(models):
        print(f"\nProcessing model: {model}")
        method_results = {}
        
        # Process methods in sorted order for consistency
        for method in sorted(methods):
            print(f"\nAnalyzing method: {method}")
            
            # Filter to get only this model-method pair for the sampled questions
            mask = (sampled_eval_df["model"] == model) & (sampled_eval_df["method"] == method)
            pair_df = sampled_eval_df.loc[mask]

            if pair_df.empty:
                print(f"No records for model {model} with method {method} in the sampled questions.")
                continue

            # Get incorrect answers for analysis
            wrong_df = pair_df[pair_df["is_correct"] == False]  # noqa: E712
            if wrong_df.empty:
                print("All answers correct for this model-method pair; nothing to categorise.")
                # Still add an empty result to keep track
                method_results[method] = pd.DataFrame(columns=["qid", "is_correct", "category_idx", "category"])
                continue

            # Check cache
            cached_results = None
            if not args.force_recalculate:
                cached_results = _load_cached_results(model, method)
                
                # Filter cached results to only include sampled questions
                if cached_results is not None and not cached_results.empty:
                    cached_results = cached_results[cached_results["qid"].isin(sampled_qids)]
                    # Sort for consistency
                    cached_results = cached_results.sort_values(by="qid")
                    print(f"Found {len(cached_results)} cached results for sampled questions")
            
            # If use_cache_only is set, skip processing new samples
            if args.use_cache_only:
                if cached_results is not None and not cached_results.empty:
                    method_results[method] = cached_results
                    print(f"Using {len(cached_results)} cached results (cache-only mode)")
                else:
                    print(f"No cached results found for model {model} with method {method} (cache-only mode)")
                    method_results[method] = pd.DataFrame(columns=["qid", "is_correct", "category_idx", "category"])
                continue
            
            # Setup for processing wrong answers
            to_process_df = wrong_df.copy()
            
            # Remove already processed questions from to_process_df
            if cached_results is not None and not cached_results.empty and not args.force_recalculate:
                already_processed_qids = set(cached_results["qid"].tolist())
                to_process_df = to_process_df[~to_process_df["qid"].isin(already_processed_qids)]
            
            if len(to_process_df) == 0:
                print("No additional samples to process.")
                method_results[method] = cached_results if cached_results is not None else pd.DataFrame()
                continue
                
            print(f"Analysing {len(to_process_df)} incorrect answers...")

            # Call Gemini concurrently for the additional samples
            results: list[Dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(_process_row, row) for _, row in to_process_df.iterrows()]
                for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
                    results.append(f.result())

            new_results_df = pd.DataFrame(results)
            
            # Combine with cached results if available
            if cached_results is not None and not cached_results.empty and not args.force_recalculate:
                res_df = pd.concat([cached_results, new_results_df], ignore_index=True)
                # Sort for consistency
                res_df = res_df.sort_values(by="qid")
                print(f"Combined {len(cached_results)} cached results with {len(new_results_df)} new results.")
            else:
                res_df = new_results_df
                # Sort for consistency
                if not res_df.empty:
                    res_df = res_df.sort_values(by="qid")
                
            # Update cache with results
            _save_to_cache(res_df, model, method)

            # Store results for combined plotting
            method_results[method] = res_df
        
        # Store results for this model
        if method_results:
            all_results[model] = method_results
    
    # Generate the combined radar chart across all models
    if len(all_results) > 0:
        # Print data structure hash for debugging
        all_models = sorted(list(all_results.keys()))
        print(f"Models to plot ({len(all_models)}): {', '.join(all_models)}")
        
        for model_name in all_models:
            method_results = all_results[model_name]
            methods = sorted(list(method_results.keys()))
            print(f"  Model {model_name} has {len(methods)} methods: {', '.join(methods)}")
            
            for method_name in methods:
                df = method_results[method_name]
                if not df.empty:
                    # Create a deterministic representation of the dataframe for hashing
                    df_hash = hashlib.md5(df.sort_values("qid").to_json().encode()).hexdigest()
                    print(f"    Method {method_name} has {len(df)} records, hash: {df_hash}")
        
        _plot_multi_model_comparison(all_results, out_dir, total_sampled)
        print(f"Saved multi-model comparison → {out_dir / 'error_categorization_multi_model_comparison.png'}")
        
        # Close all figures again to be sure
        plt.close("all")


if __name__ == "__main__":
    main()