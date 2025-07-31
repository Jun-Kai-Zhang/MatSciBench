import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
import pandas as pd
from scipy.optimize import minimize

def analyze_tokens(base_dir="results", include_datasets=None, include_models=None, include_methods=None):
    """Analyze token usage vs accuracy performance for non-image questions only
    
    Args:
        base_dir (str): Base directory containing dataset folders
        include_datasets (list): List of dataset names to include in analysis
        include_models (list): List of model name substrings to include in analysis. 
                             If None, includes all models. Partial matches are supported.
        include_methods (list): List of method names to include in analysis.
                              If None, includes all methods.
    """
    # Import here to keep module independent
    from analysis.data_processor import load_evaluation_data
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define model abbreviations
    model_abbreviations = {
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
    
    # Load data
    results_df, _ = load_evaluation_data()
    
    # Setup output directories
    results_output_dir = os.path.join("analysis", "results")
    plots_dir = os.path.join(results_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Filter for non-image questions only (assuming has_image column exists, otherwise create it)
    if 'has_image' not in results_df.columns:
        # Assume questions without image_path are non-image questions
        results_df['has_image'] = results_df.apply(
            lambda row: False if pd.isna(row.get('image_path', np.nan)) else True, 
            axis=1
        )
    
    non_image_df = results_df[results_df['has_image'] == False]
    
    # Filter for specified models if provided
    if include_models is not None:
        # Create a mask for each model substring
        model_masks = [non_image_df['model'].str.contains(model, case=False) for model in include_models]
        # Combine masks with OR operation
        combined_mask = model_masks[0]
        for mask in model_masks[1:]:
            combined_mask = combined_mask | mask
        non_image_df = non_image_df[combined_mask]
    
    # Filter for specified methods if provided
    if include_methods is not None:
        non_image_df = non_image_df[non_image_df['method'].isin(include_methods)]
    
    # Create summary for non-image questions
    # Renaming new_token_nums to avg_tokens for consistency
    if 'new_token_nums' in non_image_df.columns and 'avg_tokens' not in non_image_df.columns:
        non_image_df['avg_tokens'] = non_image_df['new_token_nums']
    
    # Count total and correct questions per model and method
    non_image_summary = non_image_df.groupby(['model', 'method']).agg({
        'is_correct': ['count', 'sum'],
        'avg_tokens': 'mean'
    })
    
    # Flatten the multi-index columns
    non_image_summary.columns = ['total_questions', 'correct_answers', 'avg_tokens']
    non_image_summary = non_image_summary.reset_index()
    
    non_image_summary['overall_accuracy'] = (
        non_image_summary['correct_answers'] / non_image_summary['total_questions'] * 100
    )
    
    # Get ranking for easier plotting
    ranking = non_image_summary.sort_values('overall_accuracy', ascending=False)
    
    # Create figure with a specific size and DPI
    plt.figure(figsize=(12, 8), dpi=200)
    
    # Get unique methods and models
    methods = ranking['method'].unique()
    models = ranking['model'].unique()
    
    # Create distinct markers for different methods
    method_markers = {}
    marker_options = ['o', '^', 's', 'D', 'P', '*', 'X', 'p']  # add more if needed
    for i, method in enumerate(methods):
        method_markers[method] = marker_options[i % len(marker_options)]
    
    # Use seaborn's tab20 color palette
    model_palette = sns.color_palette("tab20", len(models))
    model_colors = dict(zip(models, model_palette))
    
    # Plot each method with appropriate markers but color by model
    for method in methods:
        method_data = ranking[ranking['method'] == method]
        for model in models:
            model_method_data = method_data[method_data['model'] == model]
            if not model_method_data.empty:
                marker = method_markers[method]
                plt.scatter(
                    model_method_data['avg_tokens'],
                    model_method_data['overall_accuracy'],
                    c=[model_colors[model]],
                    s=200,
                    alpha=0.8,
                    marker=marker,
                    edgecolor='white',
                    linewidth=1.5
                )
    
    # Create a simple curve that stays above all points
    all_x = ranking['avg_tokens'].values
    all_y = ranking['overall_accuracy'].values

    # Create points for the curve
    x_curve = np.logspace(np.log10(min(all_x) * 0.9), np.log10(max(all_x) * 1.1), 100)
    
    y_curve = 2.46 * np.log(13.2 * np.log(0.00170 * x_curve) + 3.25) + 61.3
    
    # Plot the curve
    plt.plot(x_curve, y_curve, color='#E0D3C3', linewidth=2.5, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Average Tokens', fontsize=22, labelpad=10)
    plt.ylabel('Accuracy (%)', fontsize=22, labelpad=10)
    
    # Set tick label sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Add grid with custom style
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to logarithmic scale
    plt.xscale('log')
    
    # Create manual legends for both models and methods with improved styling
    # Method legend
    method_handles = [plt.Line2D([0], [0], marker=method_markers[m], 
                               color='gray', label=m, markersize=10, linestyle='None',
                               markeredgecolor='white', markeredgewidth=1.5)
                    for m in methods]
    
    # Model legend - use colored circles for each model with abbreviated names
    model_handles = []
    for m in models:
        # Use the abbreviated model name if available
        model_name = m
        # Find if any of the keys in model_abbreviations is contained in the model name
        for full_name, abbreviated_name in model_abbreviations.items():
            if full_name in m:
                model_name = abbreviated_name
                break
        
        model_handles.append(plt.Line2D([0], [0], marker='o', 
                               color=model_colors[m], label=model_name, linestyle='None',
                               markersize=10, markeredgecolor='white', markeredgewidth=1.5))
    
    # Model legend (bottom right, top box)
    legend_model = plt.legend(handles=model_handles, title='Model', loc='lower right', fontsize=18, title_fontsize=18,
                             frameon=True, framealpha=0.95, edgecolor='gray', ncol=1, bbox_to_anchor=(1, 0.00), borderaxespad=0.)
    plt.gca().add_artist(legend_model)

    # Method legend (bottom right, middle box)
    legend_method = plt.legend(handles=method_handles, title='Method', loc='lower right', fontsize=18, title_fontsize=18,
                              frameon=True, framealpha=0.95, edgecolor='gray', ncol=1, bbox_to_anchor=(0.7, 0), borderaxespad=0.)
    plt.gca().add_artist(legend_method)
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_vs_tokens_non_image.png'), 
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Token analysis complete. Saved to {os.path.join(plots_dir, 'accuracy_vs_tokens_non_image.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token usage vs accuracy")
    parser.add_argument("--base_dir", default="results", help="Base directory containing dataset folders")
    parser.add_argument("--models", nargs="+", help="List of models to include in analysis")
    parser.add_argument("--methods", default=["base","tool","correction"], help="List of methods to include in analysis")
    args = parser.parse_args()
    
    analyze_tokens(
        base_dir=args.base_dir,
        include_models=["gpt-4.1", "gemini-2.5-pro", "gemini-2.0-flash", 
                        "o4-mini", "llama-4-maverick", "qwen3-235b-a22b", 
                        "deepseek-chat", "deepseek-reasoner", "claude-3.7"],
        include_methods=args.methods
    )