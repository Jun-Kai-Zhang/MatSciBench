import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import data_processor
from data_processor import load_evaluation_data
import os
import argparse

from evaluation.eval import MULTIMODAL_MODELS


def setup_directories():
    """Create output directories for results and plots."""
    results_dir = os.path.join("analysis", "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return results_dir, plots_dir

def filter_and_validate_data(results_df, has_image):
    """Filter data based on image presence and validate required columns."""
    if 'has_image' in results_df.columns:
        results_df = results_df[results_df['has_image'] == has_image]
        print(f"\nAnalyzing questions {'with' if has_image else 'without'} images...")
    else:
        print("\nWarning: No image information available. Analyzing all questions.")
    
    required_columns = ['model', 'method', 'is_correct', 'difficulty_level']
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter by multimodal models when analyzing questions with images
    if has_image:
        # Filter models that start with or contain any model name in MULTIMODAL_MODELS
        multimodal_models_mask = results_df['model'].apply(
            lambda model_name: any(mm_model in model_name for mm_model in MULTIMODAL_MODELS)
        )
        filtered_df = results_df[multimodal_models_mask]
        
        if filtered_df.empty:
            print(f"Warning: No models from {MULTIMODAL_MODELS} found in the data.")
            return filtered_df
        
        print(f"\nFiltered to only include multimodal models: {filtered_df['model'].unique().tolist()}")
        return filtered_df
    
    return results_df

def calculate_multi_run_statistics(results_df):
    """Calculate statistics across multiple runs for the same model-method combination."""
    if 'run_id' not in results_df.columns:
        print("No run_id column found. Assuming single run per model-method.")
        return calculate_accuracy_stats(results_df), None
    
    # Group by model, method, difficulty_level, and run_id to get per-run accuracies
    run_stats = results_df.groupby(['model', 'method', 'difficulty_level', 'run_id']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'qid': 'nunique'
    }).reset_index()
    
    run_stats.columns = ['model', 'method', 'difficulty_level', 'run_id', 'total_entries', 'correct_answers', 'accuracy', 'unique_questions']
    run_stats['accuracy_percent'] = run_stats['accuracy'] * 100
    
    # Now calculate statistics across runs
    multi_run_stats = run_stats.groupby(['model', 'method', 'difficulty_level']).agg({
        'accuracy': ['count', 'mean', 'std', 'min', 'max'],
        'total_entries': 'mean',
        'unique_questions': 'mean'
    }).reset_index()
    
    # Flatten column names
    multi_run_stats.columns = ['model', 'method', 'difficulty_level', 'num_runs', 'mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy', 'mean_total_entries', 'mean_unique_questions']
    
    # Calculate standard error and confidence intervals
    # Handle case where std_accuracy is NaN (single run)
    multi_run_stats['std_accuracy'] = multi_run_stats['std_accuracy'].fillna(0)
    multi_run_stats['accuracy_se'] = multi_run_stats['std_accuracy'] / np.sqrt(multi_run_stats['num_runs'])
    multi_run_stats['accuracy_95ci_lower'] = multi_run_stats['mean_accuracy'] - 1.96 * multi_run_stats['accuracy_se']
    multi_run_stats['accuracy_95ci_upper'] = multi_run_stats['mean_accuracy'] + 1.96 * multi_run_stats['accuracy_se']
    
    # Convert to percentages
    for col in ['mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy', 'accuracy_se', 'accuracy_95ci_lower', 'accuracy_95ci_upper']:
        multi_run_stats[f'{col}_percent'] = multi_run_stats[col] * 100
    
    return multi_run_stats, run_stats

def calculate_accuracy_stats(results_df):
    """Calculate accuracy statistics for each model-method-difficulty combination."""
    if results_df.empty or 'difficulty_level' not in results_df.columns:
        print("\nNo valid difficulty data found.")
        return pd.DataFrame()
    
    # Filter out rows with missing difficulty levels
    valid_df = results_df.dropna(subset=['difficulty_level'])
    if valid_df.empty:
        print("\nNo rows with valid difficulty levels found.")
        return pd.DataFrame()
    
    print(f"\nFound {len(valid_df)} entries with difficulty information")
    
    # Calculate accuracy for each model-method-difficulty combination
    stats = valid_df.groupby(['model', 'method', 'difficulty_level']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'qid': 'nunique'  # Count unique questions per difficulty level
    }).reset_index()
    
    stats.columns = ['model', 'method', 'difficulty_level', 'total_entries', 'correct_answers', 'accuracy', 'unique_questions']
    stats['accuracy_percent'] = stats['accuracy'] * 100
    
    return stats.sort_values(['model', 'method', 'difficulty_level'])

def print_accuracy_results(accuracy_stats, has_image):
    """Print detailed accuracy statistics."""
    image_status = "with" if has_image else "without"
    print(f"\nAccuracy Statistics by Model, Method, and Difficulty Level (Questions {image_status} images):")
    print("=" * 80)
    
    for (model, method), group in accuracy_stats.groupby(['model', 'method']):
        print(f"\n{model} - {method}:")
        print("-" * 40)
        print(group[['difficulty_level', 'accuracy_percent', 'unique_questions', 'correct_answers', 'total_entries']].to_string(index=False))
        
        # Calculate overall accuracy
        total_correct = group['correct_answers'].sum()
        total_questions = group['total_entries'].sum()
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        print(f"\nOverall accuracy: {overall_accuracy:.2f}% (Total questions: {total_questions})")
    
    print(f"\nSummary Statistics by Difficulty Level (Questions {image_status} images):")
    print("=" * 80)
    summary = accuracy_stats.groupby(['difficulty_level']).agg({
        'accuracy_percent': ['mean', 'std', 'min', 'max'],
        'unique_questions': 'sum'
    }).round(2)
    print(summary)

def create_visualizations(accuracy_stats, plots_dir, has_image):
    """Create and save visualization plots."""
    if accuracy_stats.empty:
        return
    
    image_status = "with_images" if has_image else "without_images"
    
    # Filter for only base method and exclude undefined difficulty
    cot_data = accuracy_stats[
        (accuracy_stats['method'] == 'base') & 
        (accuracy_stats['difficulty_level'] != 'undefined')
    ].copy()  # Create an explicit copy
    
    # Define difficulty level order
    difficulty_order = ['easy', 'medium', 'hard']
    cot_data.loc[:, 'difficulty_level'] = pd.Categorical(
        cot_data['difficulty_level'], 
        categories=difficulty_order,
        ordered=True
    )
    
    # Define model order and abbreviations
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
    
    # Create a custom order for models
    all_models = cot_data['model'].unique()
    models_to_move = ['deepseek-reasoner', 'o4-mini', 'gemini-2.5-pro', 'qwen3']
    other_models = [m for m in all_models if not any(mm in m for mm in models_to_move)]
    model_order = other_models + [m for m in all_models if any(mm in m for mm in models_to_move)]
    
    # Apply abbreviations - handle partial matches
    def get_abbreviation(model_name):
        for base_name, abbrev in model_abbreviations.items():
            if base_name in model_name:
                return abbrev
        return model_name
    
    cot_data.loc[:, 'model'] = cot_data['model'].map(get_abbreviation)
    
    # Define morandi color palette with more distinct colors
    morandi_colors = ['#E0D3C3','#BFCAC2', '#5B7493',  '#F6E1DC', '#8D91AA', '#DFDFDF', '#ACD4D6', '#E0D3C3']  
    
    # Create grouped bar plot
    plt.figure(figsize=(15, 3))  # Reduced height from 4 to 3
    
    # Set up the plot
    sns.set_style("whitegrid")
    ax = sns.barplot(data=cot_data, 
                    x='model', 
                    y='accuracy_percent', 
                    hue='difficulty_level',
                    palette=morandi_colors[:len(difficulty_order)],
                    hue_order=difficulty_order,
                    order=[get_abbreviation(m) for m in model_order],
                    width=0.6)  # Removed fontsize parameter
    
    # Adjust spacing between bars
    for i, bar in enumerate(ax.patches):
        bar.set_width(0.2)  # Make individual bars narrower
        # Add spacing between groups
        if i % 3 == 0:  # For each new model group
            bar.set_x(bar.get_x() + 0.1)  # Add spacing between model groups
    
    # Customize the plot
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    plt.xticks(fontsize=12)  # Added fontsize for x-axis labels
    plt.legend(title='Difficulty Level', bbox_to_anchor=(1.01, 1.01), loc='upper left', fontsize=17, title_fontsize=17)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'accuracy_by_difficulty_grouped_{image_status}.png'), 
                bbox_inches='tight')
    plt.close()

def print_multi_run_results(accuracy_stats, run_stats, has_image):
    """Print accuracy results for multi-run analysis with error bars."""
    print("\n" + "="*80)
    print(f" MULTI-RUN ACCURACY ANALYSIS - {'WITH' if has_image else 'WITHOUT'} IMAGES ")
    print("="*80)
    
    if accuracy_stats.empty:
        print("No data available for analysis.")
        return
    
    for _, row in accuracy_stats.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"Method: {row['method']}")
        print(f"Difficulty: {row['difficulty_level']}")
        print(f"Number of runs: {row['num_runs']}")
        print(f"Questions per run: {row['mean_unique_questions']:.0f}")
        
        # Print individual run accuracies if available
        if run_stats is not None:
            individual_runs = run_stats[
                (run_stats['model'] == row['model']) & 
                (run_stats['method'] == row['method']) & 
                (run_stats['difficulty_level'] == row['difficulty_level'])
            ].sort_values('run_id')
            
            if not individual_runs.empty:
                run_accuracies = [f"Run {r['run_id']}: {r['accuracy_percent']:.2f}%" 
                                for _, r in individual_runs.iterrows()]
                print(f"Individual run accuracies: {', '.join(run_accuracies)}")
        
        std_text = f"{row['std_accuracy_percent']:.2f}" if not np.isnan(row['std_accuracy_percent']) else "0.00"
        ci_lower = f"{row['accuracy_95ci_lower_percent']:.2f}" if not np.isnan(row['accuracy_95ci_lower_percent']) else f"{row['mean_accuracy_percent']:.2f}"
        ci_upper = f"{row['accuracy_95ci_upper_percent']:.2f}" if not np.isnan(row['accuracy_95ci_upper_percent']) else f"{row['mean_accuracy_percent']:.2f}"
        
        print(f"Mean accuracy: {row['mean_accuracy_percent']:.2f}% ± {std_text}%")
        print(f"95% CI: [{ci_lower}%, {ci_upper}%]")
        print(f"Range: {row['min_accuracy_percent']:.2f}% - {row['max_accuracy_percent']:.2f}%")
        print("-" * 60)

def create_multi_run_visualizations(accuracy_stats, plots_dir, has_image):
    """Create visualizations for multi-run analysis with error bars."""
    if accuracy_stats.empty:
        print("No data to visualize.")
        return
    
    image_status = "with_images" if has_image else "without_images"
    
    # Create bar plot with error bars
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    models = accuracy_stats['model'].unique()
    methods = accuracy_stats['method'].unique()
    difficulties = accuracy_stats['difficulty_level'].unique()
    
    # Set up the plot
    x = np.arange(len(models))
    width = 0.25
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(difficulties)))
    
    for i, difficulty in enumerate(difficulties):
        data = accuracy_stats[accuracy_stats['difficulty_level'] == difficulty]
        means = []
        errors = []
        
        for model in models:
            model_data = data[data['model'] == model]
            if not model_data.empty:
                means.append(model_data['mean_accuracy_percent'].iloc[0])
                errors.append(model_data['std_accuracy_percent'].iloc[0])
            else:
                means.append(0)
                errors.append(0)
        
        plt.bar(x + i * width, means, width, 
                yerr=errors, capsize=5, 
                label=f'Difficulty {difficulty}', 
                color=colors[i], alpha=0.8)
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'Accuracy by Difficulty Level ({image_status.replace("_", " ").title()})\nWith Error Bars from Multiple Runs', fontsize=16)
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend(title='Difficulty Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'multi_run_accuracy_by_difficulty_{image_status}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detailed plot showing confidence intervals
    plt.figure(figsize=(14, 8))
    
    for i, difficulty in enumerate(difficulties):
        data = accuracy_stats[accuracy_stats['difficulty_level'] == difficulty]
        
        for j, model in enumerate(models):
            model_data = data[data['model'] == model]
            if not model_data.empty:
                row = model_data.iloc[0]
                y_pos = j + i * 0.1 - 0.1
                
                # Plot confidence interval as horizontal line
                plt.errorbar(row['mean_accuracy_percent'], y_pos, 
                           xerr=[[row['mean_accuracy_percent'] - row['accuracy_95ci_lower_percent']], 
                                [row['accuracy_95ci_upper_percent'] - row['mean_accuracy_percent']]], 
                           fmt='o', color=colors[i], capsize=5, markersize=8, linewidth=2,
                           label=f'Difficulty {difficulty}' if j == 0 else "")
    
    plt.ylabel('Model', fontsize=14)
    plt.xlabel('Accuracy (%)', fontsize=14)
    plt.title(f'95% Confidence Intervals for Accuracy ({image_status.replace("_", " ").title()})', fontsize=16)
    plt.yticks(range(len(models)), models)
    plt.legend(title='Difficulty Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'multi_run_confidence_intervals_{image_status}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-run visualizations saved to {plots_dir}")

def analyze_accuracy_by_difficulty(has_image=False):
    """Analyze accuracy by difficulty level for questions with or without images."""
    results_dir, plots_dir = setup_directories()
    
    # Load the evaluation data
    results_df, _ = load_evaluation_data()
    
    # Filter and validate data
    results_df = filter_and_validate_data(results_df, has_image)
    
    # Calculate accuracy statistics (with multi-run support)
    accuracy_stats, run_stats = calculate_multi_run_statistics(results_df)
    if accuracy_stats.empty:
        return pd.DataFrame()
    
    # Print results
    print_multi_run_results(accuracy_stats, run_stats, has_image)
    
    # Save results to CSV
    image_status = "with_images" if has_image else "without_images"
    output_file = os.path.join(results_dir, f'accuracy_by_difficulty_{image_status}.csv')
    accuracy_stats.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    # Create visualizations
    create_multi_run_visualizations(accuracy_stats, plots_dir, has_image)
    
    return accuracy_stats

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Analyze accuracy by difficulty level')
        parser.add_argument('--has_image', action='store_true', 
                           help='Analyze questions with images instead of without')
        parser.add_argument('--base_dir', type=str, default='../results',
                           help='Base directory containing dataset results')
        parser.add_argument('--difficulty_file', type=str, default='../datasets/MatSciBench/qa.csv',
                           help='Path to difficulty data CSV file')
        
        args = parser.parse_args()
        
        # Override load_evaluation_data function with our parameters
        original_load_eval = data_processor.load_evaluation_data
        
        def load_eval_with_args(*args_func, **kwargs_func):
            kwargs_func['base_dir'] = args.base_dir
            kwargs_func['difficulty_file'] = args.difficulty_file
            return original_load_eval(*args_func, **kwargs_func)
        
        data_processor.load_evaluation_data = load_eval_with_args
        
        print(f"Analyzing with parameters: base_dir={args.base_dir}, difficulty_file={args.difficulty_file}, has_image={args.has_image}")
        analyze_accuracy_by_difficulty(has_image=args.has_image)
        print("\nAnalysis completed successfully!")
        
        # Restore original function
        data_processor.load_evaluation_data = original_load_eval
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
