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
    
    required_columns = ['model', 'method', 'is_correct']
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

def get_category_columns(results_df):
    """Identify available category columns in the data."""
    expected_categories = ['Materials', 'Properties', 'Structures', 
                          'Fundamental Mechanisms', 'Processes', 'Failure Mechanisms']
    available_categories = [col for col in expected_categories if col in results_df.columns]
    
    if not available_categories:
        print(f"\nWarning: No category columns found. Available columns: {results_df.columns.tolist()}")
    else:
        print(f"\nFound category columns: {available_categories}")
        
    return available_categories

def expand_data_by_category(results_df, category_columns):
    """Expand each row into multiple rows based on categories."""
    expanded_rows = []
    
    for _, row in results_df.iterrows():
        base_data = {
            'model': row['model'],
            'method': row['method'],
            'is_correct': row['is_correct'],
            'qid': row.get('qid', row.get('full_qid', f'q_{_}')),  # Use qid, full_qid, or generate one
            'run_id': row.get('run_id', 1)  # Use run_id if available, otherwise default to 1
        }
        
        for category in category_columns:
            if pd.notna(row[category]):
                expanded_row = base_data.copy()
                expanded_row['category'] = category
                expanded_rows.append(expanded_row)
    
    return pd.DataFrame(expanded_rows)

def calculate_multi_run_statistics(expanded_df, qa_df, has_image):
    """Calculate statistics across multiple runs for the same model-method-category combination."""
    if 'run_id' not in expanded_df.columns:
        print("No run_id column found. Assuming single run per model-method.")
        return calculate_accuracy_stats(expanded_df, qa_df, has_image), None
    
    # Group by model, method, category, and run_id to get per-run accuracies
    run_stats = expanded_df.groupby(['model', 'method', 'category', 'run_id']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'qid': 'nunique'
    }).reset_index()
    
    run_stats.columns = ['model', 'method', 'category', 'run_id', 'total_entries', 'correct_answers', 'accuracy', 'unique_questions']
    run_stats['accuracy_percent'] = run_stats['accuracy'] * 100
    
    # Get total number of questions based on image presence for overall accuracy calculation
    if has_image:
        total_questions = qa_df[qa_df['image'].notna() & (qa_df['image'] != '')].shape[0]
    else:
        total_questions = qa_df[qa_df['image'].isna() | (qa_df['image'] == '')].shape[0]
    
    # Calculate overall accuracy per run (as percentage)
    run_stats['overall_accuracy'] = run_stats.apply(
        lambda row: (row['correct_answers'] / total_questions * 100) if total_questions > 0 else 0,
        axis=1
    )
    
    # Now calculate statistics across runs
    multi_run_stats = run_stats.groupby(['model', 'method', 'category']).agg({
        'accuracy': ['count', 'mean', 'std', 'min', 'max'],
        'overall_accuracy': ['mean', 'std', 'min', 'max'],
        'total_entries': 'mean',
        'unique_questions': 'mean'
    }).reset_index()
    
    # Flatten column names
    multi_run_stats.columns = ['model', 'method', 'category', 'num_runs', 'mean_category_accuracy', 'std_category_accuracy', 'min_category_accuracy', 'max_category_accuracy', 'mean_overall_accuracy', 'std_overall_accuracy', 'min_overall_accuracy', 'max_overall_accuracy', 'mean_total_entries', 'mean_unique_questions']
    
    # Calculate standard error and confidence intervals for category accuracy
    multi_run_stats['std_category_accuracy'] = multi_run_stats['std_category_accuracy'].fillna(0)
    multi_run_stats['category_accuracy_se'] = multi_run_stats['std_category_accuracy'] / np.sqrt(multi_run_stats['num_runs'])
    multi_run_stats['category_accuracy_95ci_lower'] = multi_run_stats['mean_category_accuracy'] - 1.96 * multi_run_stats['category_accuracy_se']
    multi_run_stats['category_accuracy_95ci_upper'] = multi_run_stats['mean_category_accuracy'] + 1.96 * multi_run_stats['category_accuracy_se']
    
    # Calculate standard error and confidence intervals for overall accuracy
    multi_run_stats['std_overall_accuracy'] = multi_run_stats['std_overall_accuracy'].fillna(0)
    multi_run_stats['overall_accuracy_se'] = multi_run_stats['std_overall_accuracy'] / np.sqrt(multi_run_stats['num_runs'])
    multi_run_stats['overall_accuracy_95ci_lower'] = multi_run_stats['mean_overall_accuracy'] - 1.96 * multi_run_stats['overall_accuracy_se']
    multi_run_stats['overall_accuracy_95ci_upper'] = multi_run_stats['mean_overall_accuracy'] + 1.96 * multi_run_stats['overall_accuracy_se']
    
    # Convert to percentages - accuracy values are in 0-1 scale, overall_accuracy is already in percentage scale
    for col in ['mean_category_accuracy', 'std_category_accuracy', 'min_category_accuracy', 'max_category_accuracy', 'category_accuracy_se', 'category_accuracy_95ci_lower', 'category_accuracy_95ci_upper']:
        multi_run_stats[f'{col}_percent'] = multi_run_stats[col] * 100
    
    # Overall accuracy is already in percentage scale, so just copy
    for col in ['mean_overall_accuracy', 'std_overall_accuracy', 'min_overall_accuracy', 'max_overall_accuracy', 'overall_accuracy_se', 'overall_accuracy_95ci_lower', 'overall_accuracy_95ci_upper']:
        multi_run_stats[f'{col}_percent'] = multi_run_stats[col]
    
    return multi_run_stats, run_stats

def calculate_accuracy_stats(expanded_df, qa_df, has_image):
    """Calculate accuracy statistics for each model-method-category combination."""
    if expanded_df.empty:
        print("\nNo valid category data found.")
        return pd.DataFrame()
    
    print(f"\nFound {len(expanded_df)} category entries across all questions")
    
    # Get total number of questions based on image presence
    if has_image:
        total_questions = qa_df[qa_df['image'].notna() & (qa_df['image'] != '')].shape[0]
    else:
        total_questions = qa_df[qa_df['image'].isna() | (qa_df['image'] == '')].shape[0]
    
    # Calculate accuracy for each model-method-category combination
    stats = expanded_df.groupby(['model', 'method', 'category']).agg({
        'is_correct': ['count', 'sum'],
        'qid': 'nunique'  # Count unique questions per category
    }).reset_index()
    
    stats.columns = ['model', 'method', 'category', 'total_entries', 'correct_answers', 'unique_questions']
    # Calculate per-category accuracy as before
    stats['category_accuracy'] = stats.apply(
        lambda row: (row['correct_answers'] / row['total_entries'] * 100) if row['total_entries'] > 0 else 0,
        axis=1
    )
    # Calculate overall accuracy using total questions as denominator
    stats['overall_accuracy'] = stats.apply(
        lambda row: (row['correct_answers'] / total_questions * 100) if total_questions > 0 else 0,
        axis=1
    )
    
    return stats.sort_values(['model', 'method', 'category'])

def print_accuracy_results(accuracy_stats, expanded_df, has_image):
    """Print detailed accuracy statistics."""
    image_status = "with" if has_image else "without"
    print(f"\nAccuracy Statistics by Model, Method, and Category (Questions {image_status} images):")
    print("=" * 80)
    
    for (model, method), group in accuracy_stats.groupby(['model', 'method']):
        print(f"\n{model} - {method}:")
        print("-" * 40)
        print(group[['category', 'category_accuracy', 'overall_accuracy', 'unique_questions', 'correct_answers']].to_string(index=False))
        
        # Calculate average accuracy across all categories
        avg_category_accuracy = group['category_accuracy'].mean()
        avg_overall_accuracy = group['overall_accuracy'].mean()
        print(f"\nAverage category accuracy: {avg_category_accuracy:.2f}%")
        print(f"Average overall accuracy: {avg_overall_accuracy:.2f}%")
    
    print(f"\nSummary Statistics by Category (Questions {image_status} images):")
    print("=" * 80)
    summary = accuracy_stats.groupby(['category']).agg({
        'category_accuracy': ['mean', 'std', 'min', 'max'],
        'overall_accuracy': ['mean', 'std', 'min', 'max'],
        'unique_questions': 'sum'
    }).round(2)
    print(summary)

def print_multi_run_results(accuracy_stats, run_stats, has_image):
    """Print accuracy results for multi-run analysis with error bars."""
    print("\n" + "="*80)
    print(f" MULTI-RUN CATEGORY ACCURACY ANALYSIS - {'WITH' if has_image else 'WITHOUT'} IMAGES ")
    print("="*80)
    
    if accuracy_stats.empty:
        print("No data available for analysis.")
        return
    
    for _, row in accuracy_stats.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"Method: {row['method']}")
        print(f"Category: {row['category']}")
        print(f"Number of runs: {row['num_runs']}")
        print(f"Questions per run: {row['mean_unique_questions']:.0f}")
        
        # Print individual run accuracies if available
        if run_stats is not None:
            individual_runs = run_stats[
                (run_stats['model'] == row['model']) & 
                (run_stats['method'] == row['method']) & 
                (run_stats['category'] == row['category'])
            ].sort_values('run_id')
            
            if not individual_runs.empty:
                run_accuracies = [f"Run {r['run_id']}: {r['accuracy_percent']:.2f}%" 
                                for _, r in individual_runs.iterrows()]
                print(f"Individual run category accuracies: {', '.join(run_accuracies)}")
                
                run_overall_accuracies = [f"Run {r['run_id']}: {r['overall_accuracy']:.2f}%" 
                                        for _, r in individual_runs.iterrows()]
                print(f"Individual run overall accuracies: {', '.join(run_overall_accuracies)}")
        
        # Category accuracy statistics
        std_text = f"{row['std_category_accuracy_percent']:.2f}" if not np.isnan(row['std_category_accuracy_percent']) else "0.00"
        ci_lower = f"{row['category_accuracy_95ci_lower_percent']:.2f}" if not np.isnan(row['category_accuracy_95ci_lower_percent']) else f"{row['mean_category_accuracy_percent']:.2f}"
        ci_upper = f"{row['category_accuracy_95ci_upper_percent']:.2f}" if not np.isnan(row['category_accuracy_95ci_upper_percent']) else f"{row['mean_category_accuracy_percent']:.2f}"
        
        print(f"Mean category accuracy: {row['mean_category_accuracy_percent']:.2f}% ± {std_text}%")
        print(f"Category accuracy 95% CI: [{ci_lower}%, {ci_upper}%]")
        print(f"Category accuracy range: {row['min_category_accuracy_percent']:.2f}% - {row['max_category_accuracy_percent']:.2f}%")
        
        # Overall accuracy statistics
        std_overall_text = f"{row['std_overall_accuracy_percent']:.2f}" if not np.isnan(row['std_overall_accuracy_percent']) else "0.00"
        ci_overall_lower = f"{row['overall_accuracy_95ci_lower_percent']:.2f}" if not np.isnan(row['overall_accuracy_95ci_lower_percent']) else f"{row['mean_overall_accuracy_percent']:.2f}"
        ci_overall_upper = f"{row['overall_accuracy_95ci_upper_percent']:.2f}" if not np.isnan(row['overall_accuracy_95ci_upper_percent']) else f"{row['mean_overall_accuracy_percent']:.2f}"
        
        print(f"Mean overall accuracy: {row['mean_overall_accuracy_percent']:.2f}% ± {std_overall_text}%")
        print(f"Overall accuracy 95% CI: [{ci_overall_lower}%, {ci_overall_upper}%]")
        print(f"Overall accuracy range: {row['min_overall_accuracy_percent']:.2f}% - {row['max_overall_accuracy_percent']:.2f}%")
        print("-" * 60)

def create_visualizations(accuracy_stats, plots_dir, has_image):
    """Create and save visualization plots."""
    if accuracy_stats.empty:
        return
    
    image_status = "with_images" if has_image else "without_images"
    
    # Overall bar plot for category accuracy
    plt.figure(figsize=(15, 10))
    sns.barplot(data=accuracy_stats, x='category', y='mean_category_accuracy_percent', hue='model', palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Category Accuracy by Model (Questions {image_status.replace("_", " ")})')
    plt.xlabel('Category')
    plt.ylabel('Category Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'category_accuracy_by_main_category_{image_status}.png'))
    plt.close()
    
    # Overall bar plot for overall accuracy
    plt.figure(figsize=(15, 10))
    sns.barplot(data=accuracy_stats, x='category', y='mean_overall_accuracy_percent', hue='model', palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Overall Accuracy by Model (Questions {image_status.replace("_", " ")})')
    plt.xlabel('Category')
    plt.ylabel('Overall Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'overall_accuracy_by_main_category_{image_status}.png'))
    plt.close()
    
    # Model-specific plots
    for model in accuracy_stats['model'].unique():
        model_data = accuracy_stats[accuracy_stats['model'] == model]
        if not model_data.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=model_data, x='category', y='mean_category_accuracy_percent', hue='method', palette='Set1')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Category Accuracy by Category - {model} (Questions {image_status.replace("_", " ")})')
            plt.xlabel('Category')
            plt.ylabel('Category Accuracy (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'category_accuracy_by_main_category_{model}_{image_status}.png'))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=model_data, x='category', y='mean_overall_accuracy_percent', hue='method', palette='Set1')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Overall Accuracy by Category - {model} (Questions {image_status.replace("_", " ")})')
            plt.xlabel('Category')
            plt.ylabel('Overall Accuracy (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'overall_accuracy_by_main_category_{model}_{image_status}.png'))
            plt.close()
    
    # Heatmap for category accuracy
    pivot_data = accuracy_stats.pivot_table(
        values='mean_category_accuracy_percent', index='category', columns='model', aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Category Accuracy (%)'})
    plt.title(f'Average Category Accuracy by Category and Model (Questions {image_status.replace("_", " ")})')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'category_accuracy_heatmap_main_category_{image_status}.png'))
    plt.close()
    
    # Heatmap for overall accuracy
    pivot_data = accuracy_stats.pivot_table(
        values='mean_overall_accuracy_percent', index='category', columns='model', aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Overall Accuracy (%)'})
    plt.title(f'Average Overall Accuracy by Category and Model (Questions {image_status.replace("_", " ")})')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'overall_accuracy_heatmap_main_category_{image_status}.png'))
    plt.close()

def print_formatted_table(accuracy_stats):
    """Print results in the requested table format."""
    if accuracy_stats.empty:
        return
    
    # Method mapping for display names
    method_mapping = {
        'base': '',
        'correction': '+Correction',
        'tool': '+Tool'
    }
    
    # Category column mapping
    category_mapping = {
        'Failure Mechanisms': 'Failure',
        'Fundamental Mechanisms': 'Fund.',
        'Materials': 'Materials',
        'Processes': 'Processes', 
        'Properties': 'Properties',
        'Structures': 'Structures'
    }
    
    # Create pivot table for mean overall accuracy
    pivot_df = accuracy_stats.pivot_table(
        index=['model', 'method'], 
        columns='category', 
        values='mean_overall_accuracy_percent',
        aggfunc='mean'
    ).reset_index()
    
    # Add overall average column
    category_cols = [col for col in pivot_df.columns if col not in ['model', 'method']]
    pivot_df['Overall'] = pivot_df[category_cols].mean(axis=1)
    
    # Format the results
    results = []
    
    # Process each model-method combination
    for _, row in pivot_df.iterrows():
        model_key = row['model']
        method_key = row['method']
        
        # Use original model name
        display_method = method_mapping.get(method_key, method_key)
        
        # Combine model and method names
        if display_method:
            model_name = f"{model_key} {display_method}"
        else:
            model_name = model_key
            
        # Extract values for each category
        result_row = {'Model': model_name}
        
        for category, display_category in category_mapping.items():
            if category in pivot_df.columns:
                value = row[category]
                result_row[display_category] = f"{value:.2f}" if not pd.isna(value) else "N/A"
        
        # Add overall average
        result_row['Overall'] = f"{row['Overall']:.2f}" if not pd.isna(row['Overall']) else "N/A"
        
        results.append(result_row)
    
    # Convert to DataFrame for easier formatting
    result_df = pd.DataFrame(results)
    
    # Print the formatted table
    print("\n" + "="*90)
    print("FINAL RESULTS TABLE")
    print("="*90)
    print("Model                           Failure Fund. Materials Processes Properties Structures Overall")
    print("-" * 95)
    
    # Print all results without classification
    for _, row in result_df.iterrows():
        print(f"{row['Model']:<31} {row.get('Failure', 'N/A'):>7} {row.get('Fund.', 'N/A'):>5} {row.get('Materials', 'N/A'):>9} {row.get('Processes', 'N/A'):>9} {row.get('Properties', 'N/A'):>10} {row.get('Structures', 'N/A'):>10} {row['Overall']:>7}")

def print_multi_run_table(accuracy_stats):
    """Print results with error bars for models+methods that have multiple runs."""
    if accuracy_stats.empty:
        return
    
    # Filter for models+methods with multiple runs
    multi_run_data = accuracy_stats[accuracy_stats['num_runs'] > 1].copy()
    
    if multi_run_data.empty:
        print("\n" + "="*90)
        print("MULTI-RUN RESULTS TABLE")
        print("="*90)
        print("No models with multiple runs found.")
        return
    
    # Method mapping for display names
    method_mapping = {
        'base': '',
        'correction': '+Correction',
        'tool': '+Tool'
    }
    
    # Category column mapping
    category_mapping = {
        'Failure Mechanisms': 'Failure',
        'Fundamental Mechanisms': 'Fund.',
        'Materials': 'Materials',
        'Processes': 'Processes', 
        'Properties': 'Properties',
        'Structures': 'Structures'
    }
    
    # Format the results with error bars
    results = []
    
    for _, row in multi_run_data.iterrows():
        model_key = row['model']
        method_key = row['method']
        
        # Use original model name
        display_method = method_mapping.get(method_key, method_key)
        
        # Combine model and method names
        if display_method:
            model_name = f"{model_key} {display_method}"
        else:
            model_name = model_key
        
        # Create result row with mean±std for each category
        result_row = {'Model': model_name, 'Runs': int(row['num_runs'])}
        
        # Add category results with error bars
        for orig_category, display_category in category_mapping.items():
            mean_col = f'mean_overall_accuracy_percent'
            std_col = f'std_overall_accuracy_percent'
            
            # Find the row for this specific category
            category_row = multi_run_data[
                (multi_run_data['model'] == model_key) & 
                (multi_run_data['method'] == method_key) & 
                (multi_run_data['category'] == orig_category)
            ]
            
            if not category_row.empty:
                mean_val = category_row[mean_col].iloc[0]
                std_val = category_row[std_col].iloc[0]
                
                if not pd.isna(mean_val) and not pd.isna(std_val):
                    result_row[display_category] = f"{mean_val:.2f}±{std_val:.2f}"
                elif not pd.isna(mean_val):
                    result_row[display_category] = f"{mean_val:.2f}±0.00"
                else:
                    result_row[display_category] = "N/A"
            else:
                result_row[display_category] = "N/A"
        
        # Calculate overall mean and std across categories for this model+method
        model_method_data = multi_run_data[
            (multi_run_data['model'] == model_key) & 
            (multi_run_data['method'] == method_key)
        ]
        
        if not model_method_data.empty:
            overall_mean = model_method_data['mean_overall_accuracy_percent'].mean()
            overall_std = model_method_data['std_overall_accuracy_percent'].mean()  # Average of std across categories
            
            if not pd.isna(overall_mean) and not pd.isna(overall_std):
                result_row['Overall'] = f"{overall_mean:.2f}±{overall_std:.2f}"
            elif not pd.isna(overall_mean):
                result_row['Overall'] = f"{overall_mean:.2f}±0.00"
            else:
                result_row['Overall'] = "N/A"
        else:
            result_row['Overall'] = "N/A"
        
        results.append(result_row)
    
    # Remove duplicates (since we iterate through all category rows)
    seen = set()
    unique_results = []
    for result in results:
        key = (result['Model'], result['Runs'])
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    
    # Convert to DataFrame for easier handling
    result_df = pd.DataFrame(unique_results)
    
    # Print the formatted table
    print("\n" + "="*110)
    print("MULTI-RUN RESULTS TABLE (Mean±Std)")
    print("="*110)
    print("Model                           Runs  Failure      Fund.    Materials  Processes Properties Structures  Overall")
    print("-" * 110)
    
    # Print all multi-run results
    for _, row in result_df.iterrows():
        print(f"{row['Model']:<31} {row['Runs']:>4} {row.get('Failure', 'N/A'):>10} {row.get('Fund.', 'N/A'):>8} {row.get('Materials', 'N/A'):>10} {row.get('Processes', 'N/A'):>9} {row.get('Properties', 'N/A'):>10} {row.get('Structures', 'N/A'):>10} {row.get('Overall', 'N/A'):>9}")

def analyze_accuracy_by_category(has_image=False):
    """Analyze accuracy by category for questions with or without images."""
    results_dir, plots_dir = setup_directories()
    
    # Load the evaluation data
    results_df, qa_df = load_evaluation_data()
    
    # Count questions with and without images
    questions_with_images = qa_df[qa_df['image'].notna() & (qa_df['image'] != '')].shape[0]
    questions_without_images = qa_df[qa_df['image'].isna() | (qa_df['image'] == '')].shape[0]
    total_questions = qa_df.shape[0]
    
    print(f"\nQuestion Count Analysis:")
    print(f"Total questions: {total_questions}")
    print(f"Questions with images: {questions_with_images} ({questions_with_images/total_questions*100:.1f}%)")
    print(f"Questions without images: {questions_without_images} ({questions_without_images/total_questions*100:.1f}%)")
    
    # Filter and validate data
    results_df = filter_and_validate_data(results_df, has_image)
    
    # Get category columns
    category_columns = get_category_columns(results_df)
    if not category_columns:
        return pd.DataFrame()
    
    # Expand data by category
    expanded_df = expand_data_by_category(results_df, category_columns)
    
    # Calculate accuracy statistics (with multi-run support)
    accuracy_stats, run_stats = calculate_multi_run_statistics(expanded_df, qa_df, has_image)
    if accuracy_stats.empty:
        return pd.DataFrame()
    
    # Print results
    print_multi_run_results(accuracy_stats, run_stats, has_image)
    
    # Save results to CSV
    image_status = "with_images" if has_image else "without_images"
    output_file = os.path.join(results_dir, f'accuracy_by_main_category_{image_status}.csv')
    accuracy_stats.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    # Create visualizations
    create_visualizations(accuracy_stats, plots_dir, has_image)
    
    # Print formatted table
    print_formatted_table(accuracy_stats)
    
    # Print multi-run table with error bars
    print_multi_run_table(accuracy_stats)
    
    return accuracy_stats

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Analyze accuracy by category')
        parser.add_argument('--has_image', action='store_true', 
                           help='Analyze questions with images instead of without')
        parser.add_argument('--base_dir', type=str, default='../results',
                           help='Base directory containing dataset results')
        parser.add_argument('--difficulty_file', type=str, default='../datasets/qa.csv',
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
        analyze_accuracy_by_category(has_image=args.has_image)
        print("\nAnalysis completed successfully!")
        
        # Restore original function
        data_processor.load_evaluation_data = original_load_eval
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
