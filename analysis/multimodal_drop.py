import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import argparse
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import multimodal models list
from evaluation.eval import MULTIMODAL_MODELS

# Morandi colors palette
morandi_colors = ['#F6E1DC', 
                  '#E0D3C3', 
                  '#D8B0B0', 
                  '#CB9B8F', 
                  '#BFCAC2', 
                  '#ACD4D6', 
                  '#9B8D8C', 
                  '#8D91AA', 
                  '#6E4740',
                  '#5B7493',
                  '#013E41']

def get_short_model_name(long_name):
    for short in MULTIMODAL_MODELS:
        if short.lower() in long_name.lower():
            return short
    return long_name  # fallback if no match

def analyze_images(base_dir="datasets", include_datasets=None):
    """Analyze performance on image vs. non-image questions"""
    # Import here to keep module independent
    from analysis.data_processor import load_evaluation_data
    
    # Load data
    results_df, _ = load_evaluation_data()
    
    if results_df.empty:
        print("No evaluation data available.")
        return
    
    # Print number of questions per model
    print("\nNumber of questions per model:")
    print("=" * 60)
    model_counts = results_df.groupby('model').size().sort_values(ascending=False)
    for model, count in model_counts.items():
        print(f"{model:<50} {count:>5} questions")
    print("=" * 60)
    print()
    
    # Create directories for results and plots
    results_output_dir = os.path.join("analysis", "results")
    plots_dir = os.path.join(results_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create image_df by filtering for image-related questions
    if 'has_image' not in results_df.columns:
        print("No image analysis data available. Make sure result files contain 'has_image' column.")
        return
    
    image_df = results_df.copy()
    
    if image_df.empty:
        print("No image analysis data available.")
        return
    
    # Filter to include only multimodal models
    # Handle model names that may have suffixes
    image_df = image_df[image_df['model'].apply(
        lambda model_name: any(mm_model.lower() in model_name.lower() for mm_model in MULTIMODAL_MODELS)
    )]
    
    # Filter to include only the "base" method
    image_df = image_df[image_df['method'] == 'base']
    
    if image_df.empty:
        print("No multimodal models with base method found in the data.")
        return
        
    print("Analyzing image vs. non-image performance for multimodal models (base method only)...")
    
    # Summarize image data
    image_summary = image_df.groupby(['model', 'method', 'has_image']).agg({
        'is_correct': ['count', 'sum']
    }).reset_index()
    
    # Rename the columns
    image_summary.columns = ['model', 'method', 'has_image', 'total_questions', 'correct_answers']
    
    # Add tokens column if available
    if 'new_token_nums' in image_df.columns:
        token_summary = image_df.groupby(['model', 'method', 'has_image'])['new_token_nums'].mean().reset_index()
        token_summary.rename(columns={'new_token_nums': 'avg_tokens'}, inplace=True)
        image_summary = pd.merge(image_summary, token_summary, on=['model', 'method', 'has_image'])
    else:
        image_summary['avg_tokens'] = np.nan

    image_summary['overall_accuracy'] = (
        image_summary['correct_answers'] / image_summary['total_questions'] * 100
    )

    image_summary['short_model'] = image_summary['model'].apply(get_short_model_name)

    # Calculate performance difference between image and non-image questions
    # Create a pivot table to easily calculate differences
    pivot_df = image_summary.pivot_table(
        index=['model', 'short_model'], 
        columns='has_image', 
        values='overall_accuracy'
    ).reset_index()
    
    # Rename boolean columns for clarity
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={False: 'no_image_accuracy', True: 'image_accuracy'})
    
    # Calculate the difference (image - no_image)
    pivot_df['accuracy_difference'] = pivot_df['image_accuracy'] - pivot_df['no_image_accuracy']
    
    # Print performance difference for each model
    print("\nPerformance difference (image - no_image) for each model:")
    print("=" * 100)
    print(f"{'Model':<20} {'No Image (%)':<15} {'Image (%)':<15} {'Difference (%)':<15} {'No Image (N)':<15} {'Image (N)':<15}")
    print("-" * 100)
    for _, row in pivot_df.sort_values('short_model').iterrows():
        # Get the corresponding counts from image_summary
        no_image_count = image_summary[
            (image_summary['short_model'] == row['short_model']) & 
            (image_summary['has_image'] == False)
        ]['total_questions'].values[0]
        image_count = image_summary[
            (image_summary['short_model'] == row['short_model']) & 
            (image_summary['has_image'] == True)
        ]['total_questions'].values[0]
        
        print(f"{row['short_model']:<20} {row['no_image_accuracy']:.2f}%{' '*9} {row['image_accuracy']:.2f}%{' '*9} {row['accuracy_difference']:.2f}%{' '*9} {no_image_count:<15} {image_count:<15}")
    print("=" * 100)

    # Plot 1: Basic image vs non-image performance
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='short_model', y='overall_accuracy', hue='has_image', data=image_summary,
                    palette=[morandi_colors[1], morandi_colors[9]], width=0.6)
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.xticks(fontsize=20)
    
    # # Add value labels on top of bars
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    # Fix legend to show colored boxes matching the bars
    handles, labels = ax.get_legend_handles_labels()
    # The order of labels may be ['False', 'True'], so map them to 'No' and 'Yes'
    label_map = {'False': 'No', 'True': 'Yes'}
    labels = [label_map.get(l, l) for l in labels]
    ax.legend(handles, labels, title='Has Image', title_fontsize=20, fontsize=20, loc='upper right', bbox_to_anchor=(1, 1.03))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'image_vs_nonimage_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Image analysis complete. Visualization saved to:")
    print(f"- {os.path.join(plots_dir, 'image_vs_nonimage_accuracy.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze image vs. non-image performance")
    parser.add_argument("--base_dir", default="datasets", help="Base directory containing dataset folders")
    parser.add_argument("--include_datasets", nargs="+", help="Optional list of datasets to include")
    args = parser.parse_args()
    
    analyze_images(base_dir=args.base_dir, include_datasets=[
                # "The_Science_And_Engineering_of_Materials_Askeland",
                "Electronic_Magnetic_and_Optical_Materials_Fulay",
                "Materials_Science_and_Engineering_An_Introduction_Callister",
                "Polymer_Science_and_Technology_Fried",
                "Fundamentals_of_Ceramics_Barsoum",
                "Mechanical_Behavior_of_Materials_Hosford",
                "Physical_Metallurgy_Hosford",
                "Materials_and_Process_Selection_for_Engineering_Design_Farag",
                # "Introduction_to_Materials_Science_for_Engineers_Shackelford",
                # "Fundamentals_of_Materials_Instructors"
            ])