import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def analyze_models(base_dir="datasets", include_datasets=None):
    """Analyze and visualize model performance comparison"""
    # Import here to keep module independent
    from data_processor import load_evaluation_data
    
    # Load data
    results_df, image_df, stats, model_summary, results_output_dir, plots_dir = load_evaluation_data(
        base_dir=base_dir, include_datasets=include_datasets
    )
    
    # Get model ranking
    ranking = model_summary.sort_values('overall_accuracy', ascending=False)
    
    # Plot: Overall accuracy by model with better colors
    plt.figure(figsize=(14, 8))
    sns.set_palette("viridis", len(ranking['method'].unique()))
    acc_plot = sns.barplot(x='model', y='overall_accuracy', hue='method', data=ranking)
    plt.title('Model Accuracy Comparison by Method')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_accuracy.png'))
    plt.close()
    
    print(f"Model comparison analysis complete. Saved to {os.path.join(plots_dir, 'model_accuracy.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model performance comparison")
    parser.add_argument("--base_dir", default="datasets", help="Base directory containing dataset folders")
    parser.add_argument("--include_datasets", nargs="+", help="Optional list of datasets to include")
    args = parser.parse_args()
    
    analyze_models(base_dir=args.base_dir, include_datasets=args.include_datasets)