import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

def analyze_datasets(base_dir="datasets", include_datasets=None):
    """Analyze dataset distribution and model performance by dataset"""
    # Import here to keep module independent
    from data_processor import load_evaluation_data
    
    # Load data
    results_df, image_df, stats, model_summary, results_output_dir, plots_dir = load_evaluation_data(
        base_dir=base_dir, include_datasets=include_datasets
    )
    
    # Create dataset distribution plot
    plt.figure(figsize=(12, 8))
    dataset_df = pd.DataFrame({
        'dataset': list(stats["dataset_question_counts"].keys()),
        'count': list(stats["dataset_question_counts"].values())
    })
    dataset_df = dataset_df.sort_values('count', ascending=False)
    sns.barplot(x='dataset', y='count', data=dataset_df)
    plt.title('Question Distribution Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'dataset_distribution.png'))
    plt.close()
    
    # Dataset performance heatmap
    if len(results_df) > 0:
        plt.figure(figsize=(16, 12))
        # Create a combined model-method column for better visualization
        results_df['model_method'] = results_df['model'] + '\n(' + results_df['method'] + ')'
        pivot_df = results_df.pivot_table(values='accuracy', 
                                          index='model_method', 
                                          columns='dataset')
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Model-Method Performance by Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'dataset_heatmap.png'))
        plt.close()
    
    print(f"Dataset analysis complete. Visualizations saved to:")
    print(f"- {os.path.join(plots_dir, 'dataset_distribution.png')}")
    print(f"- {os.path.join(plots_dir, 'dataset_heatmap.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset distributions and performance")
    parser.add_argument("--base_dir", default="datasets", help="Base directory containing dataset folders")
    parser.add_argument("--include_datasets", nargs="+", help="Optional list of datasets to include")
    args = parser.parse_args()
    
    analyze_datasets(base_dir=args.base_dir, include_datasets=args.include_datasets)