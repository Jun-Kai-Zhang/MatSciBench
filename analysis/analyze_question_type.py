import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def analyze_question_types(base_dir="datasets", include_datasets=None):
    """Analyze performance by question type"""
    # Import here to keep module independent
    from data_processor import load_question_type_data, load_evaluation_data
    
    # We need to get the output directories from the evaluation data loader
    _, _, _, _, results_output_dir, plots_dir = load_evaluation_data(
        base_dir=base_dir, include_datasets=include_datasets
    )
    
    # Load question type data
    type_df = load_question_type_data(base_dir=base_dir, include_datasets=include_datasets)
    
    if type_df is None:
        print("No question type data available for analysis")
        return
        
    # Plot question type performance
    plt.figure(figsize=(14, 10))
    q_type_pivot = type_df.pivot_table(
        values='accuracy', 
        index='model',
        columns='question_type',
        aggfunc='mean'
    )
    sns.heatmap(q_type_pivot, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Model Performance by Question Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'question_type_heatmap.png'))
    plt.close()
    
    print(f"Question type analysis complete. Visualization saved to {os.path.join(plots_dir, 'question_type_heatmap.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze performance by question type")
    parser.add_argument("--base_dir", default="datasets", help="Base directory containing dataset folders")
    parser.add_argument("--include_datasets", nargs="+", help="Optional list of datasets to include")
    args = parser.parse_args()
    
    analyze_question_types(base_dir=args.base_dir, include_datasets=args.include_datasets)