import os
import argparse

def analyze_overall_stats(base_dir="datasets", include_datasets=None):
    """Generate overall statistics summary"""
    # Import here to keep module independent
    from data_processor import load_evaluation_data
    
    # Load data
    results_df, image_df, stats, model_summary, results_output_dir, plots_dir = load_evaluation_data(
        base_dir=base_dir, include_datasets=include_datasets
    )
    
    # Unpack aggregated statistics
    total_records = stats["total_records"]
    total_correct = stats["total_correct"]
    dataset_question_counts = stats["dataset_question_counts"]
    model_question_counts = stats["model_question_counts"]
    image_question_count = stats["image_question_count"]
    non_image_question_count = stats["non_image_question_count"]
    
    # Generate overall statistics summary
    overall_stats = {
        'Total Questions': total_records,
        'Total Correct Answers': total_correct,
        'Overall Accuracy': (total_correct / total_records * 100) if total_records > 0 else 0,
        'Number of Datasets': len(dataset_question_counts),
        'Number of Models': len(model_question_counts),
        'Questions with Images': image_question_count,
        'Questions without Images': non_image_question_count
    }
    
    # Save overall statistics
    with open(os.path.join(results_output_dir, 'overall_statistics.txt'), 'w') as f:
        f.write("Overall Evaluation Statistics\n")
        f.write("===========================\n\n")
        for stat, value in overall_stats.items():
            if isinstance(value, float):
                f.write(f"{stat}: {value:.2f}\n")
            else:
                f.write(f"{stat}: {value}\n")
        
        f.write("\nQuestions per Dataset:\n")
        for dataset, count in sorted(dataset_question_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {dataset}: {count} questions ({count/total_records*100:.1f}%)\n")
        
        f.write("\nQuestions per Model:\n")
        for model, count in sorted(model_question_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model}: {count} questions ({count/total_records*100:.1f}%)\n")
    
    # Print overall model ranking
    print("\nModel Performance Ranking:")
    print("-" * 80)
    ranking = model_summary.sort_values('overall_accuracy', ascending=False)
    
    for i, row in enumerate(ranking.itertuples(), 1):
        print(f"{i}. {row.model} ({row.method}): {row.overall_accuracy:.2f}% accuracy, " 
              f"{row.avg_tokens:.2f} avg tokens")
              
    print(f"Overall statistics saved to {os.path.join(results_output_dir, 'overall_statistics.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate overall evaluation statistics")
    parser.add_argument("--base_dir", default="datasets", help="Base directory containing dataset folders")
    parser.add_argument("--include_datasets", nargs="+", help="Optional list of datasets to include")
    args = parser.parse_args()
    
    analyze_overall_stats(base_dir=args.base_dir, include_datasets=args.include_datasets)