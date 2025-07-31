import os
import pandas as pd
import glob
import re
from collections import defaultdict
import numpy as np


def collect_results_from_file(csv_file: str):
    """Parse one results CSV and return metadata with DataFrame."""
    filename = os.path.basename(csv_file)
    if filename == "qa.csv":
        return None

    # Handle both old format (model_method.csv) and new format with timestamp (model_method_timestamp.csv)
    # Try with timestamp first: model_method_YYYYMMDD_HHMMSS.csv
    m = re.match(r"(.+)_(\w+)_(\d{8})_(\d{6})\.csv", filename)
    if m:
        model, method, date_part, time_part = m.groups()
        timestamp = f"{date_part}_{time_part}"
        run_id = timestamp
    else:
        # Try without timestamp: model_method.csv
        m = re.match(r"(.+)_(\w+)\.csv", filename)
        if not m:
            return None
        model, method = m.groups()
        timestamp = None
        run_id = "single_run"

    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        needed = {"is_correct", "new_token_nums"}
        if df.empty or not needed.issubset(df.columns):
            return None
        return {"model": model, "method": method, "run_id": run_id, "timestamp": timestamp, "df": df}
    except Exception as e:
        print(f"Error reading {csv_file}: {str(e)}")
        return None

def load_evaluation_data(base_dir="results", difficulty_file=None, include_datasets=None):
    """Load and process all evaluation data from dataset directories."""
    print("Loading evaluation data...")
    
    # Load question data which already contains difficulty and category information
    if difficulty_file:
        qa_file = difficulty_file
    else:
        qa_file = os.path.join("datasets", "MatSciBench", "qa.csv")
    try:
        qa_df = pd.read_csv(qa_file)
        print(f"Loaded question data for {len(qa_df)} questions")
        
    except Exception as e:
        print(f"Error loading question data: {str(e)}")
        qa_df = pd.DataFrame()
    
    all_dfs = []
    
    
    # Alternative flat directory structure (base_dir/evaluation/*.csv)
    eval_dir = os.path.join(base_dir, "evaluation")
    if os.path.exists(eval_dir):
        print(f"Processing files in {eval_dir}")
        for csv_file in glob.glob(os.path.join(eval_dir, "*.csv")):
            rec = collect_results_from_file(csv_file)
            if rec is None:
                continue
            
            # Add metadata columns
            df = rec["df"].copy()
            # For flat structure, use "MatSciBench" as the default source
            df['source'] = "MatSciBench"
            df['model'] = rec["model"]
            df['method'] = rec["method"]
            df['run_id'] = rec["run_id"]
            df['timestamp'] = rec["timestamp"]
            
            
            # Add information from qa_df
            if not qa_df.empty:
                df = pd.merge(df, qa_df, on=["qid"], how="left", suffixes=("", "_qa"))
            
            all_dfs.append(df)

    if not all_dfs:
        print("No valid data found")
        return pd.DataFrame(), qa_df

    # Concatenate all dataframes
    results_df = pd.concat(all_dfs, ignore_index=True)
    
    # Set pandas display options to show full qid
    pd.set_option('display.max_colwidth', None)
    print("final qid: ", results_df['qid'].head())
    return results_df, qa_df

def print_qa_statistics(df):
    """Print statistics about the qa data"""
    print("\n" + "="*50)
    print(" QA DATA STATISTICS ")
    print("="*50)
    
    # Basic counts
    print(f"\nTotal records: {len(df)}")
    print(f"Unique sources: {df['source'].nunique()}")
    
    # Difficulty level distribution if available
    if 'difficulty_level' in df.columns:
        print("\nDifficulty level distribution:")
        difficulty_counts = df['difficulty_level'].value_counts()
        for level, count in difficulty_counts.items():
            print(f"  {level}: {count} ({count/len(df)*100:.2f}%)")
    
    # Steps count statistics if available
    if 'steps_count' in df.columns:
        print("\nSteps count statistics:")
        steps_stats = df['steps_count'].describe()
        print(f"  Mean: {steps_stats['mean']:.2f}")
        print(f"  Min: {steps_stats['min']}")
        print(f"  Max: {steps_stats['max']}")
    
    # Primary category distribution if available
    if 'primary_category' in df.columns:
        print("\nPrimary category distribution:")
        category_counts = df['primary_category'].value_counts().head(10)
        for category, count in category_counts.items():
            if pd.notna(category):
                print(f"  {category}: {count}")
    
    # Extract and analyze individual categories from parsed vectors
    if 'category_vector_parsed' in df.columns:
        print("\nCategory vector analysis:")
        category_fields = ['Materials', 'Properties', 'Structures', 
                          'Fundamental Mechanisms', 'Processes', 'Failure Mechanisms']
        
        # Count occurrences of each category
        category_counts = {}
        for _, row in df.iterrows():
            if isinstance(row['category_vector_parsed'], dict):
                for cat, val in row['category_vector_parsed'].items():
                    if cat not in category_counts:
                        category_counts[cat] = defaultdict(int)
                    category_counts[cat][val] += 1
        
        # Print category statistics
        for cat in category_fields:
            if cat in category_counts:
                print(f"\n  {cat} distribution:")
                sorted_items = sorted(category_counts[cat].items(), key=lambda x: x[1], reverse=True)
                for val, count in sorted_items[:5]:  # Show top 5
                    print(f"    {val}: {count}")
    
    print("="*50)

def print_statistics(df):
    """Print statistics about the evaluation results"""
    print("\n" + "="*50)
    print(" RESULTS STATISTICS ")
    print("="*50)
    
    # Basic counts
    print(f"\nTotal records: {len(df)}")
    print(f"Unique datasets: {df['source'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique methods: {df['method'].nunique()}")
    
    # Questions count by model and method
    print("\nQuestion counts by model and method:")
    model_method_counts = df.groupby(['model', 'method']).size().reset_index(name='count')
    for _, row in model_method_counts.iterrows():
        print(f"  {row['model']}, {row['method']}: {row['count']}")
    
    # Difficulty levels if available
    if 'difficulty_level' in df.columns:
        print("\nPerformance by difficulty level:")
        diff_levels = df.groupby('difficulty_level')
        for level, group in diff_levels:
            if level and pd.notna(level):
                acc = group['is_correct'].mean() if 'is_correct' in group.columns else None
                count = len(group)
                print(f"  {level} (n={count}): {acc:.4f}" if acc is not None else f"  {level} (n={count}): N/A")
    
    # Category statistics if available
    category_fields = ['Materials', 'Properties', 'Structures', 
                      'Fundamental Mechanisms', 'Processes', 'Failure Mechanisms']
    
    available_categories = [f for f in category_fields if f in df.columns]
    if available_categories:
        print("\nCategory statistics:")
        for category in available_categories:
            unique_values = df[df[category].notna()][category].nunique()
            if unique_values > 0:
                print(f"  {category}: {unique_values} unique values")
                top_values = df[category].value_counts().head(3)
                print("    Top values:")
                for val, count in top_values.items():
                    if pd.notna(val):
                        print(f"      {val}: {count}")
    
    print("="*50)

if __name__ == "__main__":
    # When run directly, load data and print statistics
    results_df, qa_df = load_evaluation_data()
    
    if not results_df.empty:
        print_statistics(results_df)
    
    if not qa_df.empty:
        print_qa_statistics(qa_df)
    else:
        print("No results data available for analysis")
