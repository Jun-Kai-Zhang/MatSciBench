import os
import pandas as pd
import shutil
from pathlib import Path
import glob
import argparse
import re

def extract_numeric_part(example_id):
    """Extract numeric parts from example_id for hierarchical sorting"""
    # Extract all numbers from the string
    numbers = re.findall(r'\d+', str(example_id))
    if not numbers:
        return (0, 0)  # Default value if no numbers found
    
    # For hierarchical sorting (chapter-example)
    if len(numbers) >= 2:
        return (int(numbers[0]), int(numbers[1]))  # Return as tuple for sorting
    else:
        return (int(numbers[0]), 0)  # If only one number is found

def process_textbook_data(book_name, source_csv=None, source_img_base=None, dest_dir=None):
    # Define paths with defaults if not provided
    if source_csv is None:
        source_csv = f"preprocess/revised_qa/{book_name}.csv"
    if source_img_base is None:
        source_img_base = f"preprocess/textbook_ocr/{book_name}"
    if dest_dir is None:
        dest_dir = f"datasets/{book_name}"
    
    dest_img_dir = f"{dest_dir}/images"
    
    # Create destination directories
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(dest_img_dir, exist_ok=True)
    
    # Read the old CSV
    print(f"Reading source CSV: {source_csv}")
    df = pd.read_csv(source_csv)
    
    # Sort by numeric part of example_id
    print(f"Sorting data by numeric part of example_id...")
    df['numeric_part'] = df['example_id'].apply(extract_numeric_part)
    df = df.sort_values(by='numeric_part')
    
    # Handle duplicates by adding postfix
    duplicate_groups = df.groupby('numeric_part')
    new_example_ids = []
    
    for _, group in df.iterrows():
        num_part = group['numeric_part']
        group_size = len(df[df['numeric_part'] == num_part])
        
        if group_size > 1:
            # Find position in the group
            position = len([x for x in new_example_ids if x == num_part])
            # Add postfix to the example_id
            new_id = f"{group['example_id']}_{position + 1}"
            new_example_ids.append(num_part)
        else:
            new_id = group['example_id']
        
        df.loc[_, 'example_id'] = new_id
    
    # Drop the temporary column
    df = df.drop(columns=['numeric_part'])
    
    # Create the new dataframe with the required columns
    # Initialize with empty values for new columns
    new_df = pd.DataFrame({
        'qid': df['example_id'],
        'domain': [""] * len(df),
        'type': df['answer_type'],  # Default type
        'question': df['question'],
        'image': "",  # Will update with image paths
        'solution': df['solution'],
        'answer': df['answer'],
        'unit': df.get('unit', ""),  # Use empty string if column doesn't exist
        'answer_type': df.get('answer_type', ""),
        'notes': [""] * len(df),
        'number_of_answers': df.get('number_of_answers', 1)  # Default to 1 if not present
    })
    
    # Process images
    print(f"Processing images from {source_img_base}...")
    image_count = 0
    
    # Track original to new image path mapping for updating references
    image_path_mapping = {}
    
    # First, collect all image paths referenced in the CSV
    referenced_images = []
    for i, row in df.iterrows():
        # Process question_image field
        if pd.notna(row.get('question_image')):
            # Split by comma to handle multiple images
            img_paths = [img.strip() for img in row['question_image'].split(',')]
            referenced_images.extend(img_paths)
            
        # Process solution_image field
        if pd.notna(row.get('solution_image')):
            # Split by comma to handle multiple images
            img_paths = [img.strip() for img in row['solution_image'].split(',')]
            referenced_images.extend(img_paths)
    
    # Remove duplicates
    referenced_images = list(set(referenced_images))
    
    print(f"Found {len(referenced_images)} unique images referenced in the CSV")
    
    # Process only the referenced images
    for img_rel_path in referenced_images:
        # Construct the full path to the source image
        img_full_path = os.path.join(source_img_base, img_rel_path)
        
        if os.path.exists(img_full_path):
            # Extract part name and filename
            path_parts = Path(img_rel_path).parts
            part_name = path_parts[0]  # e.g., "part_1"
            img_filename = path_parts[-1]  # e.g., "img-72.jpeg"
            
            # Create new unique filename
            new_img_filename = f"{part_name}_{img_filename}"
            dest_img_path = os.path.join(dest_img_dir, new_img_filename)
            
            # Store mapping
            image_path_mapping[img_rel_path] = new_img_filename
            
            # Copy the image
            shutil.copy2(img_full_path, dest_img_path)
            image_count += 1
        else:
            print(f"Warning: Referenced image not found: {img_full_path}")
    
    # Update image references in the dataframe
    for i, row in df.iterrows():
        image_refs = []
        
        # Process question_image field
        if pd.notna(row.get('question_image')):
            question_imgs = [img.strip() for img in row['question_image'].split(',')]
            for original_img in question_imgs:
                if original_img in image_path_mapping:
                    new_img_filename = image_path_mapping[original_img]
                    image_refs.append(f"{book_name}/images/{new_img_filename}")
                else:
                    print(f"Warning: Could not map question image: {original_img}")
        
        # Process solution_image field
        if pd.notna(row.get('solution_image')):
            solution_imgs = [img.strip() for img in row['solution_image'].split(',')]
            for original_img in solution_imgs:
                if original_img in image_path_mapping:
                    new_img_filename = image_path_mapping[original_img]
                    image_refs.append(f"{book_name}/images/{new_img_filename}")
                else:
                    print(f"Warning: Could not map solution image: {original_img}")
        
        # Join all image references with comma
        if image_refs:
            new_df.at[i, 'image'] = ','.join(image_refs)
    
    # Save the new CSV
    output_csv = f"{dest_dir}/qa.csv"
    new_df.to_csv(output_csv, index=False)
    
    print(f"Processing complete!")
    print(f"- Created new directory: {dest_dir}")
    print(f"- Copied {image_count} images to {dest_img_dir}")
    print(f"- Created new CSV: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process textbook data and organize into dataset format')
    parser.add_argument('--csv', 
                      help='Path to source CSV file (default: preprocess/revised_qa/{book}.csv)')
    args = parser.parse_args()
    
    book_name = args.csv.split('/')[-1].split('.')[0]
    images_dir = f"preprocess/textbook_ocr/{book_name}"  
    dest_dir = f"datasets/{book_name}"

    process_textbook_data(
        book_name=book_name,
        source_csv=args.csv,
        source_img_base=images_dir,
        dest_dir=dest_dir
    ) 