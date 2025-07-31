import os
import re
import glob
import csv
import argparse

def extract_examples(content, textbook=""):
    """Extract example problems from markdown content with robust handling of varied formats"""
    examples = []
    content = re.sub(r'## Page \d+.*?\n', '', content)
    
    # Pattern to match example problems but exclude DESIGN EXAMPLE
    example_pattern = r'(?:#{1,3}\s*(?<!DESIGN )(?:EXAMPLE|Example)|(?<!DESIGN )EXAMPLE|EXAMPLE PROBLEM|WORKED EXAMPLE) (\d+[-\.]\d+)(?::|)\s*(.*?)(?=(?:#{1,3}\s*(?<!DESIGN )(?:EXAMPLE|Example)|(?<!DESIGN )EXAMPLE|EXAMPLE PROBLEM)|\n# |\Z)'

    for match in re.finditer(example_pattern, content, re.DOTALL):
        example_id = match.group(1).strip()
        example_content = match.group(2).strip()
        
        # Get position information for context extraction
        start_pos = match.start()
        end_pos = match.end()
        
        # Extract reference context (approximately +/- 50 lines)
        lines = content.split('\n')
        
        # Find the line numbers where this example starts and ends
        line_positions = [0]
        current_pos = 0
        for i, line in enumerate(lines):
            current_pos += len(line) + 1  # +1 for the newline
            line_positions.append(current_pos)
        
        start_line = next(i for i, pos in enumerate(line_positions) if pos > start_pos) - 1
        end_line = next(i for i, pos in enumerate(line_positions) if pos > end_pos) - 1
        
        # Calculate context window (approximate +/- 50 lines)
        context_start = max(0, start_line - 50)
        context_end = min(len(lines), end_line + 50)
        
        # Extract the reference text
        reference_text = '\n'.join(lines[context_start:context_end])
        
        solution_match = re.search(r'#\s*(?:Solution|SOLUTION)(.*)', example_content, flags=re.DOTALL|re.IGNORECASE)
        
        if solution_match:
            problem = example_content[:solution_match.start()].strip().strip('#')
            solution = solution_match.group(1).strip()
            
            # Check for next heading that would indicate the end of this example
            next_heading_match = re.search(r'#{1,3} [A-Z]', solution)
            if next_heading_match:
                solution = solution[:next_heading_match.start()].strip()
        else:
            # If no solution marker is found, make a best guess
            parts = example_content.split('\n\n', 1)
            
            problem = parts[0].strip()
            solution = parts[1].strip() if len(parts) > 1 else ""
        
        examples.append({
            "example_id": f"Example {example_id}",
            "problem": problem,
            "solution": solution,
            "reference": reference_text
        })
    
    return examples


def process_markdown_files(directory, textbook):
    """Process markdown files from part_1, part_2, etc. directories"""
    all_examples = []
    
    # Find all part_* directories
    part_dirs = sorted(glob.glob(os.path.join(directory, "part_*")), 
                      key=lambda x: int(x.split('_')[-1]))
    
    print(f"Found {len(part_dirs)} part directories")
    
    for part_idx, part_dir in enumerate(part_dirs):
        part_num = int(part_dir.split('_')[-1])
        part_file = os.path.join(part_dir, f"part_{part_num}.md")
        
        if not os.path.exists(part_file):
            print(f"Warning: Expected file {part_file} not found, skipping...")
            continue
            
        print(f"Processing part {part_num}: {part_file}...")
        try:
            with open(part_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Extract examples from this file
            examples = extract_examples(content, textbook)
            
            # Add part number as part_id
            for ex in examples:
                ex['part_id'] = part_num
            
            all_examples.extend(examples)
            print(f"Found {len(examples)} examples")
        except Exception as e:
            print(f"Error processing {part_file}: {e}")
    
    textbook_name = directory.split("/")[-1]
    output_file = f"preprocess/extracted_qa/{textbook_name}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "problem", "solution", "part_id", "reference"])
        writer.writeheader()
        writer.writerows(all_examples)
    
    print(f"Saved {len(all_examples)} examples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract QA from markdown files")
    parser.add_argument("--directory", type=str, required=True,
                        help="Directory containing part_* directories.")
    parser.add_argument("--textbook", default="", type=str, required=False,
                        help="Textbook name.")
    args = parser.parse_args()

    process_markdown_files(args.directory, args.textbook)

if __name__ == "__main__":
    main()