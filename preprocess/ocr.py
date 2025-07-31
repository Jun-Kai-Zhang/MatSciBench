import os
from PyPDF2 import PdfReader, PdfWriter
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk, OCRResponse
import json
from pathlib import Path
import time
import base64
import argparse
from glob import glob
import re


api_key = os.getenv("MISTRAL_API_KEY")
if api_key is None:
    raise ValueError("MISTRAL_API_KEY environment variable is not set")
client = Mistral(api_key=api_key)


def split_pdf_by_pages(input_path, output_folder, pages_per_chunk=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    pdf = PdfReader(input_path)
    total_pages = len(pdf.pages)
    
    num_output_files = (total_pages + pages_per_chunk - 1) // pages_per_chunk
    
    print(f"Splitting {input_path} ({total_pages} pages) into {num_output_files} files...")
    for i in range(num_output_files):
        output = PdfWriter()
        start_page = i * pages_per_chunk
        end_page = min((i + 1) * pages_per_chunk, total_pages)
        for page_num in range(start_page, end_page):
            output.add_page(pdf.pages[page_num])
        output_path = os.path.join(output_folder, f"part_{i+1}.pdf")
        with open(output_path, "wb") as output_file:
            output.write(output_file)
        
        print(f"Created {output_path} with pages {start_page+1}-{end_page}")


def ocr_pdf(pdf_split_folder, ocr_output_dir):
    # Convert to Path objects if they're not already
    pdf_split_folder = Path(pdf_split_folder)
    ocr_output_dir = Path(ocr_output_dir)

    pdf_files = [f for f in os.listdir(pdf_split_folder) if f.endswith('.pdf')]
    pdf_files.sort()
    
    for _, pdf_file_name in enumerate(pdf_files):
        # Convert to Path object
        split_index = int(pdf_file_name.split("_")[-1].split(".")[0])
        pdf_file = pdf_split_folder / pdf_file_name
        print(f"Processing {pdf_file}...")
        
        ocr_dir = ocr_output_dir / f"part_{split_index}"
        ocr_dir.mkdir(exist_ok=True)

        try:
            # Upload file
            uploaded_file = client.files.upload(
                file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
                purpose="ocr"
            )
            
            # Get signed URL and process
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            images_dir = ocr_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            image_map = {}
            for page_num, page in enumerate(pdf_response.pages, 1):
                for img_idx, img in enumerate(page.images):
                    img_filename = img.id
                    img_path = images_dir / img_filename
                    try:
                        # Clean the base64 string - remove any potential header information
                        base64_data = img.image_base64
                        # If the base64 string contains a data URI prefix, remove it
                        if ',' in base64_data:
                            base64_data = base64_data.split(',', 1)[1]
                        # Decode the base64 data
                        img_data = base64.b64decode(base64_data)
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        image_map[img.id] = img_path.relative_to(ocr_dir)
                        
                    except Exception as img_error:
                        print(f"Error saving image {img.id}: {img_error}")
            
            # Save JSON response
            response_dict = json.loads(pdf_response.json())
            with open(ocr_dir / f"{pdf_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(response_dict, f, indent=4)
            
            # Save markdown content
            with open(ocr_dir / f"{pdf_file.stem}.md", "w", encoding="utf-8") as f:
                for page_num, page in enumerate(pdf_response.pages, 1):
                    f.write(f"## Page {(split_index-1)*100 + page_num}\n\n{page.markdown}\n\n")
            
            print(f"Successfully processed {pdf_file}")
            time.sleep(2)  # Avoid rate limits
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
        
        print(f"Completed file {pdf_file_name}")

    print("All processing complete!")


def extract_image_paths(markdown_file):
    """Extract all image references from a markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all image references in markdown format: ![alt](path)
    image_pattern = r'!\[.*?\]\((.*?)\)'
    image_paths = re.findall(image_pattern, content)
    
    # Extract just the filename from each path
    image_filenames = [os.path.basename(path) for path in image_paths]
    
    return image_filenames

def update_markdown_file(markdown_file, new_path_prefix):
    """Update image paths in the markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace image paths with new paths
    updated_content = re.sub(
        r'!\[(.*?)\]\((.*?)\)',
        lambda m: f'![{m.group(1)}]({new_path_prefix}/{os.path.basename(m.group(2))})',
        content
    )
    
    # Write the updated content back to the file
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Updated image paths in {markdown_file}")

def process_part_directory(part_dir):
    """Process a single part directory."""
    # Find the markdown file in this part directory
    markdown_files = glob(os.path.join(part_dir, "*.md"))
    
    if not markdown_files:
        print(f"No markdown files found in {part_dir}")
        return
    
    for markdown_file in markdown_files:
        print(f"Processing {markdown_file}...")
        
        # Extract image filenames from the markdown file
        image_filenames = extract_image_paths(markdown_file)
        print(f"Found {len(image_filenames)} images in {markdown_file}")
        
        if not image_filenames:
            print(f"No images to process in {markdown_file}")
            continue
        
        # Update markdown file with new image paths
        part_name = os.path.basename(part_dir)
        new_path_prefix = f"{part_name}/images"
        update_markdown_file(markdown_file, new_path_prefix)
    
    print("\nAll part directories processed successfully!")

def update_image_paths(ocr_output_dir):
        # Find all part directories
    all_parts = glob(os.path.join(ocr_output_dir, "part_*"))
    
    if not all_parts:
        print(f"No part directories found in {ocr_output_dir}")
        return 1
    
    print(f"Found {len(all_parts)} part directories")
    # Process each part directory
    for part_dir in all_parts:
        print(f"\nProcessing directory: {part_dir}")
        process_part_directory(part_dir)


def main():
    parser = argparse.ArgumentParser(description="Run OCR on a PDF.")
    parser.add_argument("--textbook", type=str, required=True,
                        help="Textbook name.")
    args = parser.parse_args()

    input_pdf = args.textbook
    textbook_name = args.textbook.split("/")[-1].split(".")[0]
    
    pdf_split_folder = f"preprocess/textbook_pdfs/{textbook_name}/"
    if not os.path.exists(pdf_split_folder):
        os.makedirs(pdf_split_folder)
    split_pdf_by_pages(input_pdf, pdf_split_folder, 100)
    
    ocr_output_dir = f"preprocess/textbook_ocr/{textbook_name}/"
    if not os.path.exists(ocr_output_dir):
        os.makedirs(ocr_output_dir)
    ocr_pdf(pdf_split_folder, ocr_output_dir)
    update_image_paths(ocr_output_dir)


if __name__ == "__main__":
    main()