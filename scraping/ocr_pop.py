import os
from pdfminer.high_level import extract_text
from langdetect import detect
import os
import subprocess

def ocr_pdf(input_dir, output_dir):
    # Initialize a counter for the PDF files
    pdf_count = 0

    # Check and create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                # Increment the counter for each PDF found
                pdf_count += 1

                # Construct the file paths
                pdf_path = os.path.join(root, file)
                rel_path = os.path.relpath(pdf_path, input_dir)
                txt_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".txt")
                
                # Skip extraction if the text file already exists
                if os.path.exists(txt_path):
                    print(f"Text file already exists for {pdf_path}, skipping extraction.")
                    continue
                
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)

                # Extract text from the PDF and write it to the text file
                

                try:
                    print(f"{txt_path} {pdf_path}")
                    # break
                    subprocess.run(["ocrmypdf","-l", "eng+kan", "--force-ocr", "--sidecar", txt_path, pdf_path, pdf_path ])
                    # os.system(f"ocrmypdf -l eng+kan --force-ocr --sidecar {txt_path} {pdf_path} {pdf_path}")
                except Exception as e:
                    print(f"Failed to extract text from {pdf_path}: {e}")
        

    # Return the total number of PDFs processed
    return pdf_count

input_directory = "./kvk_pop/KARNATAKA"  # Replace with your PDFs folder path
output_directory = "./pop_ocr/KARNATAKA"  # Replace with your desired output folder path
number_of_files = ocr_pdf(input_directory, output_directory)
print(f"Total number of files: {number_of_files}")