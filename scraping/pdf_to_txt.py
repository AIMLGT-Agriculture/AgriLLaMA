import os
from pdfminer.high_level import extract_text
# import fitz

# def extract_text(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text()
#     return text
# from langdetect import detect

# def is_english(text):
#     try:
#         return detect(text) == 'en'
#     except:
#         return False

def extract_text_from_pdfs(input_dir, output_dir):
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
                    text = extract_text(pdf_path)
                    # if is_english(text):
                    # print(f"The text in {pdf_path} is likely English")
                    with open(txt_path, "w", encoding="utf-8") as text_file:
                        text_file.write(text)
                    print(f"Text extracted from {pdf_path} to {txt_path}")
                # else:
                #     print(f"The text in {pdf_path} may not be English")
                    
                except Exception as e:
                    print(f"Failed to extract text from {pdf_path}: {e}")

    # Return the total number of PDFs processed
    return pdf_count

# Set the directory containing the PDFs and the directory to save the text files
input_directory = "./icar_journals"  # Replace with your PDFs folder path
output_directory = "./text_data"  # Replace with your desired output folder path
number_of_pdfs = extract_text_from_pdfs(input_directory, output_directory)
print(f"Total number of PDFs: {number_of_pdfs}")