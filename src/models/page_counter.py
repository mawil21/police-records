import os
import json
from PyPDF2 import PdfReader

def process_structured_extraction_directories(pdf_dir):
    directory_summary = {}
    error_files = []

    for root, _, files in os.walk(pdf_dir):
        pdf_count = 0
        pages_in_directory = 0

        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                try:
                    reader = PdfReader(pdf_path)
                    num_pages = len(reader.pages)

                    # Increment counts
                    pdf_count += 1
                    pages_in_directory += num_pages

                    # Save individual PDF info
                    if root not in directory_summary:
                        directory_summary[root] = {
                            "pdf_count": 0,
                            "total_pages": 0,
                            "pdf_details": []
                        }
                    directory_summary[root]["pdf_details"].append({
                        "pdf_name": file,
                        "page_count": num_pages
                    })
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
                    error_files.append(pdf_path)

        # Update directory totals
        if root in directory_summary:
            directory_summary[root]["pdf_count"] += pdf_count
            directory_summary[root]["total_pages"] += pages_in_directory

    return directory_summary, error_files


def save_to_json(data, error_files, output_file):
    output = {
        "directory_summary": data,
        "error_files": error_files
    }
    with open(output_file, "w") as json_file:
        json.dump(output, json_file, indent=4)
    print(f"Summary saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    directory = "../../tst/master_dataset"
    output_file = "directory_summary.json"

    directory_summary, error_files = process_structured_extraction_directories(directory)
    save_to_json(directory_summary, error_files, output_file)
