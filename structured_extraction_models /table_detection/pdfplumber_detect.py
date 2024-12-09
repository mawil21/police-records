import os
import json
import pdfplumber
import PyPDF2

input_dir = "../../tst/saved_pdfs"
tables_dir = "../../tst/table_detection_analysis/pdfplumber/output"
os.makedirs(tables_dir, exist_ok=True)

MAX_PAGES = 10

# Use os.walk to recurse through all subdirectories of input_dir
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
          

            
            # 2. Determine number of pages in the OCR’d PDF
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

            pages_to_process = min(MAX_PAGES, total_pages)

            pdf_tables_data = {
                "file": filename,
                "pages": []
            }

            # 3. Open the OCR’d PDF with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(pages_to_process):
                    page = pdf.pages[page_num]

                    # extract_tables tries to find table structures
                    tables = page.extract_tables()

                    page_tables = []
                    for t_index, table in enumerate(tables):
                        table_data = {
                            "table_index": t_index + 1,
                            "page_number": page_num + 1,
                            "cells": []
                        }

                        # table is a list of rows, each row is a list of cell texts
                        for row_index, row in enumerate(table):
                            for col_index, cell_text in enumerate(row):
                                cell_info = {
                                    "row": row_index,
                                    "col": col_index,
                                    "text": cell_text.strip() if cell_text else ""
                                }
                                table_data["cells"].append(cell_info)
                        page_tables.append(table_data)

                    pdf_tables_data["pages"].append({
                        "page_number": page_num + 1,
                        "tables": page_tables
                    })

            # 4. Save the extracted tables to JSON
            json_filename = f"{os.path.splitext(filename)[0]}_tables.json"
            json_path = os.path.join(tables_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_tables_data, f, indent=4)

            print(f"Processed {pdf_path}. Tables saved to {json_path}.")
