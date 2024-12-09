import os
import json

import PyPDF2
import tabula  # tabula-py


input_dir = "../../tst/saved_pdfs"

tables_dir = "../../tst/table_detection_analysis/tabula/output"

os.makedirs(tables_dir, exist_ok=True)

MAX_PAGES = 10

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)

            # Determine total pages of the OCR’d PDF
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

            pages_to_process = min(MAX_PAGES, total_pages)

            pdf_tables_data = {
                "file": filename,
                "pages": []
            }

            # 2. Use Tabula to extract tables from the OCR’d PDF
            # Tabula can extract tables page by page. We'll loop through pages:
            for page_num in range(1, pages_to_process + 1):
                # Extract tables from a single page using tabula
                # By default, tabula.read_pdf returns DataFrames for each table found on that page
                tables = tabula.read_pdf(
                    pdf_path,
                    pages=page_num,
                    multiple_tables=True,
                    lattice=True  # lattice=True tries line-based detection; you can also try stream=True
                )

                page_tables = []
                # tables is a list of DataFrames, each DF representing one detected table
                for t_index, df in enumerate(tables):
                    table_data = {
                        "table_index": t_index + 1,
                        "page_number": page_num,
                        "cells": []
                    }

                    # Convert DataFrame rows and columns to a cell structure
                    for row_index, row in df.iterrows():
                        for col_index, cell_value in enumerate(row):
                            cell_text = str(cell_value).strip() if cell_value is not None else ""
                            cell_info = {
                                "row": row_index,
                                "col": col_index,
                                "text": cell_text
                            }
                            table_data["cells"].append(cell_info)

                    page_tables.append(table_data)

                pdf_tables_data["pages"].append({
                    "page_number": page_num,
                    "tables": page_tables
                })

            # 3. Save the extracted tables to JSON
            json_filename = f"{os.path.splitext(filename)[0]}_tables.json"
            json_path = os.path.join(tables_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_tables_data, f, indent=4)

            print(f"Processed {pdf_path}. Tables saved to {json_path}.")
