import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Polygon

##########################
# Standardization Helpers
##########################

def is_within(inner_bbox, outer_bbox):
    """
    Returns True if inner_bbox (list of (x, y) tuples) is completely contained within outer_bbox.
    """
    return Polygon(inner_bbox).within(Polygon(outer_bbox))

def get_top(bbox):
    """Return the smallest y coordinate from the bbox."""
    return min(pt[1] for pt in bbox)

def get_bottom(bbox):
    """Return the largest y coordinate from the bbox."""
    return max(pt[1] for pt in bbox)

###############################
# Combine Page Data Functions
###############################

def combine_page_data(page_tables, page_kv_pairs, page_lines):
    """
    Combine table and key–value pair data for a single page.
    
    A kv_pair is included only if its bounding box is not completely contained within any table's bounding box.
    For each structured data item (table or kv_pair), context lines ("k-lines")
    from page_lines (those lines whose bottom is above the item's top) are collected.
    Additionally, any context line whose text appears in any structured content is removed.
    """
    # Filter out kv_pair items that lie completely inside any table's bbox.
    filtered_kv = []
    for kv in page_kv_pairs:
        contained = False
        for table in page_tables:
            if is_within(kv['bbox'], table['bbox']):
                contained = True
                break
        if not contained:
            filtered_kv.append(kv)
    
    combined_data = page_tables + filtered_kv
    combined_data.sort(key=lambda item: get_top(item['bbox']))
    
    # Collect structured texts so they can be excluded from k-lines.
    structured_texts = [item.get("content", "").strip() for item in (page_tables + page_kv_pairs)]
    
    remaining_lines = page_lines.copy()
    final_output = []
    for data in combined_data:
        region_top = get_top(data['bbox'])
        context_lines = []
        new_remaining = []
        for line in remaining_lines:
            if get_bottom(line['bbox']) < region_top:
                context_lines.append(line['content'].strip())
            else:
                new_remaining.append(line)
        remaining_lines = new_remaining
        # Remove any context line that appears in any structured text.
        filtered_context = [
            line for line in context_lines
            if not any(line and (line in s) for s in structured_texts)
        ]
        final_output.append({
            "page_no": data['page_no'],
            "k-lines": "\n".join(filtered_context),
            "type": data['type'],
            "content": data['content'],
            "bbox": data['bbox'],
            
        })
    return final_output

##############################
# Standardize Input Functions
##############################

def standardize_table_item(item):
    """
    Convert a table item from the Document Intelligence JSON format to a uniform format.
    Assumes:
      - "page_no" or "page_number"
      - "table_bounding_region" with "point.x" and "point.y"
      - "cells": list of cell objects
    """
    page = item.get("page_no") or item.get("page_number")
    try:
        page = int(page)
    except Exception:
        pass
    region = item.get("table_bounding_region", {})
    pts = list(zip(region.get("point.x", []), region.get("point.y", [])))
    table_dict = {}
    for cell in item.get("cells", []):
        row = cell.get("rowIndex", 0)
        col = cell.get("columnIndex", 0)
        content = cell.get("content", "")
        if row not in table_dict:
            table_dict[row] = {}
        table_dict[row][col] = content
    max_col = max((max(d.keys()) for d in table_dict.values()), default=0)
    rows = []
    for r in sorted(table_dict.keys()):
        row_cells = [table_dict[r].get(c, "") for c in range(max_col + 1)]
        rows.append("\t".join(row_cells))
    table_content = "\n".join(rows)
    return {
        "page_no": page,
        "type": "table",
        "bbox": pts,
        "content": table_content
    }

def standardize_kv_item(item):
    """
    Convert a key–value pair item from the Document Intelligence JSON format to a uniform format.
    Assumes:
      - "page_number"
      - "key_bounding_box" and "value_bounding_box"
      - "key" and "value"
    """
    page = item.get("page_number")
    try:
        page = int(page)
    except Exception:
        pass
    def pair_coords(bbox):
        return list(zip(bbox.get("point.x", []), bbox.get("point.y", [])))
    key_bbox = pair_coords(item.get("key_bounding_box", {}))
    value_bbox = pair_coords(item.get("value_bounding_box", {}))
    if value_bbox and len(value_bbox) >= 4 and key_bbox and len(key_bbox) >= 4:
        combined_bbox = [key_bbox[0], value_bbox[1], value_bbox[2], key_bbox[3]]
    else:
        combined_bbox = key_bbox
    key_content = item.get("key", "")
    value_content = item.get("value", "")
    combined_content = key_content if not value_content else f"{key_content} {value_content}"
    return {
        "page_no": page,
        "type": "kv_pair",
        "bbox": combined_bbox,
        "content": combined_content
    }

def standardize_line_item(page_item):
    """
    Standardize a lines JSON page item.
    Assumes:
      - "page_no" or "page_number"
      - "content": list of objects with "text" and "bbox"
    """
    page = page_item.get("page_no") or page_item.get("page_number")
    try:
        page = int(page)
    except Exception:
        pass
    standardized_lines = []
    for line in page_item.get("content", []):
        bbox_dict = line.get("bbox", {})
        pts = list(zip(bbox_dict.get("point.x", []), bbox_dict.get("point.y", [])))
        standardized_lines.append({
            "page_no": page,
            "bbox": pts,
            "content": line.get("text", "")
        })
    return standardized_lines

def load_and_standardize_tables(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    tables = data.get("tables", [])
    return [standardize_table_item(item) for item in tables]

def load_and_standardize_kv(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [standardize_kv_item(item) for item in data]

def load_and_standardize_lines(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    standardized = []
    for page_item in data:
        standardized.extend(standardize_line_item(page_item))
    return standardized

def group_by_page(items, page_field="page_no"):
    groups = {}
    for item in items:
        page = item.get(page_field)
        try:
            page = int(page)
        except Exception:
            continue
        groups.setdefault(page, []).append(item)
    return groups

####################################
# Combine All JSON Files for a File
####################################

def combine_all_json_files(table_file_path, kv_file_path, lines_file_path):
    """
    Load and standardize three JSON files (tables, key–value pairs, and lines),
    group the items by page, and then combine the data on each page.
    Returns a dictionary keyed by page number.
    """
    standardized_tables = load_and_standardize_tables(table_file_path)
    standardized_kv = load_and_standardize_kv(kv_file_path)
    standardized_lines = load_and_standardize_lines(lines_file_path)
    
    tables_by_page = group_by_page(standardized_tables)
    kv_by_page = group_by_page(standardized_kv)
    lines_by_page = group_by_page(standardized_lines)
    
    all_pages = set(tables_by_page.keys()) | set(kv_by_page.keys()) | set(lines_by_page.keys())
    all_pages = sorted(all_pages)
    
    combined_results = {}
    for page in all_pages:
        page_tables = tables_by_page.get(page, [])
        page_kv = kv_by_page.get(page, [])
        page_lines = lines_by_page.get(page, [])
        combined_results[page] = combine_page_data(page_tables, page_kv, page_lines)
    return combined_results

###################################
# Multi-File Processing Functions
###################################

def process_single_file(sub_dir, file, table_dir, kv_pair_dir, lines_dir, output_extraction_dir):
    """
    Process a single file.
    Constructs full paths for the table, kv_pair, and lines JSON files based on sub_dir and file name.
    Calls combine_all_json_files to generate the combined structured data, and saves the output.
    """
    full_table_path = os.path.join(table_dir, sub_dir, file)
    full_kv_path = os.path.join(kv_pair_dir, sub_dir, file)
    full_lines_path = os.path.join(lines_dir, sub_dir, file)
    print(f"Processing: {full_table_path}, {full_kv_path}, {full_lines_path}")
    
    combined_data = combine_all_json_files(full_table_path, full_kv_path, full_lines_path)
    
    # Use the base filename to create a subdirectory for output
    base_filename = os.path.splitext(file)[0]
    output_subdir = os.path.join(output_extraction_dir, sub_dir, base_filename)
    os.makedirs(output_subdir, exist_ok=True)
    
    output_path = os.path.join(output_subdir, "combined_structured_data.json")
    with open(output_path, "w") as outfile:
        json.dump(combined_data, outfile, indent=4)
    
    print(f"Processed file. Output saved to {output_path}")

def process_structured_extraction_directories(table_dir, kv_pair_dir, lines_dir, output_extraction_dir):
    """
    Walk through the table_dir (assuming the same file structure exists in kv_pair_dir and lines_dir),
    and process each JSON file concurrently.
    """
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for root, _, files in os.walk(table_dir):
            sub_dir = os.path.relpath(root, table_dir)
            for file in files:
                if file.endswith('.json'):
                    tasks.append(
                        executor.submit(
                            process_single_file, sub_dir, file, table_dir, kv_pair_dir, lines_dir, output_extraction_dir
                        )
                    )
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print("Error processing file:", e)

###################################
# Main Entry Point
###################################

if __name__ == "__main__":
    document_intelligence_tables_dir = "../tst/document_intelligence_tables"
    document_intelligence_kv_pairs_dir = "../tst/document_intelligence_kv_pairs"
    document_intelligence_lines_dir = "../tst/document_intelligence_lines"
    output_extraction_dir = "../tst/pipeline_results_v2"
    
    process_structured_extraction_directories(document_intelligence_tables_dir,
                                              document_intelligence_kv_pairs_dir,
                                              document_intelligence_lines_dir,
                                              output_extraction_dir)
