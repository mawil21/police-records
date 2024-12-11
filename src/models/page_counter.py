import os
import json
from PyPDF2 import PdfReader
import shutil
import random


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


# Optimization Problem - Knapsack Problem
def knapsack_pdfs(all_pdfs, capacity=5000):
    # all_pdfs is a list of tuples: (directory, pdf_name, page_count)
    pages = [p[2] for p in all_pdfs]

    # We'll store which items are chosen with backtracking from DP table.
    n = len(pages)
    # DP table dimensions: (n+1) x (capacity+1)
    dp = [[0]*(capacity+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for w in range(capacity+1):
            if pages[i-1] <= w:
                # Option to include this PDF
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-pages[i-1]] + pages[i-1])
            else:
                # Can't include this PDF
                dp[i][w] = dp[i-1][w]

    # Backtrack to find which items were chosen
    chosen = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            chosen.append(all_pdfs[i-1])
            w -= pages[i-1]

    chosen.reverse()
    return chosen

# Randomly sample from each cluster
def sample_from_clusters(directory_summary, pdf_dir, sample_ratio=0.5, max_pages=5000):
    chosen_pdfs = []
    total_pages = 0

    # Step 1: Randomly sample 50% from each cluster
    for cluster_dir, info in directory_summary.items():
        pdfs = info["pdf_details"]
        # Shuffle and pick 50%
        random.shuffle(pdfs)
        sample_count = max(1, int(len(pdfs) * sample_ratio)) if pdfs else 0
        selected = pdfs[:sample_count]

        for pdf in selected:
            chosen_pdfs.append((cluster_dir, pdf["pdf_name"], pdf["page_count"]))

    # Step 2: Check total pages
    total_pages = sum(p[2] for p in chosen_pdfs)

    # Step 3: If over max_pages, randomly remove until under limit
    if total_pages > max_pages:
        random.shuffle(chosen_pdfs)
        # Remove PDFs until under limit
        while total_pages > max_pages and chosen_pdfs:
            removed_pdf = chosen_pdfs.pop()
            total_pages -= removed_pdf[2]

    return chosen_pdfs, total_pages

def create_sampled_data(chosen_pdfs, pdf_dir, target_dir="sampled_data"):
    os.makedirs(target_dir, exist_ok=True)

    for directory, pdf_name, _ in chosen_pdfs:
        relative_path = os.path.relpath(directory, pdf_dir)
        dest_dir = os.path.join(target_dir, relative_path)
        os.makedirs(dest_dir, exist_ok=True)

        source_path = os.path.join(directory, pdf_name)
        dest_path = os.path.join(dest_dir, pdf_name)
        shutil.copyfile(source_path, dest_path)

    print(f"Copied {len(chosen_pdfs)} PDFs into '{target_dir}' with directory structure.")


def sample_from_clusters(directory_summary, pdf_dir, sample_ratio=0.5, max_pages=5000):
    chosen_pdfs = []
    total_pages = 0

    # Step 1: Randomly sample 50% from each cluster
    for cluster_dir, info in directory_summary.items():
        pdfs = info["pdf_details"]
        # Shuffle and pick 50%
        random.shuffle(pdfs)
        sample_count = max(1, int(len(pdfs) * sample_ratio)) if pdfs else 0
        selected = pdfs[:sample_count]

        for pdf in selected:
            chosen_pdfs.append((cluster_dir, pdf["pdf_name"], pdf["page_count"]))

    # Step 2: Check total pages
    total_pages = sum(p[2] for p in chosen_pdfs)

    # Step 3: If over max_pages, randomly remove until under limit
    if total_pages > max_pages:
        random.shuffle(chosen_pdfs)
        # Remove PDFs until under limit
        while total_pages > max_pages and chosen_pdfs:
            removed_pdf = chosen_pdfs.pop()
            total_pages -= removed_pdf[2]

    return chosen_pdfs, total_pages


# Example usage:
if __name__ == "__main__":
    directory = "../../tst/master_dataset"
    output_file = "directory_summary.json"

    directory_summary, error_files = process_structured_extraction_directories(directory)
    save_to_json(directory_summary, error_files, output_file)
    
    with open(output_file, "r") as f:
        data = json.load(f)

    directory_summary = data["directory_summary"]

    # Flatten and ensure uniqueness
    # all_pdfs = []
    # for d, info in directory_summary.items():
    #     for pdf in info["pdf_details"]:
    #         all_pdfs.append((d, pdf["pdf_name"], pdf["page_count"]))

    # # Remove duplicates
    # unique_dict = {}
    # for d, pdf_name, pages in all_pdfs:
    #     unique_dict[(d, pdf_name)] = pages

    # all_pdfs = [(d, p, unique_dict[(d, p)]) for (d, p) in unique_dict]

    # chosen_pdfs = knapsack_pdfs(all_pdfs, capacity=5000)
    # total_pages = sum(p[2] for p in chosen_pdfs)
    # print(f"Number of PDFs chosen: {len(chosen_pdfs)}")
    # print(f"Total pages: {total_pages}")

    # # Create a directory with chosen PDFs, preserving directory structure
    # create_sampled_data(chosen_pdfs, directory, "sampled_data")


    # Randomly sample 50% from each cluster, ensuring under 5000 pages total
    chosen_pdfs, total_pages = sample_from_clusters(directory_summary, directory, sample_ratio=0.5, max_pages=5000)
    print(f"Number of PDFs chosen: {len(chosen_pdfs)}")
    print(f"Total pages: {total_pages}")

    # Create a directory with chosen PDFs, preserving directory structure
    create_sampled_data(chosen_pdfs, directory, "sampled_data")

