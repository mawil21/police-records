# police-records

# Set Up 
```python
# Cloning the repo 
git clone git@github.com:mawil21/police-records.git
```

# Structured Extraction 
## Set Up 
```python
pip3 install docling
```

## Add dataset path and structured extraction path in `src/structured_extraction.py`
```python
input_pdf_dir =  # Input Directory containg the dataset 
output_pdf_dir =  # Output Directory to save structured extraction output
```

## Running the Extraction 
```python
cd src 
python3 structured_extraction.py 
```

#### Changing Extraction Methods 
Currently, we are using `Docling` to extract tables and `Azure Document Intelligence` to extract key-value pairs. If you would like to change models, feel free to go to the `models` directory and add your model. And change the model parameter in `structured_extraction.py`.
```python
def process_pdf(pdf_file, input_dir, output_dir):
    ...
    #Process the PDF with docling
    tables = # TODO 
    print(f"Extracted {len(tables)} tables from {relative_path}")
    key_value_pairs = # TODO 
    print(f"Extracted {len(key_value_pairs)} key-value pairs from {relative_path}")
```