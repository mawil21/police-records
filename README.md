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

## Add dataset path and structured extraction path in src/structured_extraction.py
```python
input_pdf_dir =  # Input Directory containg the dataset 
output_pdf_dir =  # Output Directory to save structured extraction output
```

## Running the Extraction 
```python
cd src 
python3 structured_extraction.py 
```