import zipfile
import os

def unzip_data(zip_path, extract_to):
    # Check if zip file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at: {zip_path}")

    # Extract the ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_ref.namelist()}")

if __name__ == "__main__":
    zip_path = "../data/archive.zip"   
    extract_to = "../data/"             
    unzip_data(zip_path, extract_to)
