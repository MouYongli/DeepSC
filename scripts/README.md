# Overview （待修改）

This folder contains of multiple scripts and data files used for web crawling, downloading, and processing specific datasets. Each file serves a distinct purpose, and their detailed descriptions are provided below.

## File Descriptions

### Scripts

- **crawl_3ca.py**
  - A Python script designed to crawl web data from predefined sources.
  - It extracts relevant information based on the defined parameters and stores the results in a structured format.
  - It uses the `requests` library to make HTTP requests and `BeautifulSoup` to parse HTML content.
  - Usage:
    ```sh
    python crawl_3ca.py
    ```

- **download_3ca.py**
  - A Python script responsible for downloading files based on the URLs obtained from `crawl_3ca.py`.
  - Uses `requests` to fetch files and stores them in a designated directory.
  - Includes error handling to ensure successful downloads and retries in case of failures.
  - Usage:
    ```sh
    ./download_3ca.sh
    ```

- **download_3ca.sh**
  - Fetches files from URLs specified in a text file and saves them locally.
  - Ensures proper execution by setting the script as executable:
    ```sh
    chmod +x download_3ca.sh
    ./download_3ca.sh
    ```

- **process_3ca.sh**
  -Identify which matrix file contains the raw count data.
  -Insert datasets with raw count into target_dataset_files.csv.
  -Use another script (not provided) to generate a dataset ID for each dataset.

- **noiseAnalysis.py**
  -Analyze the proportion of noise within the matrix.

- **merge_adata.py**
  -Combine .mtx, gene, and cell files into AnnData objects.
- **filter_datasets.py**
  -Filter out genes with low expression levels or cells with a low number of genes

### Data Files

- **data_info.csv**
  - A structured CSV file that contains metadata about the downloaded files.
  - Includes the following columns:
    - `Study_uuid`: Unique identifier for each study.
    - `Organ`: The organ category related to the study.
    - `Title`: Title of the study.
    - `Title_link`: URL link to the study.
    - `Data`: Data information related to the study.
    - `Data_link`: URL to access the data.
    - `Meta data`: Metadata associated with the study.
    - `Meta data_link`: URL to access metadata.
    - `Cell types`: Types of cells analyzed.
    - `Cell types_link`: URL to access cell type information.
    - `Summary`: Summary of the study.
    - `Summary_link`: URL to access the summary.
    - `Disease`: Disease category related to the study.
    - `Technology`: Technology used in the study.
    - `#samples`: Number of samples in the study.
    - `#cells`: Number of cells analyzed.
    - `Meta programs`: Meta programs associated with the study.
    - `Meta programs_link`: URL to access meta program details.
    - `CNAs`: Copy number alterations.
    - `CNAs_link`: URL to access CNA data.
    - `UMAP`: UMAP visualization data.
    - `UMAP_link`: URL to access UMAP data.
    - `Cell cycle`: Cell cycle information.
    - `Cell cycle_link`: URL to access cell cycle data.
  - Can be loaded and analyzed using Pandas in Python:
    ```python
    import pandas as pd
    df = pd.read_csv('data_info.csv')
    print(df.head())
    ```

- **organ_list.txt**
  - A plain text file that lists various organ categories, each on a new line.
  - Categories included:
    - Head-and-neck
    - Lung
    - Liver/Biliary
    - Kidney
    - Prostate
    - Sarcoma
    - Other Models
    - Brain
    - Breast
    - Pancreas
    - Neuroendocrine
    - Colorectal
    - Ovarian
    - Skin
    - Hematologic
  - This file can be used as a reference for filtering or classification purposes.

## Usage Instructions

1. **Install Dependencies**
   - If using Python, install the required packages:
     ```sh
     pip install requests beautifulsoup4 pandas
     ```

2. **Run the Web Crawling Script**
   - Execute `crawl_3ca.py` to scrape data from predefined sources:
     ```sh
     python crawl_3ca.py
     ```

3. **Download the Extracted Data**
   - use `download_3ca.sh` for a shell-based approach:
     ```sh
     ./download_3ca.sh
     ```

4. **Process the Data**
   - Load `data_info.csv` to review the metadata and verify successful downloads.
   - Use `organ_list.txt` for categorization.

## Additional Notes
- Ensure that you have a stable internet connection before running the scripts.
- Verify that `data_info.csv` contains accurate URLs before initiating downloads.
- Modify the scripts as needed to adapt to different data sources.

## Contact Information
For any issues, questions, or enhancements, contact the project maintainers.

## License
This project is licensed under the MIT License.
