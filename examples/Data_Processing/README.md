# Data Processing Examples

This folder contains various examples of data processing using Bodo. Each example demonstrates different techniques and best practices for efficiently processing large datasets. 

### 1. Easy FASTA  Biometric file parallelization with no code change.

- **File**: `bio_parallelizing_fasta_processing.py` 

- **Description:** This script uses the [Biopython](https://biopython.org) library within a Bodo application to read and process multiple FASTA files.

- **Bodo Benefits:** This is a great example of library support within Bodo; it demonstrates how easily code using outside libraries can be parallelized with minimal code changes.

### 2. Accelerating large data set data exploration and pattern analysis.

- **File**: `chicago-crimes_pattern_analysis.ipynb` 

- **Description:** This notebook performs an exploratory data analysis (EDA) of Chicago crime data, using Pandas operations accelerated by Bodo. It demonstrates common data cleaning, preprocessing, and aggregation steps in a real-world scenario.

- **Bodo Benefits:** This example showcases how Bodo can be used to accelerate data exploration and pattern analysis in large datasets. It highlights the use of Pandas functions like `drop_duplicates`, `groupby`, `pivot_table`, and date/time manipulations within a Bodo-optimized context.

### 3. Accelerating mortgage data preparation pipelines.

- **File**: `etl_data_conversion_mortgages.ipynb` 

- **Description:**  This notebook adapts a RAPIDS example to perform ETL (Extract, Transform, Load) and data conversion operations on mortgage data. It demonstrates complex data manipulation, including date/time handling, merges, and feature engineering.

- **Bodo Benefits:** This provides a complex, real-world example of how Bodo can be used to accelerate data preparation pipelines. It showcases the power of Bodo in handling multiple data transformations, joins, and aggregations, which are common in ETL workflows.

### 4. Accelerating large data set analyses  using Pandas APIs.

- **File**: `nyc-taxi_trip_analysis.ipynb` 

- **Description:** This notebook analyzes New York City taxi trip data (yellow and green taxi trips) to identify patterns and relationships.  It uses Pandas APIs, accelerated by Bodo, for data loading, cleaning, and querying.

- **Bodo Benefits:** This demonstrates how Bodo can be applied to analyze large transportation datasets.  It showcases various Pandas operations, such as `read_parquet`, `read_csv`, `groupby`, `pivot_table`, and date/time functions, within a Bodo environment. 

### 5. Accelerating  large datasets location analytics.

- **File**: `nyc_parking-tickets_analysis.ipynb` 

- **Description:** This notebook analyzes NYC parking ticket data to create maps and extract insights. It demonstrates data loading, aggregation, filtering, and merging operations using Pandas, accelerated by Bodo.

- **Bodo Benefits:** This is a practical example of urban data analysis, showing how Bodo can handle real-world datasets and provide performance improvements for operations like groupby, merge, and filtering.
