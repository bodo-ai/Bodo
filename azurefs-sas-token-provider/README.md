# bodo-azurefs-sas-token-provider

This Python package provides the necessary configuration and functionality to
enable direct uploads to Snowflake stages backed by Azure Data Lake Storage (ADLS) for Bodo.

## Overview

In order to write to ADLS-backed stages in Snowflake, Snowflake provides a SAS (Shared Access Signature) token for authentication. The only way we know to write to ADLS with a SAS token is hadoop.
Hadoop's core-site.xml configuration doesn't directly support the use of SAS tokens. Instead, a custom implementation of `SASTokenProvider` is required. This package provides a way to configure and use the `BodoSASTokenProvider` to fetch the SAS token from a local file, allowing seamless integration with ADLS backed Snowflake stages.

## Configuration Variables

### `SF_AZURE_WRITE_HDFS_CORE_SITE` (str)

This variable specifies the content to be added to the `core-site.xml` file for Snowflake's write operations. It configures a custom `SASTokenProvider` class called `BodoSASTokenProvider`, which is responsible for retrieving the SAS token from a specified file.

- **Value**: This is a constant that should be set to the XML content specifying the implementation of `BodoSASTokenProvider`.
- **Note**: This variable is intended for internal use and should not be modified by end-users.

### `SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION` (str)

This variable specifies the temporary location where the SAS token will be written. The `BodoSASTokenProvider` reads this file to retrieve the SAS token for use during the upload to ADLS.

- **Value**: Path to the file where the SAS token will be stored temporarily.
- **Note**: This variable is for internal use only and should not be modified by end-users.

## How It Works

### **`SF_AZURE_WRITE_HDFS_CORE_SITE`**:

- This value is inserted into the `core-site.xml` file used by Snowflake for writing data to ADLS. It references the custom `BodoSASTokenProvider`, which handles the retrieval of the SAS token.

### **`SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION`**:

- This points to a file location where the SAS token is temporarily stored. The `BodoSASTokenProvider` reads from this file and injects the SAS token into Snowflake’s upload process.

### **`BodoSASTokenProvider`**:

- This custom class reads the SAS token from the file location specified by `SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION` and returns it during the Snowflake upload process.

## Usage

### Install from package manager

Install the package via pip (requires a separate JDK 11 install):

```bash
pip install bodo-azurefs-sas-token-provider
```

or conda:

```bash
conda install -c bodo.ai bodo-azurefs-sas-token-provider
```

### Install from source

Requires JDK 11 and maven to be installed.

```bash
cd azurefs-sas-token-provider && pip install .
```
