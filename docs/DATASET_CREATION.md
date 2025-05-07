# Dataset Creation Guide

This feature allows you to create Hugging Face-compatible datasets from web sources, especially from government websites like legislation.gov.uk.

## API Endpoints

The following API endpoints are available for dataset creation:

### Create a Dataset

```
POST /api/datasets
```

Creates a new dataset by crawling and processing a web source.

Request body:
```json
{
  "name": "UK Primary Legislation",
  "source_type": "legislation",
  "url": "https://www.legislation.gov.uk/primary+secondary",
  "description": "All UK primary and secondary legislation",
  "max_depth": 8,
  "user_id": "optional-user-id",
  "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1}
}
```

Source types:
- `legislation`: Special handling for legislation.gov.uk pages
- `text`: Generic text extraction for standard web pages
- `document`: Generic document extraction with title and sections
- `qa`: Question-answering format (experimental)

### List All Datasets

```
GET /api/datasets
```

Returns a list of all datasets and their creation progress.

### Get Dataset Status

```
GET /api/datasets/{dataset_id}
```

Returns the status and progress of a specific dataset creation process.

### Download Dataset

```
GET /api/datasets/{dataset_id}/download
```

Returns the path to the completed dataset.

## Dataset Structure

The created datasets follow the Hugging Face Datasets format and include:

1. A `dataset` directory containing:
   - `train`, `validation`, and `test` splits
   - Dataset metadata
   - A README.md file with dataset information

2. A Python script for loading the dataset

3. Metadata and progress information

## Using the Dataset

Once a dataset is created, it can be loaded using the Hugging Face Datasets library:

```python
from datasets import load_from_disk

dataset_path = "/data/datasets/{dataset_id}/dataset"
dataset = load_from_disk(dataset_path)

print(f"Train set: {len(dataset['train'])} examples")
print(f"Validation set: {len(dataset['validation'])} examples")
print(f"Test set: {len(dataset['test'])} examples")

# Access a sample
print(dataset["train"][0])
```

The dataset can also be pushed to the Hugging Face Hub for wider sharing if desired.

## Agent Instructions

You can use the LangChain agent to create datasets by using natural language instructions. For example:

```
I want to create a model that has full knowledge of all UK law and legislation. 
Create a dataset from https://www.legislation.gov.uk/primary+secondary, which 
contains all current acts and laws.
```

The agent will:
1. Parse your request to determine the dataset name, source, and description
2. Make an API call to create the dataset
3. Provide you with the dataset ID and a way to check the progress
4. Once completed, give you instructions on how to access and use the dataset

## Storage Location

Datasets are stored in the `/data/datasets/{dataset_id}/` directory, which is persistent across restarts.