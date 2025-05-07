"""Dataset creator for generating Hugging Face datasets from web sources."""
import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import datasets
import requests
from bs4 import BeautifulSoup
from datasets import Dataset, Features, Value
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for storing datasets
DATA_DIR = os.environ.get("DATA_MOUNT_PATH", "/data")
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create DATA_DIR at {DATA_DIR}: {e}")
        # Fall back to a temporary directory
        DATA_DIR = tempfile.mkdtemp(prefix="datasets_")
        logger.info(f"Using temporary directory for datasets: {DATA_DIR}")


class DatasetType(str, Enum):
    """Types of datasets that can be created."""
    TEXT = "text"
    LEGISLATION = "legislation"
    DOCUMENT = "document"
    QA = "qa"


class DatasetStatus(str, Enum):
    """Status of dataset creation process."""
    PENDING = "pending"
    CRAWLING = "crawling"
    PROCESSING = "processing"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetRequest(BaseModel):
    """Request to create a dataset from a web source."""
    name: str
    source_type: DatasetType
    url: str
    description: str
    max_depth: int = 8
    user_id: Optional[str] = None
    split_ratio: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}


class DatasetProgress(BaseModel):
    """Progress information about dataset creation."""
    dataset_id: str
    name: str
    status: DatasetStatus
    url: str
    description: str
    progress: float = 0.0
    total_documents: int = 0
    processed_documents: int = 0
    creation_time: str
    completion_time: Optional[str] = None
    error_message: Optional[str] = None
    dataset_path: Optional[str] = None
    splits: Optional[Dict[str, int]] = None


class DatasetCreator:
    """Creator for Hugging Face datasets from web sources."""
    
    def __init__(self):
        """Initialize dataset creator."""
        self.progress_registry = {}
        self.datasets_dir = os.path.join(DATA_DIR, "datasets")
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def _get_dataset_dir(self, dataset_id: str) -> str:
        """Get the directory path for a dataset."""
        return os.path.join(self.datasets_dir, dataset_id)
    
    def create_progress_tracker(self, request: DatasetRequest) -> DatasetProgress:
        """Create a new progress tracker for a dataset creation request."""
        dataset_id = f"{request.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        progress = DatasetProgress(
            dataset_id=dataset_id,
            name=request.name,
            status=DatasetStatus.PENDING,
            url=request.url,
            description=request.description,
            creation_time=datetime.now().isoformat(),
        )
        self.progress_registry[dataset_id] = progress
        os.makedirs(self._get_dataset_dir(dataset_id), exist_ok=True)
        
        return progress
    
    def update_progress(self, dataset_id: str, **kwargs) -> None:
        """Update the progress for a dataset."""
        if dataset_id in self.progress_registry:
            progress = self.progress_registry[dataset_id]
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            
            # Save progress to a file for persistence
            progress_path = os.path.join(self._get_dataset_dir(dataset_id), "progress.json")
            with open(progress_path, "w") as f:
                f.write(progress.json())
    
    def get_progress(self, dataset_id: str) -> Optional[DatasetProgress]:
        """Get the progress for a dataset."""
        return self.progress_registry.get(dataset_id)
    
    def list_datasets(self) -> List[DatasetProgress]:
        """List all datasets and their progress."""
        return list(self.progress_registry.values())
    
    def legislation_extractor(self, html: str) -> str:
        """Extract text from legislation.gov.uk pages."""
        soup = BeautifulSoup(html, "lxml")
        
        # Remove navigation, header, and other non-content elements
        for element in soup.select('nav, header, footer, .backtotop, .interface'):
            if element:
                element.decompose()
        
        # Extract the main content
        main_content = soup.select_one('#layout2, #content, article, .LegContent')
        
        if main_content:
            # Extract title and metadata
            title_elem = soup.select_one('.pageTitle, h1')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract content
            text = []
            if title:
                text.append(f"# {title}\n\n")
            
            # Extract section headings and content
            for section in main_content.select('.LegSection, .LegPart, .LegSchedule'):
                section_title = section.select_one('.LegTitle, h2, h3, h4')
                if section_title:
                    text.append(f"## {section_title.get_text(strip=True)}\n\n")
                
                paragraphs = section.select('.LegP, p')
                for para in paragraphs:
                    text.append(para.get_text(strip=True) + "\n\n")
            
            # If no structured sections were found, extract all paragraphs
            if len(text) <= 1:
                for para in main_content.select('p'):
                    text.append(para.get_text(strip=True) + "\n\n")
            
            return "".join(text).strip()
        
        # Fallback to simple extraction
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()
    
    async def crawl_website(self, dataset_id: str, url: str, max_depth: int = 8) -> List[Dict]:
        """Crawl a website to extract content."""
        progress = self.get_progress(dataset_id)
        if not progress:
            return []
        
        self.update_progress(dataset_id, status=DatasetStatus.CRAWLING)
        
        documents = []
        
        try:
            # Determine if we need to use Sitemap or recursive crawling
            if progress.url.endswith(".xml") or "sitemap" in progress.url.lower():
                # Use SitemapLoader for sitemap URLs
                loader = SitemapLoader(
                    progress.url,
                    filter_urls=[url.split('/sitemap')[0]],
                    parsing_function=self.legislation_extractor if progress.url.startswith("https://www.legislation.gov.uk") else None,
                )
                docs = loader.load()
                
                for i, doc in enumerate(docs):
                    documents.append({
                        "id": f"doc_{i}",
                        "url": doc.metadata.get("source", ""),
                        "title": doc.metadata.get("title", f"Document {i}"),
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })
                    
                    # Update progress periodically
                    if i % 10 == 0:
                        self.update_progress(
                            dataset_id, 
                            processed_documents=i,
                            total_documents=len(docs),
                            progress=float(i) / max(len(docs), 1) * 50  # First 50% is crawling
                        )
            else:
                # Use RecursiveUrlLoader for regular URLs
                extractor = self.legislation_extractor if progress.url.startswith("https://www.legislation.gov.uk") else None
                
                loader = RecursiveUrlLoader(
                    url=progress.url,
                    max_depth=max_depth,
                    extractor=extractor,
                    prevent_outside=True,
                    use_async=True,
                    timeout=600,
                )
                
                docs = loader.load()
                
                # Extract documents
                for i, doc in enumerate(docs):
                    documents.append({
                        "id": f"doc_{i}",
                        "url": doc.metadata.get("source", ""),
                        "title": doc.metadata.get("title", f"Document {i}"),
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })
                    
                    # Update progress periodically
                    if i % 10 == 0:
                        self.update_progress(
                            dataset_id, 
                            processed_documents=i,
                            total_documents=len(docs),
                            progress=float(i) / max(len(docs), 1) * 50  # First 50% is crawling
                        )
            
            # Update final crawling progress
            self.update_progress(
                dataset_id,
                processed_documents=len(documents),
                total_documents=len(documents),
                progress=50.0  # Crawling complete - 50%
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error crawling website: {e}")
            self.update_progress(
                dataset_id,
                status=DatasetStatus.FAILED,
                error_message=f"Error crawling website: {str(e)}"
            )
            return []
    
    async def process_documents(self, dataset_id: str, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Process documents into train/val/test splits."""
        progress = self.get_progress(dataset_id)
        if not progress:
            return {}
        
        self.update_progress(dataset_id, status=DatasetStatus.PROCESSING)
        
        try:
            # Get split ratios
            total_docs = len(documents)
            if total_docs == 0:
                self.update_progress(
                    dataset_id,
                    status=DatasetStatus.FAILED,
                    error_message="No documents were extracted from the website"
                )
                return {}
            
            # Default split ratios
            split_ratio = {"train": 0.8, "validation": 0.1, "test": 0.1}
            
            # Calculate document counts for each split
            train_end = int(total_docs * split_ratio["train"])
            val_end = train_end + int(total_docs * split_ratio["validation"])
            
            # Create splits
            splits = {
                "train": documents[:train_end],
                "validation": documents[train_end:val_end],
                "test": documents[val_end:],
            }
            
            # Update progress
            self.update_progress(
                dataset_id,
                processed_documents=total_docs,
                total_documents=total_docs,
                progress=75.0,  # Processing complete - 75%
                splits={split: len(docs) for split, docs in splits.items()}
            )
            
            return splits
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            self.update_progress(
                dataset_id,
                status=DatasetStatus.FAILED,
                error_message=f"Error processing documents: {str(e)}"
            )
            return {}
    
    async def build_dataset(self, dataset_id: str, splits: Dict[str, List[Dict]]) -> bool:
        """Build a Hugging Face dataset from processed documents."""
        progress = self.get_progress(dataset_id)
        if not progress:
            return False
        
        self.update_progress(dataset_id, status=DatasetStatus.BUILDING)
        
        try:
            dataset_dir = self._get_dataset_dir(dataset_id)
            
            # Define features based on dataset type
            features = Features({
                "id": Value("string"),
                "url": Value("string"),
                "title": Value("string"),
                "text": Value("string"),
            })
            
            # Create dataset for each split
            hf_datasets = {}
            for split_name, docs in splits.items():
                if not docs:
                    continue
                    
                # Create dataset
                ds = Dataset.from_dict(
                    {
                        "id": [doc["id"] for doc in docs],
                        "url": [doc["url"] for doc in docs],
                        "title": [doc["title"] for doc in docs],
                        "text": [doc["text"] for doc in docs],
                    },
                    features=features
                )
                
                hf_datasets[split_name] = ds
            
            # Create a DatasetDict
            dataset_dict = datasets.DatasetDict(hf_datasets)
            
            # Save the dataset
            dataset_dict.save_to_disk(os.path.join(dataset_dir, "dataset"))
            
            # Write dataset info for HF
            dataset_card = f"""---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- text-generation
task_ids:
- language-modeling
---

# Dataset Card for {progress.name}

## Dataset Description

- **Repository:** {progress.url}
- **Created:** {progress.creation_time}

### Dataset Summary

{progress.description}

### Supported Tasks and Leaderboards

The dataset is intended for training language models on {progress.name}.

### Languages

The text in the dataset is in English.

## Dataset Structure

### Data Instances

A typical data instance contains the following:
- `id`: A unique identifier
- `url`: Source URL
- `title`: Title of the document
- `text`: Text content

### Data Fields

- `id`: A string feature containing the document ID
- `url`: A string feature containing the source URL
- `title`: A string feature containing the document title
- `text`: A string feature containing the document text

### Data Splits

- Train: {len(splits.get('train', []))} examples
- Validation: {len(splits.get('validation', []))} examples
- Test: {len(splits.get('test', []))} examples

## Dataset Creation

### Source Data

The dataset was created by crawling and processing {progress.url}.

## Considerations for Using the Data

### Citation Information

Please cite the original source if you use this dataset.
"""
            
            # Write the dataset card
            with open(os.path.join(dataset_dir, "dataset", "README.md"), "w") as f:
                f.write(dataset_card)
            
            # Create a script file for loading the dataset
            script_content = f"""# coding=utf-8
# Copyright 2023 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import datasets

_DESCRIPTION = \"\"\"
{progress.description}
\"\"\"

_CITATION = \"\"\"
@misc{{{dataset_id.replace("_", "-")},
  author = {{Automatically Generated}},
  title = {{{progress.name}}},
  year = {{2023}},
  url = {{{progress.url}}},
}}
\"\"\"

_HOMEPAGE = "{progress.url}"
_LICENSE = "cc-by-4.0"


class {dataset_id.replace("_", "")}(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {{
                    "id": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }}
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={{"split": "train"}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={{"split": "validation"}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={{"split": "test"}},
            ),
        ]

    def _generate_examples(self, split):
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
        dataset = datasets.load_from_disk(dataset_path)
        for i, example in enumerate(dataset[split]):
            yield i, {{
                "id": example["id"],
                "url": example["url"],
                "title": example["title"],
                "text": example["text"],
            }}
"""
            
            # Write the script file
            script_path = os.path.join(dataset_dir, f"{dataset_id}.py")
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Create a simple metadata.json file
            metadata = {
                "id": dataset_id,
                "name": progress.name,
                "description": progress.description,
                "url": progress.url,
                "created_at": progress.creation_time,
                "completed_at": datetime.now().isoformat(),
                "splits": {split: len(docs) for split, docs in splits.items()}
            }
            
            with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Update progress to completed
            self.update_progress(
                dataset_id,
                status=DatasetStatus.COMPLETED,
                progress=100.0,
                completion_time=datetime.now().isoformat(),
                dataset_path=os.path.join(dataset_dir, "dataset")
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error building dataset: {e}")
            self.update_progress(
                dataset_id,
                status=DatasetStatus.FAILED,
                error_message=f"Error building dataset: {str(e)}"
            )
            return False
    
    async def create_dataset(self, request: DatasetRequest) -> str:
        """Create a dataset from a web source."""
        # Create progress tracker
        progress = self.create_progress_tracker(request)
        dataset_id = progress.dataset_id
        
        # Start the dataset creation in a separate task
        asyncio.create_task(self._create_dataset_task(dataset_id, request))
        
        return dataset_id
    
    async def _create_dataset_task(self, dataset_id: str, request: DatasetRequest) -> None:
        """Background task for dataset creation."""
        try:
            # Step 1: Crawl the website
            documents = await self.crawl_website(dataset_id, request.url, request.max_depth)
            
            if not documents:
                return
            
            # Step 2: Process documents into splits
            splits = await self.process_documents(dataset_id, documents)
            
            if not splits:
                return
            
            # Step 3: Build the dataset
            await self.build_dataset(dataset_id, splits)
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            self.update_progress(
                dataset_id,
                status=DatasetStatus.FAILED,
                error_message=f"Error creating dataset: {str(e)}"
            )


# Singleton instance
creator = DatasetCreator()


async def create_dataset(request: DatasetRequest) -> str:
    """Create a dataset from a web source."""
    return await creator.create_dataset(request)


def get_dataset_progress(dataset_id: str) -> Optional[DatasetProgress]:
    """Get the progress of a dataset creation task."""
    return creator.get_progress(dataset_id)


def list_datasets() -> List[DatasetProgress]:
    """List all datasets and their creation progress."""
    return creator.list_datasets()