from setuptools import setup, find_packages

long_description = """
# Example Usage

```python
from multimodal_rag import MultimodalRAG

# Initialize MultimodalRAG instance
mm_rag = MultimodalRAG(pdf_directory="/path/to/pdf_directory", output_directory="/path/to/output_directory")

# Preprocess documents
mm_rag.preprocess(directory="/path/to/pdf_directory", use_multiprocessing=True)

# Perform a multimodal query
query = "example query text"
search_results, result_paths = mm_rag.multimodal_query(query, k=5)

# Display results
print("Search Results:", search_results)
print("Result Paths:", result_paths)
```
"""

setup(
    name='MultimodalGraphRetrieval',
    version='0.2.2',
    description='Multimodal Graph retrieval',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='JVNK',
    author_email='jaya11vibhav@gmail.com',
    url='https://github.com/jayavibhavnk/MultimodalGraphRetrieval',
    packages=find_packages(),
    install_requires=[
      'langchain',
      'sentence_transformers',
      'langchain-community',
      'pypdf2==3.0.0',
      'pypdf',
      'pymupdf',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
