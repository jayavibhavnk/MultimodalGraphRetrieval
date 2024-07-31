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
# Multimodal Retrieval (with captioning and image and graph linkage)

```python
from langchain_core.documents.base import Document

# Prepare text documents
text_documents = [
    Document(page_content="That car was on fire.", metadata={"source": "doc1.pdf", "page": 1}),
    Document(page_content="That vehicle is called lava", metadata={"source": "doc2.pdf", "page": 1})
]

# Prepare image paths
image_paths = [
    "/content/car.jpg",
    "/content/fire.jpg",
]

# Initialize and preprocess
rag = MultimodalGraphRAG()
rag.preprocess(text_documents, image_paths, similarity_threshold=0.2)

# Perform a query
query = "that car is fire"
results = rag.query(query, k=3, use_multi_hop=True)

results = rag.query_balanced(query, k_text=3, k_image=3, use_multi_hop=True)

# Print the results
print("Text Results:")
for doc, score in results["text_results"]:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print()

print("Image Results:")
for metadata, score in results["image_results"]:
    print(f"Image Path: {metadata['path']}")
    print(f"Caption: {metadata['caption']}")
    print(f"Score: {score}")
    print()
```
"""

setup(
    name='MultimodalGraphRetrieval',
    version='0.0.1a',
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
