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
