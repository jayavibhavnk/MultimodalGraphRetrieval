import logging
import pickle
import networkx as nx
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_core.documents.base import Document
from .utils import parallel_process

class GraphDocument(Document):
    def __init__(self, page_content, metadata):
        """
        Initialize a GraphDocument instance.

        Args:
            page_content (str): The content of the document.
            metadata (dict): Metadata associated with the document.
        """
        super().__init__(page_content=page_content, metadata=metadata)

    def __repr__(self):
        return f"GraphDocument(page_content='{self.page_content}', metadata={self.metadata})"

class GraphRAG:
    def __init__(self):
        """
        Initialize a GraphRAG instance.
        """
        self.graph = None
        self.documents = None
        self.embeddings = None
        self.embedding_model = "all-MiniLM-L6-v2"
        self.retrieval_model = "a_star"

    def constructGraph(self, documents, similarity_threshold=0, chunk_size=1250, chunk_overlap=100, metadata=True, use_multiprocessing=False):
        """
        Construct a similarity graph from the provided documents.

        Args:
            documents (list): List of documents to process.
            similarity_threshold (float): Threshold for similarity edges.
            chunk_size (int): Size of text chunks.
            chunk_overlap (int): Overlap between text chunks.
            metadata (bool): Whether to use metadata.
            use_multiprocessing (bool): Whether to use multiprocessing for graph construction.

        Returns:
            tuple: (graph, documents, embeddings)
        """
        try:
            documents = [GraphDocument(doc.page_content, doc.metadata) for doc in documents]
            model = SentenceTransformer(self.embedding_model)
            embeddings = model.encode([doc.page_content for doc in documents])
            graph = nx.Graph()

            if use_multiprocessing:
                add_edges = lambda i: [
                    graph.add_edge(i, j, weight=cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
                    for j in range(i, len(documents))
                    if cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] > similarity_threshold
                ]
                parallel_process(add_edges, range(len(documents)))
            else:
                for i in range(len(documents)):
                    for j in range(i, len(documents)):
                        similarity = cosine_similarity([embeddings[i]], [embeddings[j]])
                        if similarity[0][0] > similarity_threshold:
                            graph.add_edge(i, j, weight=similarity[0][0])

            self.graph = graph
            self.documents = documents
            self.embeddings = embeddings

            logging.info("Graph constructed successfully.")
            return graph, documents, embeddings
        except Exception as e:
            logging.error(f"Error constructing graph: {e}")
            raise

    def compute_similarity(self, current_node, query_embedding):
        """
        Compute similarity between the current node and the query embedding.

        Args:
            current_node (int): The current node in the graph.
            query_embedding (array): The embedding of the query.

        Returns:
            list: List of similar nodes and their similarity scores.
        """
        similar_nodes = []
        for neighbor in self.graph.neighbors(current_node):
            neighbor_embedding = self.embeddings[neighbor]
            neighbor_similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
            similar_nodes.append((neighbor, neighbor_similarity))
        return similar_nodes

    def a_star_search(self, query_text, k=5):
        """
        Perform A* search to find the most similar nodes to the query text.

        Args:
            query_text (str): The query text.
            k (int): Number of similar nodes to retrieve.

        Returns:
            list: List of similar nodes and their similarity scores.
        """
        try:
            model = SentenceTransformer(self.embedding_model)
            query_embedding = model.encode([query_text])[0]

            pq = [(0, None, 0)]
            visited = set()
            similar_nodes = []

            while pq and len(similar_nodes) < k:
                _, current_node, similarity_so_far = heapq.heappop(pq)

                if current_node is not None:
                    similar_nodes.append((current_node, similarity_so_far))

                neighbors = self.graph.neighbors(current_node) if current_node is not None else range(len(self.documents))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        neighbor_embedding = self.embeddings[neighbor]
                        neighbor_similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
                        priority = -neighbor_similarity
                        heapq.heappush(pq, (priority, neighbor, similarity_so_far + neighbor_similarity))
                        visited.add(neighbor)

            logging.info("A* search completed successfully.")
            return similar_nodes
        except Exception as e:
            logging.error(f"Error during A* search: {e}")
            raise

    def similarity_search(self, query, retrieval_model="a_star", k=5):
        """
        Perform a similarity search using the specified retrieval model.

        Args:
            query (str): The query text.
            retrieval_model (str): The retrieval model to use.
            k (int): Number of similar nodes to retrieve.

        Returns:
            list: List of similar documents.
        """
        try:
            retrieval_model = self.retrieval_model
            similar_nodes = []

            if retrieval_model == "a_star":
                similar_indices = [index for index, _ in self.a_star_search(query, k)]
            # Implement other retrieval models as needed

            logging.info("Similarity search completed successfully.")
            return [self.documents[index] for index in similar_indices]
        except Exception as e:
            logging.error(f"Error during similarity search: {e}")
            raise

    def save_db(self, file_path):
        """
        Save the current graph, documents, and embeddings to a file.

        Args:
            file_path (str): The path to save the file.
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump((self.graph, self.documents, self.embeddings), file)
                logging.info("Database saved successfully.")
        except Exception as e:
            logging.error(f"Error saving database: {e}")
            raise

    def load_db(self, file_path):
        """
        Load the graph, documents, and embeddings from a file.

        Args:
            file_path (str): The path to the file.
        """
        try:
            with open(file_path, 'rb') as file:
                self.graph, self.documents, self.embeddings = pickle.load(file)
                logging.info("Database loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading database: {e}")
            raise
