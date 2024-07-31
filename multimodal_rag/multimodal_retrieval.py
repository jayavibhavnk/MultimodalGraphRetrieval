import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer
import networkx as nx
import heapq
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class GraphDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"GraphDocument(page_content='{self.page_content}', metadata={self.metadata})"

class MultimodalRetrieval:
    def __init__(self):
        self.text_graph = nx.Graph()
        self.image_graph = nx.Graph()
        self.text_documents = []
        self.image_metadata = []
        self.text_embeddings = []
        self.caption_embeddings = []
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.tfidf_vectorizer = TfidfVectorizer()

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_new_tokens=20)
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption

    def construct_text_graph(self, documents, similarity_threshold=0.5):
        documents = [GraphDocument(doc.page_content, doc.metadata) for doc in documents]
        embeddings = self.text_model.encode([doc.page_content for doc in documents])
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
        
        for i, doc in enumerate(documents):
            node_id = f"text_{i}"
            self.text_graph.add_node(node_id, doc=doc, embedding=embeddings[i], tfidf=tfidf_matrix[i])
            self.text_documents.append(doc)
            self.text_embeddings.append(embeddings[i])
        
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > similarity_threshold:
                    self.text_graph.add_edge(f"text_{i}", f"text_{j}", weight=similarity)

    def construct_image_graph(self, image_paths, similarity_threshold=0.5):
        for i, image_path in enumerate(image_paths):
            caption = self.generate_caption(image_path)
            caption_embedding = self.text_model.encode([caption])[0]
            metadata = {
                "path": image_path,
                "caption": caption
            }
            node_id = f"image_{i}"
            self.image_graph.add_node(node_id, metadata=metadata, embedding=caption_embedding)
            self.image_metadata.append(metadata)
            self.caption_embeddings.append(caption_embedding)

        for i in range(len(self.image_metadata)):
            for j in range(i+1, len(self.image_metadata)):
                similarity = cosine_similarity([self.caption_embeddings[i]], [self.caption_embeddings[j]])[0][0]
                if similarity > similarity_threshold:
                    self.image_graph.add_edge(f"image_{i}", f"image_{j}", weight=similarity)

    def link_graphs(self, similarity_threshold=0.5):
        for i, text_embedding in enumerate(self.text_embeddings):
            for j, caption_embedding in enumerate(self.caption_embeddings):
                similarity = cosine_similarity([text_embedding], [caption_embedding])[0][0]
                if similarity > similarity_threshold:
                    self.text_graph.add_edge(f"text_{i}", f"image_{j}", weight=similarity)

    def attention_mechanism(self, query_embedding, node_embeddings):
        attention_scores = F.softmax(torch.matmul(query_embedding, node_embeddings.T), dim=-1)
        return attention_scores

    def hybrid_search(self, query_text, k=5, alpha=0.5):
        query_embedding = self.text_model.encode([query_text])[0]
        query_tfidf = self.tfidf_vectorizer.transform([query_text])
        
        similarities = []
        for node in self.text_graph.nodes():
            if node.startswith("text_"):
                node_data = self.text_graph.nodes[node]
                embedding_sim = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
                tfidf_sim = cosine_similarity(query_tfidf, node_data['tfidf'])[0][0]
                hybrid_sim = alpha * embedding_sim + (1 - alpha) * tfidf_sim
                similarities.append((node, hybrid_sim))
            elif node.startswith("image_"):
                node_data = self.image_graph.nodes[node]
                embedding_sim = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
                similarities.append((node, embedding_sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def multi_hop_query(self, query_text, k=5, max_hops=2):
        query_embedding = self.text_model.encode([query_text])[0]
        initial_results = self.hybrid_search(query_text, k)
        
        pq = [(1 - sim, 0, node) for node, sim in initial_results]
        heapq.heapify(pq)
        
        visited = set()
        results = []
        
        while pq and len(results) < k:
            neg_sim, hops, node = heapq.heappop(pq)
            
            if node in visited:
                continue
            
            visited.add(node)
            results.append((node, 1 - neg_sim))
            
            if hops < max_hops:
                neighbors = list(self.text_graph.neighbors(node)) if node.startswith("text_") else list(self.image_graph.neighbors(node))
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        if neighbor.startswith("text_"):
                            neighbor_embedding = self.text_graph.nodes[neighbor]['embedding']
                        else:
                            neighbor_embedding = self.image_graph.nodes[neighbor]['embedding']
                        
                        similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
                        heapq.heappush(pq, (1 - similarity, hops + 1, neighbor))
        
        if not results:
            return []

        # Optimize tensor creation
        result_embeddings = np.stack([
            self.text_graph.nodes[node]['embedding'] if node.startswith("text_") else self.image_graph.nodes[node]['embedding']
            for node, _ in results
        ])
        result_embeddings = torch.from_numpy(result_embeddings)
        
        attention_scores = self.attention_mechanism(torch.from_numpy(query_embedding), result_embeddings)
        
        weighted_results = [(node, sim * attention_scores[i].item()) for i, (node, sim) in enumerate(results)]
        return sorted(weighted_results, key=lambda x: x[1], reverse=True)
        
    def query(self, query_text, k=5, use_multi_hop=True):
        if use_multi_hop:
            results = self.multi_hop_query(query_text, k)
        else:
            results = self.hybrid_search(query_text, k)
        
        text_results = []
        image_results = []
        for node, score in results:
            if node.startswith("text_"):
                text_results.append((self.text_graph.nodes[node]['doc'], score))
            else:
                image_results.append((self.image_graph.nodes[node]['metadata'], score))
        
        return {
            "text_results": text_results,
            "image_results": image_results
        }

    def preprocess(self, text_documents, image_paths, similarity_threshold=0.5):
        self.construct_text_graph(text_documents, similarity_threshold)
        self.construct_image_graph(image_paths, similarity_threshold)
        self.link_graphs(similarity_threshold)
        print("Graphs constructed successfully with text and images linked!")

    def save_db(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.text_graph, self.image_graph, self.text_documents, self.image_metadata, self.text_embeddings, self.caption_embeddings, self.tfidf_vectorizer), file)
            print("Database saved!")

    def load_db(self, file_path):
        with open(file_path, 'rb') as file:
            self.text_graph, self.image_graph, self.text_documents, self.image_metadata, self.text_embeddings, self.caption_embeddings, self.tfidf_vectorizer = pickle.load(file)
            print("Database loaded!")
            
    def query_balanced(self, query_text, k_text=3, k_image=3, use_multi_hop=True):
        if use_multi_hop:
            all_results = self.multi_hop_query(query_text, k_text + k_image)
        else:
            all_results = self.hybrid_search(query_text, k_text + k_image)
        
        text_results = []
        image_results = []
        
        for node, score in all_results:
            if node.startswith("text_") and len(text_results) < k_text:
                text_results.append((self.text_graph.nodes[node]['doc'], score))
            elif node.startswith("image_") and len(image_results) < k_image:
                image_results.append((self.image_graph.nodes[node]['metadata'], score))
            
            if len(text_results) == k_text and len(image_results) == k_image:
                break
        
        return {
            "text_results": text_results,
            "image_results": image_results
        }
