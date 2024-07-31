import os
import fitz
import logging
import json
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .graph_rag import GraphRAG
from .utils import setup_logging

setup_logging()

class MultimodalRAG(GraphRAG):
    def __init__(self, pdf_directory=None, output_directory=None):
        """
        Initialize a MultimodalRAG instance.

        Args:
            pdf_directory (str): Directory containing PDF files.
            output_directory (str): Directory to save images.
        """
        super().__init__()
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        self.image_info_list = []

    def create_graph(self, similarity_threshold=0.5, use_multiprocessing=False):
        """
        Create a similarity graph from the PDF documents.

        Args:
            similarity_threshold (float): Threshold for similarity edges.
            use_multiprocessing (bool): Whether to use multiprocessing for graph construction.

        Returns:
            tuple: (graph, documents, embeddings)
        """
        try:
            loader = DirectoryLoader(self.pdf_directory, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=150, add_start_index=True)
            splits = text_splitter.split_documents(docs)

            self.graph, self.documents, self.embeddings = self.constructGraph(
                splits, similarity_threshold=similarity_threshold, use_multiprocessing=use_multiprocessing
            )
            logging.info("Graph created successfully.")
            return self.graph, self.documents, self.embeddings
        except Exception as e:
            logging.error(f"Error creating graph: {e}")
            raise

    def convert_pdfs_to_images(self):
        """
        Convert PDF files to images and save them to the output directory.

        Returns:
            list: List of image information (image path, source PDF, page number).
        """
        try:
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)

            for pdf_file in os.listdir(self.pdf_directory):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(self.pdf_directory, pdf_file)
                    doc = fitz.open(pdf_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        image_filename = f"{os.path.splitext(pdf_file)[0]}_page_{page_num + 1}.png"
                        image_path = os.path.join(self.output_directory, image_filename)
                        pix.save(image_path)
                        self.image_info_list.append([image_path, pdf_file, page_num + 1])

            logging.info("PDFs converted to images successfully.")
            return self.image_info_list
        except Exception as e:
            logging.error(f"Error converting PDFs to images: {e}")
            raise

    def preprocess(self, directory, use_multiprocessing=False):
        """
        Preprocess PDF documents to create a graph and convert PDFs to images.

        Args:
            directory (str): Directory containing PDF files.
            use_multiprocessing (bool): Whether to use multiprocessing for graph construction.
        """
        self.pdf_directory = directory
        self.output_directory = os.path.join(directory, 'images')

        logging.info("Creating graph.")
        self.create_graph(use_multiprocessing=use_multiprocessing)
        logging.info("Converting PDFs to images.")
        self.convert_pdfs_to_images()
        logging.info("Preprocessing completed.")

    def extract_page_numbers(self, search_results):
        """
        Extract page numbers from search results.

        Args:
            search_results (list): List of search result documents.

        Returns:
            list: List of page numbers and source PDFs.
        """
        return [[doc.metadata['source'], doc.metadata['page']] for doc in search_results]

    def search_images(self, search_list):
        """
        Search for images corresponding to the provided search list.

        Args:
            search_list (list): List of search items (PDF, page number).

        Returns:
            list: List of image paths.
        """
        result_list = []
        for search_item in search_list:
            search_pdf, search_page = search_item
            for image_info in self.image_info_list:
                image_path, source_pdf, page_number = image_info
                if os.path.basename(search_pdf) == source_pdf and search_page == page_number:
                    result_list.append(image_path)
                    break
        return result_list

    def multimodal_query(self, query, k=7):
        """
        Perform a multimodal query to retrieve similar documents and their images.

        Args:
            query (str): The query text.
            k (int): Number of similar documents to retrieve.

        Returns:
            tuple: (search_results, result_paths)
        """
        try:
            search_results = self.similarity_search(query, k=k)
            pg_nos = self.extract_page_numbers(search_results)
            result_paths = self.search_images(pg_nos)
            logging.info("Multimodal query completed successfully.")
            return search_results, result_paths
        except Exception as e:
            logging.error(f"Error during multimodal query: {e}")
            raise

    def save_image_db(self, file_path="image_info_list.json"):
        """
        Save image information to a file.

        Args:
            file_path (str): The path to save the file.
        """
        try:
            with open(file_path, "w") as file:
                json.dump(self.image_info_list, file)
                logging.info("Image database saved successfully.")
        except Exception as e:
            logging.error(f"Error saving image database: {e}")
            raise

    def load_image_db(self, file_path="image_info_list.json"):
        """
        Load image information from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: List of image information.
        """
        try:
            with open(file_path, "r") as file:
                self.image_info_list = json.load(file)
                logging.info("Image database loaded successfully.")
                return self.image_info_list
        except Exception as e:
            logging.error(f"Error loading image database: {e}")
            raise

    def format_docs(self, docs):
        """
        Format documents for display.

        Args:
            docs (list): List of documents to format.

        Returns:
            str: Formatted document strings.
        """
        return "\n\n".join(doc.page_content + " source: " + doc.metadata['source'] for doc in docs)
