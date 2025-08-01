# D:\medical_ai_project\rag_system.py

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MedicalProductRAG:
    def __init__(self, data_path="medical_products.json", model_name="all-MiniLM-L6-v2"):
        """
        Initializes the RAG system by loading data, creating embeddings, and building a FAISS index.
        Args:
            data_path (str): Path to the JSON file containing medical product data.
            model_name (str): Name of the Sentence Transformer model to use for embeddings.
        """
        self.data_path = data_path
        # Sentence Transformer model is downloaded on first use.
        # It converts text into numerical vectors (embeddings).
        self.model = SentenceTransformer(model_name)
        self.products = self._load_data()
        self.index = None
        self.product_texts = [] # Store the text that was embedded for retrieval (useful for debugging)
        self._build_faiss_index()

    def _load_data(self):
        """Loads medical product data from the JSON file."""
        print(f"Loading data from {self.data_path}...")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} medical products.")
            return data
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found. Please ensure the file exists.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.data_path}. Check file format.")
            print("Please ensure your medical_products.json is a valid JSON array of objects.")
            return []

    def _build_faiss_index(self):
        """
        Creates embeddings for product descriptions and builds a FAISS index.
        We'll embed a combination of 'product_name' and 'use_for' for better search.
        """
        if not self.products:
            print("No products loaded, cannot build FAISS index.")
            return

        print("Building FAISS index...")
        self.product_texts = []
        for product in self.products:
            # Combine relevant fields into a single string for embedding
            # Using .get() with a default value prevents KeyErrors if a field is missing
            text_to_embed = f"Product Name: {product.get('product_name', 'N/A')}. Use Case: {product.get('use_for', 'N/A')}."
            self.product_texts.append(text_to_embed)

        # Generate embeddings in batches for efficiency.
        # This can be slow for very large datasets, but the progress bar helps.
        embeddings = self.model.encode(self.product_texts, show_progress_bar=True, convert_to_numpy=True)
        embedding_dim = embeddings.shape[1]

        # Initialize FAISS index
        # IndexFlatL2 is a simple index that uses L2 (Euclidean) distance
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings.astype('float32')) # FAISS requires float32
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve_products(self, query_symptoms, k=3):
        """
        Retrieves the top-k most relevant medical products based on query symptoms.
        Args:
            query_symptoms (str): User's input symptoms.
            k (int): Number of top products to retrieve.
        Returns:
            list: A list of dictionaries, each representing a retrieved product.
        """
        if self.index is None:
            print("FAISS index not built. Cannot retrieve products. Check data loading.")
            return []

        # Encode the query symptoms into an embedding
        query_embedding = self.model.encode([query_symptoms])[0].astype('float32')

        # D: Distances, I: Indices
        # Search the FAISS index for the k nearest neighbors
        distances, indices = self.index.search(np.array([query_embedding]), k)

        retrieved_products = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # Ensure a valid index was found (FAISS returns -1 for no match)
                product_info = self.products[idx].copy()
                # Calculate relevance score: 1 - (normalized_distance). Lower distance = higher relevance.
                # Assuming distances are positive and we want a score between 0 and 1.
                # Max distance could vary, so 1 - dist is a simple proxy.
                product_info['relevance_score'] = 1 - (distances[0][i] / (distances[0][0] + 1e-6)) # Normalize by closest dist
                retrieved_products.append(product_info)
        
        # Sort by relevance score in descending order
        retrieved_products.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        
        return retrieved_products

if __name__ == "__main__":
    # This block runs only when rag_system.py is executed directly
    print("--- Testing MedicalProductRAG System ---")
    
    # Initialize RAG system (ensure medical_products.json is in the same directory)
    rag_system = MedicalProductRAG()

    # Test queries
    test_queries = [
        "I have a fever and body pain.",
        "I'm sneezing a lot and have a runny nose.",
        "I feel chest congestion and sore throat.",
        "My stomach hurts and I have heartburn.",
        "I have a minor cut on my finger and it's bleeding a little.",
        "I feel very tired and dehydrated because of diarrhea.",
        "My skin is itchy and red from an insect bite.",
        "I have a dry cough that keeps me up at night."
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        retrieved = rag_system.retrieve_products(query, k=3) # Retrieve top 3 products
        if retrieved:
            print("  Relevant Products Found:")
            for product in retrieved:
                print(f"    - Product: {product.get('product_name', 'N/A')}")
                print(f"      Use: {product.get('use_for', 'N/A')}")
                print(f"      Side Effects: {product.get('side_effects', 'N/A')}")
                print(f"      Relevance Score: {product.get('relevance_score', 0.0):.4f}") # Display more precision
        else:
            print("  No relevant products found.")
    print("\n--- RAG System Test Complete ---")