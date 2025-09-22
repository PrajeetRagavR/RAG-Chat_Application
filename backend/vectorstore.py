# backend/vectorstore.py
import os
from typing import List, Dict, Any, Optional
import chromadb
from langchain.schema import Document
from dotenv import load_dotenv
import hashlib

load_dotenv()

class VectorStore:
    def __init__(self, collection_name=None):
        self.collection_name = "RAG_Collection" # Ensure a consistent collection name
        
        # Ensure chroma_db directory exists
        if not os.path.exists("./chroma_db"):
            os.makedirs("./chroma_db")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_documents(self, documents: List[Document]):
        if not documents:
            return False
            
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = []
            embeddings = []
            ids = []
            
            for doc in documents:
                # Create metadata without embedding
                metadata = {k: v for k, v in doc.metadata.items() if k != "embedding"}
                metadatas.append(metadata)
                
                # Get embedding from metadata
                if "embedding" in doc.metadata:
                    embeddings.append(doc.metadata["embedding"])
                else:
                    # If no embedding, we'll need to generate one later
                    embeddings.append(None)
                
                # Generate unique ID
                unique_id = hashlib.sha256(
                    (doc.page_content + str(metadata.get("source", ""))).encode()
                ).hexdigest()
                ids.append(unique_id)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings if all(embeddings) else None,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing documents: {str(e)}")
            return False
    
    def search_similar(self, query_vector: List[float], limit: int = 25) -> List[Dict[str, Any]]:
        if not query_vector:
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i]
                    })
                
            return formatted_results
            
        except Exception as e:
            print(f"Error searching for similar documents: {str(e)}")
            return []
    
    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False