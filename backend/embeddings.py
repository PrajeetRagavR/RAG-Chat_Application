# backend/embeddings.py
import os
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class EmbeddingsManager:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return []
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
            
        texts = [doc.page_content for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        if not embeddings:
            return documents
            
        for i, doc in enumerate(documents):
            doc.metadata["embedding"] = embeddings[i]
            
        return documents
    
    def embed_text(self, text: str) -> List[float]:
        if not text:
            return []
            
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []