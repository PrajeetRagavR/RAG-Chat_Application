# backend/reranker.py
from typing import List
from langchain.schema import Document
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
            
        # Prepare pairs for cross encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores from cross encoder
        scores = self.model.predict(pairs)
        
        # Add scores to document metadata
        for i, doc in enumerate(documents):
            doc.metadata["relevance_score"] = float(scores[i])
        
        # Sort documents by score (descending)
        sorted_docs = sorted(documents, key=lambda x: x.metadata["relevance_score"], reverse=True)
        
        return sorted_docs