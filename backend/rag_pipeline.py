# backend/rag_pipeline.py
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from typing import List, Dict, Any
from langchain.schema import Document
from vectorstore import VectorStore
from embeddings import EmbeddingsManager
from reranker import Reranker
from logger import get_logger

from llm import get_llm
import json

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embeddings_manager = EmbeddingsManager()
        self.reranker = Reranker()
        self.llm = get_llm()

    def store_documents(self, documents: List[Document]):
        """Store documents in the vector store with embeddings"""
        documents_with_embeddings = self.embeddings_manager.embed_documents(documents)
        self.vector_store.store_documents(documents_with_embeddings)
    
    def retrieve_documents(self, query: str, limit: int = 25) -> List[Document]:
        """Retrieve relevant documents for a query"""
        query_embedding = self.embeddings_manager.embed_text(query)
        results = self.vector_store.search_similar(query_embedding, limit)
        
        # Convert results to Document objects
        documents = []
        for result in results:
            documents.append(Document(
                page_content=result["text"],
                metadata=result["metadata"]
            ))
        
        # The previous line was causing an error as VectorStore does not have 'similarity_search'
        # We will use the documents obtained from search_similar directly.
        # retrieved_docs = self.vector_store.similarity_search(query, k=self.top_k)
        logger.info(f"Retrieved {len(documents)} documents for query: {query}")
        return documents

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query"""
        return self.reranker.rerank_documents(query, documents)
    
    def expand_query(self, query: str) -> List[str]:
        """Expand the user's query into multiple related queries using the LLM."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query. Generate 3 related search queries, one per line, in JSON format. For example: {\"queries\": [\"query1\", \"query2\", \"query3\"]}"},
            {"role": "user", "content": f"Generate multiple search queries for: {query}"}
        ]
        
        try:
            completion = self.llm.invoke(messages)
            expanded_queries = json.loads(completion.content)["queries"]
            logger.info(f"Expanded queries for '{query}': {expanded_queries}")
            return expanded_queries
        except Exception as e:
            logger.error(f"Error expanding query with LLM: {str(e)}")
            return [query] # Fallback to original query

    def generate_response(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a structured response to the query using retrieved documents and chat history"""
        # chat_history here is the history *before* the current query.

        # Expand the query
        expanded_queries = self.expand_query(query)
        logger.info(f"Expanded queries: {expanded_queries}")
        
        # Retrieve relevant documents for each expanded query
        all_retrieved_docs = []
        logger.info(f"Attempting to retrieve documents for {len(expanded_queries)} expanded queries.")
        for expanded_query in expanded_queries:
            all_retrieved_docs.extend(self.retrieve_documents(expanded_query, limit=10))
        
        # Remove duplicates while preserving order
        unique_docs = []
        seen_page_content = set()
        for doc in all_retrieved_docs:
            if doc.page_content not in seen_page_content:
                unique_docs.append(doc)
                seen_page_content.add(doc.page_content)

        retrieved_docs = unique_docs
        logger.info(f"Retrieved {len(retrieved_docs)} unique documents.")
        
        if not retrieved_docs:
            return {
                "response": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": []
            }
        
        # Rerank documents
        reranked_docs = self.reranker.rerank_documents(query, retrieved_docs)
        logger.info(f"Reranked {len(reranked_docs)} documents.")

        # Check if the highest reranked score is below a threshold
        if reranked_docs and reranked_docs[0].metadata.get("relevance_score", 0) < 0.5:
            return {
                "response": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": []
            }
        
        # Prepare context from top documents
        context = ""
        sources = []
        
        for i, doc in enumerate(reranked_docs[:5]):  # Use top 5 documents
            context += f"Document {i+1}:\n{doc.page_content}\n\n"
            
            # Extract source information
            source_info = doc.metadata.get("source", "Unknown")
            page_info = doc.metadata.get("page", "")
            doc_type = doc.metadata.get("type", "")
            
            source_detail = source_info
            if page_info:
                source_detail += f" (Page {page_info})"
            if doc_type:
                source_detail += f" - {doc_type}"
                
            if source_detail not in sources:
                sources.append(source_detail)
        
        # Construct messages for the LLM, including chat history
        messages = [
            {"role": "system", "content": "You are ALFRED, a digital butler. Your primary goal is to provide concise and accurate answers based *Strictly* on the provided documents and conversation history. If the information is not available in the given context, state that you cannot find the answer in the provided documents. Avoid making assumptions or inventing information. Maintain a professional and helpful tone."}
        ]
        logger.info(f"Messages sent to LLM: {messages}")
        
        # Add chat history to messages
        for msg in chat_history:
            messages.append(msg)

        # Now add the current user query, augmented with context
        messages.append({"role": "user", "content": f"Based on the following context, answer the question:\nContext:\n{context}\n\nQuestion: {query}"})

        try:
            completion = self.llm.invoke(messages)
            generated_text = completion.content
            logger.info(f"LLM generated response: {generated_text}")
            
            # Clean up the answer
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            # Update chat_history with current user query and assistant's response
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

            return {
                "response": answer,
                "sources": sources,
                "detailed_sources": [
                    {
                        "source": source,
                        "relevance_score": doc.metadata.get("relevance_score", 0) if i < len(reranked_docs) else 0
                    }
                    for i, source in enumerate(sources)
                ],
                "chat_history": chat_history # Return updated chat history
            }
                
        except Exception as e:
            logger.error(f"Error calling NVIDIA API: {str(e)}")
            # Fallback to a more user-friendly response
            chat_history.append({"role": "user", "content": query}) # Add user query even on error
            chat_history.append({"role": "assistant", "content": "I apologize, but I encountered an error while trying to generate a response. Please try again later.", "error": True})
            return {
                "response": "I apologize, but I encountered an error while trying to generate a response. Please try again later.",
                "sources": sources,
                "detailed_sources": [{"source": source, "relevance_score": 0} for source in sources],
                "chat_history": chat_history # Return updated chat history even on error
            }