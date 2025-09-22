# backend/api.py
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import tempfile
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json

from rag_pipeline import RAGPipeline
from logger import get_logger
from loaders.pdf_loader import PDFProcessor
from loaders.audio_loader import AudioProcessor
from loaders.video_loader import VideoProcessor
from logger import setup_logging
from vectorstore import VectorStore
from langgraph_chatbot import Chatbot # Import the Chatbot

router = APIRouter()
logger = get_logger(__name__)

# Initialize processors
pdf_processor = PDFProcessor()
audio_processor = AudioProcessor()
video_processor = VideoProcessor()

# Initialize RAG pipeline (only for document processing, not for chat)
rag_pipeline = RAGPipeline()

# Initialize LangGraph Chatbot
chatbot = Chatbot()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_id = str(uuid.uuid4())
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Process based on file type
        documents = []
        if file_ext == '.pdf':
            chunks, image_docs = pdf_processor.process_pdf(temp_file_path)
            documents.extend(chunks)
            documents.extend(image_docs)
        elif file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
            audio_docs = audio_processor.process_audio_files([temp_file_path])
            documents.extend(audio_docs)
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_docs = video_processor.process_video_files([temp_file_path])
            documents.extend(video_docs)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store documents in vector store
        if documents:
            rag_pipeline.store_documents(documents)
            logger.info(f"Stored {len(documents)} documents from {file.filename}")
        else:
            logger.warning(f"No documents extracted from {file.filename}")
        
        # Clean up
        os.unlink(temp_file_path)
        
        response_content = {"message": f"File processed successfully. Extracted {len(documents)} documents."}
        if documents and (file_ext in ['.mp3', '.wav', '.m4a', '.ogg'] or file_ext in ['.mp4', '.avi', '.mov', '.mkv']):
            # Assuming the first document contains the main transcription
            response_content["transcribed_text"] = documents[0].page_content

        return JSONResponse(
            status_code=200,
            content=response_content
        )
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/clear_db")
async def clear_database():
    try:
        rag_pipeline.vector_store.delete_collection()
        # Re-initialize the vector store within the existing RAG pipeline instance
        rag_pipeline.vector_store = VectorStore()
        logger.info("Vector store collection cleared and re-initialized.")
        return JSONResponse(status_code=200, content={"message": "Vector store cleared successfully."})
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")

@router.post("/query")
async def query_documents(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        
        # Use the LangGraph chatbot to generate the response
        response_content = chatbot.invoke(request.query, session_id, session_id) # user_id is also session_id for now
        
        # The response from chatbot.invoke is a string, so we need to format it
        # to match the expected structure of the frontend.
        # For now, we'll just return the response content as the 'response' field.
        # We might need to adjust this if the frontend expects more details like sources.
        return JSONResponse(status_code=200, content={
            "response": response_content,
            "session_id": session_id,
            "detailed_sources": [] # LangGraph chatbot doesn't provide sources directly in this setup
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/faq")
async def get_faq():
    try:
        faq_file_path = current_dir / "faq.json"
        if not faq_file_path.exists():
            raise HTTPException(status_code=404, detail="FAQ file not found")
        with open(faq_file_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        return JSONResponse(status_code=200, content=faq_data)
    except Exception as e:
        logger.error(f"Error retrieving FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving FAQ: {str(e)}")