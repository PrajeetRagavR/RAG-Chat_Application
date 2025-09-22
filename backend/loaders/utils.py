# backend/loaders/utils.py
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
    """Clean text by removing extra whitespace"""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=500, chunk_overlap=120):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)