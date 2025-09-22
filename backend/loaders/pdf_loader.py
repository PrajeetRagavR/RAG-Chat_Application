# backend/loaders/pdf_loader.py
import os
import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import re
import fitz  # PyMuPDF
import base64
import io
from PIL import Image
from langchain_nvidia_ai_endpoints import ChatNVIDIA

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

class PDFProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.llm = ChatNVIDIA(model="nvidia/neva-22b", nvidia_api_key=os.getenv("NVIDIA_API_KEY"))

    def _img_to_base64_string(self, image_bytes, image_ext):
        """Converts image bytes to a base64 string."""
        buffered = io.BytesIO(image_bytes)
        img = Image.open(buffered)
        if img.width > 800 or img.height > 800:
            img.thumbnail((800, 800))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        temp_buffered = io.BytesIO()
        img.save(temp_buffered, format="JPEG", quality=85)
        return base64.b64encode(temp_buffered.getvalue()).decode()

    def process_image_with_neva(self, image_data):
        """Processes a single image with NVIDIA Neva-22B and returns a description."""
        image_bytes = image_data["bytes"]
        image_ext = image_data["ext"]
        page_number = image_data["page"]

        try:
            base64_image = self._img_to_base64_string(image_bytes, image_ext)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ]
            
            response = self.llm.invoke(messages)
            description = response.content
            return Document(page_content=description, metadata={"source": f"image_from_page_{page_number}"})
        except Exception as e:
            print(f"Failed to process image from page {page_number}: {e}")
            return None

    def process_images(self, images):
        """Processes a list of images with NVIDIA Neva-22B."""
        if not images:
            return []
        
        processed_documents = []
        for image_data in images:
            doc = self.process_image_with_neva(image_data)
            if doc:
                processed_documents.append(doc)
        return processed_documents

    def extract_images(self, pdf_path):
        """Extract images from PDF using PyMuPDF"""
        images = []
        try:
            pdf_file = fitz.open(pdf_path)
            for page_index in range(len(pdf_file)):
                page = pdf_file.load_page(page_index)
                image_list = page.get_images(full=True)

                for image_index, img in enumerate(image_list, start=1):
                    xref = img[0]
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    images.append({"bytes": image_bytes, "ext": image_ext, "page": page_index + 1})
            pdf_file.close()
            return images
        except Exception as e:
            print(f"Failed to extract images: {e}")
            return []

    def extract_text_with_ocr(self, pdf_path):
        """Extract text from scanned PDFs using OCR"""
        text = ""
        try:
            pdf_file = fitz.open(pdf_path)
            for i, page in enumerate(pdf_file):
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                text += pytesseract.image_to_string(img)
            pdf_file.close()
        except Exception as e:
            print(f"Failed to perform OCR: {e}")
        return text

    def extract_text(self, pdf_path):
        """Attempt to extract text using direct extraction, fallback to OCR if needed"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if not text.strip():
            text = self.extract_text_with_ocr(pdf_path)
        return text

    def extract_tables(self, pdf_path):
        """Extract tables using pdfplumber and return as text"""
        table_texts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = "\n".join(["\t".join([cell for cell in row if cell is not None]) for row in table if row is not None])
                        if table_str:
                            table_texts.append(table_str)
            return "\n\n".join(table_texts)
        except Exception as e:
            print(f"Failed to extract tables: {e}")
            return ""

    def load_document(self, pdf_path):
        if not pdf_path.endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        text = self.extract_text(pdf_path)
        tables = self.extract_tables(pdf_path)

        cleaned_text = clean_text(text)

        return [Document(page_content=cleaned_text, metadata={"source": pdf_path, "tables": tables.strip()})]

    def split_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def process_pdf(self, pdf_path):
        docs = self.load_document(pdf_path)
        images_data = self.extract_images(pdf_path)
        image_documents = self.process_images(images_data)
        chunks = self.split_documents(docs)
        return chunks, image_documents