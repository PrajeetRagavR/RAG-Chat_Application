# backend/loaders/audio_loader.py
import os
from typing import List, Optional
from langchain.schema import Document
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class AudioProcessor:
    def __init__(self, model_name="openai/whisper-large-v3"):
        self.model_name = model_name
        self.client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"],
        )
    
    def transcribe_audio(self, audio_path: str) -> str:
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            output = self.client.automatic_speech_recognition(audio_bytes, model=self.model_name)
            return output.text
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {str(e)}")
            return ""
    
    def audio_to_document(self, audio_path: str) -> Optional[Document]:
        transcription = self.transcribe_audio(audio_path)
        if not transcription:
            return None
            
        metadata = {
            "source": audio_path,
            "type": "audio_transcription"
        }
        
        return Document(page_content=transcription, metadata=metadata)
    
    def process_audio_files(self, audio_paths: List[str]) -> List[Document]:
        documents = []
        for path in audio_paths:
            doc = self.audio_to_document(path)
            if doc:
                documents.append(doc)
        
        return documents