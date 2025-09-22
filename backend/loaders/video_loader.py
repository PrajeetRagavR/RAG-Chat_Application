import os
import tempfile
import requests
from typing import List, Optional
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence
from langchain.schema import Document
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import io
from embeddings import EmbeddingsManager

# Load env variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

class VideoProcessor:
    def __init__(self):
        self.model_name = "openai/whisper-large-v3"
        self.client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"],
        )
        self.embeddings_manager = EmbeddingsManager()
        self.kSupportedList = {
            "mp4": ["video/mp4", "video"],
            "wav": ["audio/wav", "audio"],
        }

    def get_extention(self, filename):
        _, ext = os.path.splitext(filename)
        return ext[1:].lower()

    def mime_type(self, ext):
        return self.kSupportedList[ext][0]

    def media_type(self, ext):
        return self.kSupportedList[ext][1]

    def extract_audio(self, video_path: str) -> str:
        """Extracts audio from a video file and saves it to a temporary WAV file."""
        try:
            video = VideoFileClip(video_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
                video.audio.write_audiofile(audio_path)
            video.close()
            return audio_path
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {str(e)}")
            return ""

    def transcribe_audio(self, audio_path: str) -> str:
        try:
            audio = AudioSegment.from_wav(audio_path)
            audio = audio.set_frame_rate(16000)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            
            output = self.client.automatic_speech_recognition(buffer.read(), model=self.model_name)
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

    def process_video_files(self, video_paths: List[str]) -> List[Document]:
        """Process videos and return documents with audio transcription and embeddings"""
        documents = []
        for video_path in video_paths:
            audio_file_path = self.extract_audio(video_path)
            if audio_file_path:
                # Use the new audio processing functions
                audio_documents = self.process_audio_files([audio_file_path])
                documents.extend(audio_documents)
                try:
                    os.unlink(audio_file_path)
                except:
                    pass
        return documents
