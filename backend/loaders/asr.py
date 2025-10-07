# backend/loaders/asr.py
import os
import json
import wave
import queue
import threading
from typing import List, Optional, Callable
from langchain.schema import Document
from vosk import Model, KaldiRecognizer, SetLogLevel

# Import optional dependencies with fallbacks
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("PyAudio not available. Real-time voice input will be disabled.")
    PYAUDIO_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("pyttsx3 not available. Text-to-speech will be disabled.")
    TTS_AVAILABLE = False

# Set Vosk logging level to reduce output
SetLogLevel(-1)

class VoiceAssistant:
    """
    Voice Assistant class that handles both speech recognition and text-to-speech.
    """
    def __init__(self, model_path=None, on_transcription_callback=None):
        """
        Initialize the Voice Assistant with both ASR and TTS capabilities.
        
        Args:
            model_path: Path to the Vosk model. If None, will download a small model.
            on_transcription_callback: Callback function that receives transcribed text
        """
        # Check dependencies
        if not PYAUDIO_AVAILABLE:
            print("WARNING: PyAudio is not installed. Real-time voice input will not work.")
            print("Install PyAudio with: pip install pyaudio")
            
        if not TTS_AVAILABLE:
            print("WARNING: pyttsx3 is not installed. Text-to-speech will not work.")
            print("Install pyttsx3 with: pip install pyttsx3")
        
        # Initialize ASR
        self.asr = VoskASR(model_path, on_transcription_callback)
        
        # Initialize TTS if available
        self.tts = TextToSpeech() if TTS_AVAILABLE else None
        
        # State variables
        self.is_listening = False
        self.listening_thread = None
    
    def start_listening(self):
        """Start real-time voice recognition"""
        if not PYAUDIO_AVAILABLE:
            print("ERROR: Cannot start listening - PyAudio is not installed")
            return
            
        if not self.is_listening:
            self.is_listening = True
            self.asr.start_listening()
    
    def stop_listening(self):
        """Stop real-time voice recognition"""
        if self.is_listening:
            self.is_listening = False
            self.asr.stop_listening()
    
    def speak(self, text):
        """Convert text to speech"""
        if not TTS_AVAILABLE or self.tts is None:
            print(f"TEXT OUTPUT (TTS not available): {text}")
            return
            
        self.tts.speak(text)
    
    def transcribe_file(self, audio_path):
        """Transcribe an audio file"""
        return self.asr.transcribe_audio(audio_path)
    
    def shutdown(self):
        """Clean up resources"""
        self.stop_listening()
        if TTS_AVAILABLE and self.tts is not None:
            self.tts.shutdown()


class VoskASR:
    def __init__(self, model_path=None, on_transcription_callback=None):
        """
        Initialize the Vosk ASR system.
        
        Args:
            model_path: Path to the Vosk model. If None, will download a small model.
            on_transcription_callback: Callback function that receives transcribed text
        """
        # If model_path is not provided, use a default location
        if model_path is None:
            model_path = os.path.join(os.path.expanduser("~"), "vosk-model-small-en-us-0.15")
            
            # Check if model exists, if not, download it
            if not os.path.exists(model_path):
                import urllib.request
                import zipfile
                
                print("Downloading Vosk model (this may take a while)...")
                model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                zip_path = model_path + ".zip"
                
                # Download the model
                urllib.request.urlretrieve(model_url, zip_path)
                
                # Extract the model
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to parent directory
                    zip_ref.extractall(os.path.dirname(model_path))
                
                # Remove the zip file
                os.remove(zip_path)
                print("Model downloaded and extracted successfully.")
        
        # Load the model
        try:
            self.model = Model(model_path)
            print(f"Loaded Vosk model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Vosk model: {str(e)}")
        
        # Real-time recognition variables
        self.on_transcription_callback = on_transcription_callback
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.listening_thread = None
    
    def start_listening(self):
        """Start real-time voice recognition"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._listen_worker)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        print("Started listening for voice input...")
    
    def stop_listening(self):
        """Stop real-time voice recognition"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=1)
        print("Stopped listening for voice input.")
    
    def _listen_worker(self):
        """Worker thread for real-time voice recognition"""
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open microphone stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8000)
        
        # Create recognizer
        rec = KaldiRecognizer(self.model, 16000)
        rec.SetWords(True)
        
        print("Microphone is now active, speak clearly...")
        
        # Process audio in real-time
        while self.is_listening:
            data = stream.read(4000, exception_on_overflow=False)
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get('text', '').strip()
                
                if text and self.on_transcription_callback:
                    self.on_transcription_callback(text)
        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file using Vosk.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Open the audio file
            wf = wave.open(audio_path, "rb")
            
            # Check if the audio format is compatible
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise ValueError("Audio file must be WAV format mono PCM.")
            
            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            # Process audio
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result.get('text', ''))
            
            # Get final result
            part_result = json.loads(rec.FinalResult())
            results.append(part_result.get('text', ''))
            
            # Combine all results
            transcription = ' '.join(results).strip()
            return transcription
            
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {str(e)}")
            return ""
    
    def audio_to_document(self, audio_path: str) -> Optional[Document]:
        """
        Convert audio file to a Document object.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Document object with transcription as content
        """
        transcription = self.transcribe_audio(audio_path)
        if not transcription:
            return None
            
        metadata = {
            "source": audio_path,
            "type": "audio_transcription",
            "transcriber": "vosk"
        }
        
        return Document(page_content=transcription, metadata=metadata)
    
    def process_audio_files(self, audio_paths: List[str]) -> List[Document]:
        """
        Process multiple audio files and convert them to Documents.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of Document objects
        """
        documents = []
        for path in audio_paths:
            doc = self.audio_to_document(path)
            if doc:
                documents.append(doc)
        
        return documents


class TextToSpeech:
    """Text-to-speech using pyttsx3"""
    
    def __init__(self, voice_id=None, rate=150, volume=1.0):
        """
        Initialize the TTS engine.
        
        Args:
            voice_id: ID of the voice to use (None for default)
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        if voice_id:
            self.engine.setProperty('voice', voice_id)
        
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
    
    def speak(self, text):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
        """
        if not text:
            return
        
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_available_voices(self):
        """Get list of available voices"""
        return self.engine.getProperty('voices')
    
    def shutdown(self):
        """Clean up resources"""
        self.engine.stop()


# Example usage
if __name__ == "__main__":
    def on_transcription(text):
        print(f"Recognized: {text}")
        
        # Simple echo response
        if text:
            assistant.speak(f"You said: {text}")
    
    # Create voice assistant
    assistant = VoiceAssistant(on_transcription_callback=on_transcription)
    
    try:
        # Start listening
        assistant.start_listening()
        
        # Keep the program running
        print("Press Ctrl+C to exit")
        while True:
            pass
    except KeyboardInterrupt:
        # Clean up on exit
        assistant.shutdown()
        print("Voice assistant stopped.")