# RAG-Chat_Application

## Multimodal RAG Chatbot

### 1. Project Overview

This project implements a multimodal Retrieval-Augmented Generation (RAG) chatbot designed to interact with users through various input modalities, including text, audio, and potentially video. The chatbot leverages advanced Natural Language Processing (NLP) and Automatic Speech Recognition (ASR) techniques to provide comprehensive and contextually relevant responses. The core idea is to enhance traditional chatbot capabilities by allowing it to process information from diverse sources and respond intelligently.

### 2. Features

- **Multimodal Input**: Accepts user queries via text and speech using *ASR(Automatic-Speech-Recognition)* for user input queries, documents(PDF,Docx,txt), audio and video files for the RAG pipeline.
- **Retrieval-Augmented Generation**: Utilizes a RAG pipeline to fetch relevant information from a knowledge base before generating responses, ensuring accuracy and reducing hallucinations.
- **Automatic Speech Recognition (ASR)**: Integrates Vosk ASR for accurate and efficient transcription of spoken language into text.
- **Text-to-Speech (TTS)**: Generates natural-sounding voice responses to users, enhancing the conversational experience.
- **Document Loading**: Capable of processing various document types, including PDFs, audio files, and video files, to build a rich knowledge base.
- **Modular Backend**: A well-structured backend with distinct modules for API handling, embeddings, LLM interaction, data loading, RAG pipeline management, and vector store operations.
- **User-Friendly Frontend**: A simple yet effective frontend for seamless user interaction.

### 3. Backend Architecture

The backend is designed with a modular approach to handle various aspects of the RAG chatbot's functionality.

- **API (`api.py`, `main.py`)**: Provides the interface for the frontend and other services to interact with the chatbot. It handles incoming requests, routes them to appropriate modules, and returns responses.
- **Embeddings (`embeddings.py`)**: Responsible for generating numerical representations (embeddings) of text data. These embeddings are crucial for semantic search and retrieval within the RAG pipeline.
- **Large Language Model (LLM) (`llm.py`)**: Integrates with a powerful Large Language Model to generate human-like text responses based on the retrieved context and user queries.
- **Loaders (`loaders/`)**: A collection of modules responsible for ingesting and processing data from various sources into a standardized format (e.g., `langchain.schema.Document`).
    - **`pdf_loader.py`**: Handles the extraction and processing of text from PDF documents.
    - **`audio_loader.py`**: Processes audio files, potentially using transcription services to convert speech to text. Currently uses HuggingFace's `InferenceClient` with a Whisper model.
    - **`video_loader.py`**: Extracts information from video files, which might involve transcribing audio tracks and/or analyzing visual content.
    - **`asr.py`**: Specifically designed for real-time Automatic Speech Recognition using Vosk, converting live voice input into text for processing by the chatbot. It also includes Text-to-Speech (TTS) functionality using `pyttsx3` for voice responses.
- **RAG Pipeline (`rag_pipeline.py`, `langgraph_chatbot.py`)**: The core of the chatbot's intelligence. It orchestrates the retrieval of relevant information from the vector store based on the user's query and then uses the LLM to generate a coherent and informed response. `langgraph_chatbot.py` likely implements a more advanced conversational flow using LangGraph.
- **Vector Store (`vectorstore.py`)**: Stores the embeddings of the knowledge base documents, enabling efficient semantic search and retrieval of relevant information during the RAG process.
- **Reranker (`reranker.py`)**: (If implemented) Further refines the retrieved documents by re-ranking them based on their relevance to the query, improving the quality of the context provided to the LLM.
- **Logger (`logger.py`)**: Manages logging throughout the application for debugging, monitoring, and auditing purposes.
- **Memory Utilities (`memory_utils.py`)**: Handles conversational memory, allowing the chatbot to maintain context across multiple turns in a conversation.

### 4. Frontend Structure

The frontend, located in the `frontend/` directory, provides the user interface for interacting with the RAG chatbot. It is built using a web framework (e.g., Streamlit, Flask, React, etc. - *specific framework to be determined or added based on implementation*). The `app.py` file within the `frontend/` directory is the main entry point for the frontend application.
*The RAG - Chat Application currently uses Streamlit for prototyping.*

- **User Interface**: Presents a conversational interface where users can input text queries or use voice input through the integrated ASR.
- **Interaction with Backend**: Communicates with the backend API to send user queries, receive chatbot responses, and manage multimodal inputs/outputs. This typically involves HTTP requests to the endpoints exposed by the backend's `api.py`.
- **Real-time Feedback**: Provides real-time feedback to the user, such as displaying transcribed speech, indicating when the chatbot is processing a request, and playing back voice responses.

### 5. Technologies and Models

This project leverages several key technologies and models to achieve its multimodal RAG capabilities:

- **LangChain**: A powerful framework for developing applications powered by language models. It provides abstractions for working with LLMs, managing conversational memory, building RAG pipelines, and integrating with various data sources and tools. LangChain is central to orchestrating the flow of information and interactions within the chatbot.
- **Vosk ASR**: An open-source, offline, and lightweight speech recognition toolkit. Vosk is chosen for its ability to perform real-time speech-to-text transcription locally, ensuring privacy and reducing latency for voice input. Its small model sizes make it suitable for deployment in various environments.
- **HuggingFace Inference Client (Whisper Model)**: Used in `audio_loader.py` for transcribing pre-recorded audio files. The Whisper model, known for its high accuracy across multiple languages, is accessed via HuggingFace's Inference Client, providing a robust solution for batch audio processing.
- **`pyttsx3`**: A cross-platform text-to-speech conversion library in Python. It is used in `asr.py` to generate natural-sounding voice responses from the chatbot's text output, enhancing the user experience by providing an auditory feedback mechanism.
- **Python**: The primary programming language for both the backend and frontend (if using frameworks like Streamlit or Flask). Its rich ecosystem of libraries for NLP, machine learning, and web development makes it an ideal choice for this project.
- **Docker**: Used for containerization of both the backend and frontend applications, ensuring consistent environments across development, testing, and deployment. This simplifies dependency management and deployment processes.

### 6. Setup Instructions

To get started with the RAG-ChatApplication, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/RAG-ChatApplication.git
    cd RAG-ChatApplication
    ```

2.  **Set up a Virtual Environment**:
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Install the required Python packages.
    ```bash
pip install -r requirements.txt
    ```
    *Note for PyAudio*: If you encounter issues installing `PyAudio` (e.g., `ModuleNotFoundError` or compilation errors), you might need to install system-level dependencies. On Windows, you might need to install [PortAudio](http://www.portaudio.com/archives/pa_stable_v190600_20161030.tgz) manually. For other OS, please refer to PyAudio's official documentation.

4.  **Download Vosk Model**:
    The Vosk ASR module will automatically download the necessary model the first time it's run. Ensure you have an active internet connection.

5.  **HuggingFace Token (Optional)**:
    If you plan to use HuggingFace models that require authentication or higher rate limits, set your HuggingFace token as an environment variable:
    ```bash
    export HUGGINGFACE_TOKEN="your_huggingface_token"
    # On Windows
    $env:HUGGINGFACE_TOKEN="your_huggingface_token"
    ```
6.  **NVIDIA API KEY (Optional)**:
    If you plan to use models provided by NVIDIA that require authentication or higher rate limits, set your NVIDIA API KEY as an environment variable:
    ```bash
    export NVIDIA_API_KEY="your_NVIDIA_API_KEY"
    # On Windows
    $env:NVIDIA_API_KEY="your_NVIDIA_API_KEY"
    ```
    Also if the select models requires a base URL refer to the respective model's NVIDIA documentation.

### 7. Usage

#### Running the Backend

To start the backend API server:

```bash
cd backend
uvicorn main:app --reload
```

This will start the FastAPI server, typically accessible at `http://localhost:8000`.

#### Running the Voice Assistant (ASR/TTS)[In-Development.. feel free to explore your own logic]

To use the real-time voice assistant functionality:

```bash
cd backend/loaders
python asr.py
```

The voice assistant will start listening for your input. Speak into your microphone, and it will transcribe your speech and respond verbally.

#### Interacting with the Frontend

Once the backend is running, you can open the frontend application in your web browser. The specific instructions for running the frontend will depend on the chosen framework (e.g., React, Vue, Angular). Please refer to the frontend's dedicated documentation for detailed steps.

Typically, you would navigate to the frontend directory and run a command like:

```bash
cd frontend
streamlit run app.py
```

Then, open your browser to `http://localhost:3000` (or the port specified by your frontend).
