import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

def get_llm():
    # For LLM, we'll use NVIDIA
    client = OpenAI(
        base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key = os.getenv("NVIDIA_API_KEY")
    )
    llm = ChatNVIDIA(model="nvidia/nvidia-nemotron-nano-9b-v2", temperature=0.4, max_tokens=2048)
    return llm