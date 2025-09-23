import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, END, START
from trustcall import create_extractor

from llm import get_llm
from memory_utils import UserProfile, get_across_thread_memory, get_within_thread_memory
from rag_pipeline import RAGPipeline # Import RAGPipeline


# Set up Google Generative AI
# Ensure GOOGLE_API_KEY is set in your environment variables
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

model = get_llm()
model_with_structure = get_llm().with_structured_output(UserProfile)

CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history. \
This will be saved for long-term memory. If there is an existing memory, simply update it. \
Here is the existing memory (it may be empty): {memory}"""

MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user."""

class Chatbot:
    def __init__(self):
        self.builder = StateGraph(MessagesState)
        self.builder.add_node("chatbot", self.call_model)
        self.builder.add_node("write_memory", self.write_memory)
        self.builder.set_entry_point("chatbot")
        self.builder.add_edge("chatbot", "write_memory")
        self.builder.add_edge("write_memory", END)

        self.across_thread_memory = get_across_thread_memory()
        self.within_thread_memory = get_within_thread_memory()
        self.rag_pipeline = RAGPipeline() # Initialize RAGPipeline

        self.graph = self.builder.compile(
            checkpointer=self.within_thread_memory,
            store=self.across_thread_memory
        )

    def call_model(self, state: MessagesState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        namespace = ("memory", user_id)
        existing_memory = self.across_thread_memory.get(namespace, "user_memory")

        formatted_memory = ""
        if existing_memory and existing_memory.value:
            memory_dict = existing_memory.value
            formatted_memory = (
                f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
                f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
                f"Interests: {', '.join(memory_dict.get('interests', []))}"
            )

        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

        # Extract the latest user message
        user_message_content = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                user_message_content = message.content
                break

        print(f"User message content: {user_message_content}")
        # Retrieve documents using RAGPipeline
        retrieved_docs = []
        if user_message_content:
            retrieved_docs = self.rag_pipeline.retrieve_documents(user_message_content)
            print(f"Retrieved documents: {retrieved_docs}")
            if retrieved_docs:
                system_msg += "\n\nRelevant Documents:\n"
                for doc in retrieved_docs:
                    system_msg += f"- {doc.page_content}\n"
        print(f"System message after RAG: {system_msg}")

        response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])

        return {"messages": [response]}

    def write_memory(self, state: MessagesState, config: RunnableConfig):
        user_id = config["configurable"]["user_id"]
        namespace = ("memory", user_id)
        existing_memory = self.across_thread_memory.get(namespace, "user_memory")

        formatted_memory = ""
        if existing_memory and existing_memory.value:
            memory_dict = existing_memory.value
            formatted_memory = (
                f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
                f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
                f"Interests: {', '.join(memory_dict.get('interests', []))}"
            )

        new_memory = model_with_structure.invoke([SystemMessage(content=CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory))] + state["messages"])

        if new_memory:
            key = "user_memory"
            self.across_thread_memory.put(namespace, key, new_memory.model_dump())
        else:
            print("Warning: model_with_structure.invoke returned None for new_memory. Memory not updated.")

        return state

    def clear_memory(self):
        self.across_thread_memory = get_across_thread_memory()
        self.within_thread_memory = get_within_thread_memory()
        print("Chatbot memory cleared.")

    def invoke(self, message: str, thread_id: str, user_id: str):
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        response = self.graph.invoke({"messages": [HumanMessage(content=message)]}, config)
        llm_response = response["messages"][-1].content
        print(f"LLM Response: {llm_response}") # Add this line to print the LLM response
        return llm_response
