from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


class UserProfile(BaseModel):
    user_name: str = Field(description="The name of the user.")
    user_location: str = Field(description="The location of the user.")
    interests: list[str] = Field(description="A list of the user's interests.")
    chat_history: list[dict] = Field(default_factory=list, description="A list of past query-response pairs.")


def get_across_thread_memory():
    return InMemoryStore()


def get_within_thread_memory():
    return MemorySaver()