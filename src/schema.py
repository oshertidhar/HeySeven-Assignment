from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Category = Literal[
    "intent",
    "value",
    "sentiment",
    "life_event",
    "competitive",
    "risk_or_friction",
]

class SignalDetection(BaseModel):
    category: Category
    subcategory: str
    evidence: str = Field(description="Short quote or snippet from the conversation")
    confidence: float = Field(ge=0.0, le=1.0)

class ConversationSignals(BaseModel):
    conversation_id: str
    messages: List[str]
    signals: List[SignalDetection]