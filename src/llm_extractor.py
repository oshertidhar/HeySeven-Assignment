import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI


SYSTEM_PROMPT = """
You are an information extraction system for casino guest-host conversations.

Your task is to identify and extract actionable guest signals from a conversation.
Return only signals that are explicitly or strongly implicitly supported by the conversation.

Allowed categories:
- intent
- value
- sentiment
- life_event
- competitive
- risk_or_friction

Examples of subcategories:
intent: trip_planning, room_booking, restaurant_booking, show_booking, host_contact_request
value: high_spend, premium_preference, group_size_large, suite_preference, vip_history
sentiment: positive, negative, neutral
life_event: anniversary, birthday, promotion, bachelor_bachelorette, reunion
competitive: competitor_mention, competitor_offer, loyalty_elsewhere
risk_or_friction: complaint, unmet_expectation, churn_risk, frustration

Rules:
1. Return a JSON object only.
2. Extract multiple signals if applicable.
3. Each signal must include:
   - category
   - subcategory
   - evidence: short exact quote or short snippet
   - confidence: float between 0 and 1
4. If no meaningful signals exist, return an empty list.
5. Do not invent facts.
"""


class SignalDetection(BaseModel):
    category: str
    subcategory: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class ConversationExtraction(BaseModel):
    conversation_id: str
    signals: List[SignalDetection]


def build_user_prompt(conversation_id: str, messages: List[str]) -> str:
    joined = "\n".join([f"- {m}" for m in messages])
    return f"""
Conversation ID: {conversation_id}

Conversation:
{joined}

Return JSON in this format:
{{
  "conversation_id": "{conversation_id}",
  "signals": [
    {{
      "category": "intent",
      "subcategory": "restaurant_booking",
      "evidence": "book the steakhouse",
      "confidence": 0.91
    }}
  ]
}}
""".strip()


def extract_signals_openai(
    conversation_id: str,
    messages: List[str],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(conversation_id, messages)},
        ],
    )

    content = response.choices[0].message.content
    result = json.loads(content)

    # Light validation / cleanup
    if "conversation_id" not in result:
        result["conversation_id"] = conversation_id
    if "signals" not in result:
        result["signals"] = []

    return result


if __name__ == "__main__":
    sample_messages = [
        "We're coming next month for our anniversary.",
        "Can you also help us reserve a nice dinner?"
    ]
    out = extract_signals_openai("demo_001", sample_messages)
    print(json.dumps(out, indent=2))