import json
from evaluate import load_jsonl


def rules_extract(messages):
    text = " ".join(messages).lower()
    signals = []

    if "coming" in text or "trip" in text or "next month" in text or "march" in text:
        signals.append({
            "category": "intent",
            "subcategory": "trip_planning",
            "evidence": "travel-related mention",
            "confidence": 0.65,
        })

    if "book" in text or "reserve" in text or "reservation" in text or "steakhouse" in text:
        signals.append({
            "category": "intent",
            "subcategory": "restaurant_booking",
            "evidence": "booking-related mention",
            "confidence": 0.75,
        })

    if "suite" in text:
        signals.append({
            "category": "value",
            "subcategory": "suite_preference",
            "evidence": "suite mention",
            "confidence": 0.8,
        })

    if "budget isn't a concern" in text or "budget is not a concern" in text:
        signals.append({
            "category": "value",
            "subcategory": "high_spend",
            "evidence": "budget is not a concern",
            "confidence": 0.9,
        })

    if "anniversary" in text:
        signals.append({
            "category": "life_event",
            "subcategory": "anniversary",
            "evidence": "anniversary mention",
            "confidence": 0.95,
        })

    if "birthday" in text:
        signals.append({
            "category": "life_event",
            "subcategory": "birthday",
            "evidence": "birthday mention",
            "confidence": 0.95,
        })

    if "wynn" in text or "cosmo" in text:
        signals.append({
            "category": "competitive",
            "subcategory": "competitor_mention",
            "evidence": "competitor mention",
            "confidence": 0.8,
        })

    if "disappointed" in text or "frustrated" in text:
        signals.append({
            "category": "sentiment",
            "subcategory": "negative",
            "evidence": "negative wording",
            "confidence": 0.8,
        })

    return signals


def main():
    rows = load_jsonl("data/synthetic_dataset.jsonl")
    outputs = []

    for row in rows:
        outputs.append({
            "conversation_id": row["conversation_id"],
            "signals": rules_extract(row["messages"])
        })

    with open("data/rules_predictions.jsonl", "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(outputs)} rule-based predictions.")


if __name__ == "__main__":
    main()