import json
from typing import List, Dict, Tuple
from collections import Counter


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize_signal(signal: Dict) -> Tuple[str, str]:
    return (
        signal["category"].strip().lower(),
        signal["subcategory"].strip().lower(),
    )


def score_predictions(gold_rows: List[Dict], pred_rows: List[Dict]) -> Dict:
    gold_map = {row["conversation_id"]: row for row in gold_rows}
    pred_map = {row["conversation_id"]: row for row in pred_rows}

    tp = 0
    fp = 0
    fn = 0

    for conv_id, gold_row in gold_map.items():
        gold_signals = set(normalize_signal(s) for s in gold_row.get("signals", []))
        pred_signals = set(normalize_signal(s) for s in pred_map.get(conv_id, {}).get("signals", []))

        tp += len(gold_signals & pred_signals)
        fp += len(pred_signals - gold_signals)
        fn += len(gold_signals - pred_signals)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def category_breakdown(gold_rows: List[Dict], pred_rows: List[Dict]) -> Dict[str, Dict]:
    gold_map = {row["conversation_id"]: row for row in gold_rows}
    pred_map = {row["conversation_id"]: row for row in pred_rows}

    counts = {}

    for conv_id, gold_row in gold_map.items():
        gold_signals = set(normalize_signal(s) for s in gold_row.get("signals", []))
        pred_signals = set(normalize_signal(s) for s in pred_map.get(conv_id, {}).get("signals", []))

        cats = set([g[0] for g in gold_signals] + [p[0] for p in pred_signals])
        for cat in cats:
            if cat not in counts:
                counts[cat] = {"tp": 0, "fp": 0, "fn": 0}

            gold_cat = set(x for x in gold_signals if x[0] == cat)
            pred_cat = set(x for x in pred_signals if x[0] == cat)

            counts[cat]["tp"] += len(gold_cat & pred_cat)
            counts[cat]["fp"] += len(pred_cat - gold_cat)
            counts[cat]["fn"] += len(gold_cat - pred_cat)

    for cat, d in counts.items():
        p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        d["precision"] = p
        d["recall"] = r
        d["f1"] = f1

    return counts


if __name__ == "__main__":
    gold = load_jsonl("data/synthetic_dataset.jsonl")
    pred = load_jsonl("data/rules_predictions.jsonl")

    overall = score_predictions(gold, pred)
    print("Overall:", json.dumps(overall, indent=2))

    breakdown = category_breakdown(gold, pred)
    print("By category:", json.dumps(breakdown, indent=2))