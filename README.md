# Signal Extraction System – Take Home Assignment

## Overview

This project implements a system for extracting actionable guest signals from short host–guest conversations in a casino context.

Instead of treating the task as simple classification, I framed it as a **structured information extraction problem**, where multiple signals can be identified per conversation and returned in a consistent JSON format.

Each detected signal includes:

* category
* subcategory
* supporting evidence
* confidence score

---

## System Design

The system is composed of two main components:

### 1. Rule-Based Baseline (Implemented)

A deterministic extractor based on keyword and pattern matching.

* Fast and interpretable
* Serves as a baseline for comparison
* Limited in capturing implicit or contextual signals

### 2. LLM-Based Extractor (Modular Design)

A structured extraction pipeline using an LLM (not executed due to API constraints, but fully implemented and ready).

* Designed to improve recall and semantic understanding
* Outputs structured JSON aligned with the schema
* Easily pluggable into the evaluation pipeline


## Implementation Notes

- The system is designed around a shared schema (`schema.py`) to ensure consistent structured outputs across different extraction methods.
- The LLM-based extractor (`llm_extractor.py`) is implemented as a modular component, allowing seamless integration into the pipeline once API access is available.
---

## Signal Schema

Signals are extracted across the following categories:

* **intent** (trip planning, bookings, host requests)
* **value** (high spend, premium preference, group size, VIP history)
* **sentiment** (positive, negative)
* **life_event** (birthday, anniversary, promotion, etc.)
* **competitive** (competitor mentions, offers, loyalty)
* **risk_or_friction** (complaints, frustration, churn risk)

---

## Data Strategy

Since no labeled dataset was provided, I created a **synthetic benchmark dataset** that includes:

* Single-signal conversations
* Multi-signal conversations
* Ambiguous and borderline cases
* Negative examples (no signal)

This allows controlled evaluation of extraction quality.

Dataset file:

```
data/synthetic_dataset.jsonl
```

---

## Evaluation

Evaluation is performed by comparing predicted signals to ground truth labels using:

* Precision
* Recall
* F1 score

Matching is done at the `(category, subcategory)` level.

---

## Results (Rule-Based Baseline)

### Overall Performance

```
Precision: 0.52  
Recall:    0.29  
F1 Score:  0.37  
```

### Per-Category Breakdown

| Category         | Precision | Recall | F1   |
| ---------------- | --------- | ------ | ---- |
| intent           | 0.25      | 0.27   | 0.26 |
| value            | 0.75      | 0.27   | 0.40 |
| life_event       | 1.00      | 0.43   | 0.60 |
| risk_or_friction | 0.00      | 0.00   | 0.00 |
| sentiment        | 1.00      | 0.50   | 0.67 |
| competitive      | 0.50      | 0.25   | 0.33 |

---

## Analysis

The rule-based system achieves **moderate precision (~0.52)** but **low recall (~0.29)**.

This indicates that:

* When signals are detected, they are often correct
* However, many signals are missed due to limited coverage

### Observations:

* **Life events and sentiment** are easier to detect (clear keywords)
* **Intent and friction signals** require deeper contextual understanding
* **Risk/friction signals were not captured well**, highlighting limitations of rule-based approaches

---

## Limitations

* Rule-based approach lacks semantic understanding
* Cannot generalize to unseen phrasing
* Misses implicit or nuanced signals
* Sensitive to wording variations

---

## Future Improvements

* Use LLM-based extraction to improve recall and flexibility
* Add human-labeled validation dataset
* Improve confidence calibration
* Perform error-driven iteration
* Expand synthetic dataset coverage

---

## Example Error Cases

* Missed implicit intent when phrasing was indirect
* Failure to detect frustration without explicit keywords
* Confusion between intent and value in mixed sentences

---

## How to Run

### 1. Generate Dataset

```bash
python3 src/synthetic_data.py
```

### 2. Run Rule-Based Extraction

```bash
python3 src/rules_baseline.py
```

### 3. Evaluate

```bash
python3 src/evaluate.py
```

---

## Project Structure

```
src/
  synthetic_data.py        # synthetic dataset generation
  rules_baseline.py        # rule-based signal extractor
  evaluate.py              # evaluation script (precision / recall / F1)
  llm_extractor.py         # modular LLM-based extractor (not executed)
  schema.py                # shared schema for structured outputs

data/
  synthetic_dataset.jsonl
  rules_predictions.jsonl
```

---

## Final Note

This solution focuses on building a **clear, extensible pipeline** with:

* structured outputs
* reproducible evaluation
* modular architecture

The rule-based system serves as a baseline, while the LLM extractor is designed as a natural next step to improve performance in a production setting.
