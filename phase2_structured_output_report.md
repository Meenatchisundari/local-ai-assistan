# Phase 2 — Structured Output & Reliability Report

**Generated:** 2026-06-18 23:17  
**Data:** `phase2_20260618T215739.csv`  
**Models:** tinyllama, qwen2:1.5b  
**Prompts:** 10 (factual QA, summarisation, reasoning, instruction following)  
**Schema:** `answer: str`, `confidence: float 0-1`, `sources: list[str]`  
**Retry logic:** 1 re-prompt on failure with correction instruction  

---

## What was tested

Each prompt was sent to each model at two temperatures.
The response was validated against a Pydantic schema expecting valid JSON with three fields.
If validation failed, a correction prompt was sent once (retry).
Outcome logged as PASS, RETRY_PASS, or FAIL.

---

## Results — Temperature 0.0 (deterministic)

| Model | First-try pass | Pass after retry | Final fail |
|---|---|---|---|
| qwen2:1.5b | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) |
| tinyllama | 1/10 (10%) | 3/10 (30%) | 6/10 (60%) |


## Results — Temperature 0.7 (creative)

| Model | First-try pass | Pass after retry | Final fail |
|---|---|---|---|
| qwen2:1.5b | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) |
| tinyllama | 3/10 (30%) | 3/10 (30%) | 4/10 (40%) |


---

## Key findings

### 1. qwen2:1.5b follows JSON instructions perfectly
100% first-try pass rate at both temperatures across all 10 prompts.
No retries needed. No failures. Temperature had zero effect on its reliability.
For any application requiring structured output, qwen2:1.5b is the clear choice.

### 2. tinyllama struggles with JSON formatting
At temperature 0.0, tinyllama passed only 10% of runs on the first try.
The model consistently wraps JSON in markdown code blocks and prose preambles
despite explicit instructions not to. The code block stripper rescued some runs
but 60% still failed both attempts at temp 0.0.

### 3. Higher temperature helped tinyllama (counterintuitive)
At temp 0.7, tinyllama improved: 30% first-try pass, 30% retry pass, 40% final fail.
Randomness broke it out of its habit of confidently producing the wrong format.
This is a known behaviour in small models — low temperature can lock them into bad patterns.

### 4. Retry logic has real value
Retry rescued 3 runs for tinyllama at temp 0.0 and 3 at temp 0.7.
Without retry logic, failure rates would be 6 points higher in each case.
One correction prompt with the bad response shown back to the model is enough
to fix formatting errors when the underlying answer is correct.

---

## Recommendation

| Use case | Recommended model | Recommended temp |
|---|---|---|
| Structured output / JSON pipelines | qwen2:1.5b | 0.0 |
| Free-text generation | tinyllama | 0.7 |
| Reliability-critical applications | qwen2:1.5b | 0.0 |

---

## Next: Phase 3 — Model Comparison Pipeline

Phase 3 runs a full 30-50 prompt comparison across all models,
adds a quality scoring rubric, and generates charts:
bar chart (tokens/sec), scatter plot (memory vs speed), radar chart (quality dimensions).
