# Higher Criticism (HC) Integration for Adaptive-RAG

This integration adds HC-based adaptive retrieval as a 4th baseline strategy (`hc_qa`) alongside the existing `nor_qa`, `oner_qa`, and `ircot_qa` strategies.

## Overview

**Higher Criticism (HC)** is a statistical method that adaptively determines how many documents to retrieve by analyzing the distribution of BM25 scores. Unlike fixed retrieval strategies (e.g., always retrieve 15 documents), HC selects the optimal number k based on the score distribution for each query.

## Key Features

- **Adaptive Selection**: Automatically determines optimal k for each query
- **Score-Agnostic**: Works with any scoring function (BM25 in this case)
- **Statistically Principled**: Based on rigorous statistical theory
- **Drop-in Replacement**: Can replace `ircot_qa` with minimal changes

## Architecture

### New Files Created

```
commaqa/hc/
├── __init__.py                    # Module exports
├── higher_criticism.py            # Core HC logic (BM25-adapted)
├── null_distribution.py           # Null distribution class
└── build_null_dist.py             # Script to build null distributions

commaqa/inference/
└── hc_retrieval.py                # HC retrieval participant

base_configs/
├── hc_qa_gpt_hotpotqa.jsonnet
├── hc_qa_gpt_2wikimultihopqa.jsonnet
├── hc_qa_gpt_musique.jsonnet
├── hc_qa_gpt_nq.jsonnet
├── hc_qa_gpt_trivia.jsonnet
├── hc_qa_gpt_squad.jsonnet
├── hc_qa_flan_t5_xl_hotpotqa.jsonnet
├── hc_qa_flan_t5_xl_2wikimultihopqa.jsonnet
├── hc_qa_flan_t5_xl_musique.jsonnet
├── hc_qa_flan_t5_xl_nq.jsonnet
├── hc_qa_flan_t5_xl_trivia.jsonnet
├── hc_qa_flan_t5_xl_squad.jsonnet
├── hc_qa_flan_t5_xxl_hotpotqa.jsonnet
├── hc_qa_flan_t5_xxl_2wikimultihopqa.jsonnet
├── hc_qa_flan_t5_xxl_musique.jsonnet
├── hc_qa_flan_t5_xxl_nq.jsonnet
├── hc_qa_flan_t5_xxl_trivia.jsonnet
└── hc_qa_flan_t5_xxl_squad.jsonnet
```

### Modified Files

- `commaqa/inference/constants.py`: Registered `HCRetrieveAndSelectParticipant`
- `run.py`: Added `hc_qa` to `instantiation_schemes` with hyperparameters
- `runner.py`: Added `hc_qa` to system choices
- `run_retrieval_dev.sh`: Added `hc_qa` to valid systems
- `run_retrieval_test.sh`: Added `hc_qa` to valid systems

## How It Works

### 1. Retrieval Phase
HC retrieves **more candidates than needed** (e.g., 30 documents) via BM25:
```python
retrieval_count = 30  # More than typical k=15
```

### 2. Score Analysis
Extracts BM25 scores and computes HC statistics:
```python
scores = [item["score"] for item in retrieval_results]
hc_result = hc.compute_hc_threshold(scores, gamma=0.1, min_hc=0.0)
k_selected = hc_result.k  # Adaptive k based on HC statistic
```

### 3. Adaptive Selection
Selects top-k documents based on HC:
```python
selected_docs = sorted_docs[:k_selected]  # Only use optimal k
```

### Key Parameters

- **bm25_retrieval_count**: Number of candidates to retrieve (default: 30)
- **gamma**: HC search window (default: 0.1 = top 10% of candidates)
- **min_hc**: Minimum HC statistic threshold (default: 0.0)
- **null_dist_path**: Path to pre-computed null distribution

## Usage

### Step 1: Build Null Distributions

Before running HC, you must build null distributions for each dataset:

```bash
# Make sure Elasticsearch is running on port 9200
python -m commaqa.hc.build_null_dist hotpotqa dev_500
python -m commaqa.hc.build_null_dist 2wikimultihopqa dev_500
python -m commaqa.hc.build_null_dist musique dev_500
python -m commaqa.hc.build_null_dist nq dev_500
python -m commaqa.hc.build_null_dist trivia dev_500
python -m commaqa.hc.build_null_dist squad dev_500
```

This creates null distribution files in `data/hc_null_distributions/`:
```
data/hc_null_distributions/
├── hotpotqa_null_dist.pkl
├── 2wikimultihopqa_null_dist.pkl
├── musique_null_dist.pkl
├── nq_null_dist.pkl
├── trivia_null_dist.pkl
└── squad_null_dist.pkl
```

**What it does:**
- Samples 100 queries from the dev set
- Retrieves 100 documents per query via BM25
- Collects BM25 scores (mostly non-relevant, since datasets have 87-93% noise)
- Stores score distribution as null distribution

### Step 2: Run HC on Dev Set

```bash
# Run with GPT
bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010

# Run with FLAN-T5-XL
bash run_retrieval_dev.sh hc_qa flan-t5-xl hotpotqa 8010

# Run with FLAN-T5-XXL
bash run_retrieval_dev.sh hc_qa flan-t5-xxl hotpotqa 8010
```

This will:
1. Generate config files with different hyperparameters
2. Run prediction on dev_500
3. Evaluate results
4. Show summary

### Step 3: Run HC on Test Set

```bash
bash run_retrieval_test.sh hc_qa gpt hotpotqa 8010
```

This uses the best hyperparameters from dev to run on the full test set.

### Step 4: Compare Results

Compare HC results to IRCoT baseline:

```bash
# HC results
cat predictions/dev_500/hc_qa_gpt_hotpotqa_*/metrics.json

# IRCoT baseline (for reference)
cat predictions/dev_500/ircot_qa_gpt_hotpotqa_*/metrics.json
```

Expected metrics format:
```json
{
  "f1": 0.XXX,
  "exact_match": 0.XXX
}
```

## Hyperparameter Tuning

Current defaults in `run.py`:
```python
"hc_qa": {
    "bm25_retrieval_count": ["30"],  # Try 20, 30, 40
    "gamma": ["0.1"],                # Try 0.05, 0.1, 0.2
    "min_hc": ["0.0"],               # Try 0.0, 0.5, 1.0
}
```

To tune hyperparameters:
1. Edit `run.py` instantiation_schemes["hc_qa"]
2. Run dev script: `bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010`
3. Check results: `python runner.py hc_qa gpt hotpotqa summarize --prompt_set 1 --sample_size 500 --llm_port_num 8010`
4. Select best HP based on F1/EM

## Implementation Details

### HC Algorithm

1. **Compute p-values**: For each BM25 score, compute p-value under null distribution
2. **Sort by significance**: Order candidates by p-value (ascending)
3. **Compute HC statistic**: For each k in top gamma%, compute HC(k) = sqrt(k) * max_i(sqrt(i/k) - p[i])
4. **Select k**: Choose k that maximizes HC statistic (if above min_hc threshold)

### BM25 Adaptation

The original HC implementation used cosine similarity with vector databases. This integration adapts it for BM25:

- **Original**: `vector_db.retrieve()` → cosine similarities
- **Adapted**: Elasticsearch BM25 → BM25 scores

Key changes:
- `NullDistribution` stores BM25 score distribution (not cosine similarities)
- `BM25NullDistribution.build_from_dataset()` uses Elasticsearch retrieval
- `HCRetrieveAndSelectParticipant` extracts BM25 scores from retrieval response

### Integration Pattern

HC follows the same pattern as `oner_qa` (single-step retrieval):
1. Retrieve candidates via BM25
2. Apply HC selection
3. Pass selected documents to QA model

This is simpler than `ircot_qa` (iterative retrieval), making HC a drop-in replacement.

## Troubleshooting

### Error: "Null distribution not found"
**Solution**: Run `python -m commaqa.hc.build_null_dist <dataset> dev_500` first

### Error: "Retrieval failed"
**Solution**: Check Elasticsearch is running: `curl http://localhost:9200`

### Error: "No module named 'commaqa.hc'"
**Solution**: Ensure `commaqa/hc/__init__.py` exists and is not empty

### Low F1/EM scores
**Possible causes**:
- Null distribution not representative (try more queries: `--num_queries 200`)
- Hyperparameters not tuned (try different gamma/min_hc values)
- BM25 retrieval issues (check retriever logs)

## Comparison to Baselines

| Strategy | Retrieval | Selection | Typical k |
|----------|-----------|-----------|-----------|
| nor_qa   | None      | None      | 0 |
| oner_qa  | BM25 × 1  | Top-k     | 15 |
| ircot_qa | BM25 × N  | Top-k per iteration | 6 per iteration |
| **hc_qa** | **BM25 × 1** | **HC-based** | **Adaptive (0-30)** |

**Key differences**:
- HC is **adaptive**: k varies per query based on score distribution
- HC is **statistically principled**: uses null distribution and HC statistic
- HC is **single-step**: like oner_qa, not iterative like ircot_qa

## Expected Performance

Based on IRCoT baseline (dev_500):
- **IRCoT**: F1=0.604, EM=0.446
- **HC**: Expected similar or better (to be determined)

HC should excel on queries where:
- Score separation is clear (relevant docs have much higher BM25 than non-relevant)
- Fixed k is suboptimal (either too many or too few documents)

## Next Steps

1. **Run experiments**: Compare HC to IRCoT on all datasets
2. **Hyperparameter tuning**: Find optimal gamma/min_hc per dataset
3. **Analysis**: When does HC outperform fixed-k strategies?
4. **Integration with classifier**: Optionally add HC as 4th option in adaptive routing

## Notes

- **Original HC code preserved**: `hc_copied/hc/` is untouched per user constraint
- **No dependencies added**: Uses existing libraries (numpy, requests)
- **Compatible with existing pipeline**: Works with dev/test scripts
- **Minimal changes**: Only adds new files + registers participant
