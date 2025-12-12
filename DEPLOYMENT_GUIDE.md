# HC Integration - Complete Deployment Guide

This guide provides step-by-step instructions to deploy and run the HC integration on a fresh clone of the repository.

## 📋 Pre-Deployment Checklist

Verify these files exist in your repository:

### New Files (24 total)

**HC Module** (`commaqa/hc/`):
- [ ] `__init__.py` (311 bytes)
- [ ] `higher_criticism.py` (6,088 bytes)
- [ ] `null_distribution.py` (9,189 bytes)
- [ ] `build_null_dist.py` (7,017 bytes)

**HC Retrieval Participant** (`commaqa/inference/`):
- [ ] `hc_retrieval.py` (11,596 bytes)

**Configuration Files** (`base_configs/`):
- [ ] `hc_qa_gpt_hotpotqa.jsonnet`
- [ ] `hc_qa_gpt_2wikimultihopqa.jsonnet`
- [ ] `hc_qa_gpt_musique.jsonnet`
- [ ] `hc_qa_gpt_nq.jsonnet`
- [ ] `hc_qa_gpt_trivia.jsonnet`
- [ ] `hc_qa_gpt_squad.jsonnet`
- [ ] `hc_qa_flan_t5_xl_hotpotqa.jsonnet`
- [ ] `hc_qa_flan_t5_xl_2wikimultihopqa.jsonnet`
- [ ] `hc_qa_flan_t5_xl_musique.jsonnet`
- [ ] `hc_qa_flan_t5_xl_nq.jsonnet`
- [ ] `hc_qa_flan_t5_xl_trivia.jsonnet`
- [ ] `hc_qa_flan_t5_xl_squad.jsonnet`
- [ ] `hc_qa_flan_t5_xxl_hotpotqa.jsonnet`
- [ ] `hc_qa_flan_t5_xxl_2wikimultihopqa.jsonnet`
- [ ] `hc_qa_flan_t5_xxl_musique.jsonnet`
- [ ] `hc_qa_flan_t5_xxl_nq.jsonnet`
- [ ] `hc_qa_flan_t5_xxl_trivia.jsonnet`
- [ ] `hc_qa_flan_t5_xxl_squad.jsonnet`

**Documentation**:
- [ ] `HC_INTEGRATION_README.md`
- [ ] `HC_QUICKSTART.md`

### Modified Files (5 total)

Check these files contain HC additions:

```bash
# Should show "hc_retrieve_and_select"
grep "hc_retrieve_and_select" commaqa/inference/constants.py

# Should show "hc_qa" in instantiation_schemes
grep -A 5 '"hc_qa"' run.py

# Should show hc_qa in valid_systems
grep "valid_systems=" run_retrieval_dev.sh
grep "valid_systems=" run_retrieval_test.sh

# Should show hc_qa in choices
grep "hc_qa" runner.py
```

---

## 🚀 Complete Deployment Steps

### Step 1: Clone Repository to Linux VM

```bash
# On your Linux GPU VM
git clone https://github.com/starsuzi/Adaptive-RAG.git
cd Adaptive-RAG
git checkout lihu/hc_implementation
```

### Step 2: Set Up Environment

```bash
# Create conda environment (Python 3.8)
conda create -n adaptive_rag python=3.8
conda activate adaptive_rag

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('Transformers installed')"
python -c "import elasticsearch; print('Elasticsearch client installed')"
```

### Step 3: Download Data (Following Original README)

```bash
# Download datasets
bash download/download_all_datasets.sh

# Download Wikipedia corpus
bash download/download_processed_corpus.sh

# Verify data exists
ls -lh data/hotpotqa/
ls -lh data/corpus/
```

### Step 4: Start Elasticsearch Server

```bash
# Download and start Elasticsearch 7.10.2
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/

# Start Elasticsearch
./bin/elasticsearch

# In another terminal, verify it's running
curl http://localhost:9200
# Should return JSON with cluster info
```

### Step 5: Index Wikipedia Corpus

```bash
# In a new terminal (keep Elasticsearch running)
cd Adaptive-RAG

# Index the corpus (takes ~1-2 hours)
python processing_scripts/index_corpus_to_elasticsearch.py \
    --corpus_file data/corpus/wiki_all.jsonl \
    --port 9200

# Verify indexing
curl -X GET "localhost:9200/_cat/indices?v"
# Should show indices for each dataset
```

### Step 6: Start Retriever Server

```bash
# Start retriever server on port 9200 (Elasticsearch default)
# Already running from Step 4
# Verify:
curl http://localhost:9200/_cluster/health
```

### Step 7: Start LLM Server

**Option A: GPT (OpenAI)**
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Start GPT server on port 8010
python llm_server/run_server_openai.py \
    --model gpt-4o-mini \
    --port 8010

# In another terminal, verify:
curl http://localhost:8010/health
```

**Option B: FLAN-T5 (Local GPU)**
```bash
# Start FLAN-T5-XL server on port 8010
python llm_server/run_server_hf.py \
    --model google/flan-t5-xl \
    --port 8010 \
    --device cuda

# Or FLAN-T5-XXL (requires more GPU memory)
python llm_server/run_server_hf.py \
    --model google/flan-t5-xxl \
    --port 8010 \
    --device cuda
```

### Step 8: Verify HC Integration

```bash
# Verify HC module can be imported
python -c "
from commaqa.hc import HigherCriticism, NullDistribution, BM25NullDistribution
from commaqa.inference.hc_retrieval import HCRetrieveAndSelectParticipant
print('✓ All HC modules imported successfully')
"

# Verify config files exist
ls base_configs/hc_qa_gpt_hotpotqa.jsonnet
ls base_configs/hc_qa_flan_t5_xl_hotpotqa.jsonnet
ls base_configs/hc_qa_flan_t5_xxl_hotpotqa.jsonnet

# Verify HC is registered in runner
python runner.py --help | grep -A 1 "system"
# Should show hc_qa in choices
```

### Step 9: Build Null Distributions (CRITICAL)

This step is **REQUIRED** before running HC. It builds the statistical null distribution needed for HC threshold computation.

```bash
# Build null distribution for HotpotQA (example)
# Takes ~5-10 minutes per dataset
python -m commaqa.hc.build_null_dist hotpotqa dev_500

# Expected output:
# INFO:__main__:Loaded 500 examples from data/hotpotqa/dev_500.jsonl
# INFO:__main__:Building null distribution for hotpotqa
# INFO:__main__:Processing query 1/100: What is...
# ...
# INFO:__main__:Null distribution statistics:
# INFO:__main__:  Count: 10000
# INFO:__main__:  Mean: 15.234
# INFO:__main__:  Std: 3.456
# INFO:__main__:  Min: 8.123
# INFO:__main__:  Max: 28.456
# INFO:__main__:Saved null distribution to data/hc_null_distributions/hotpotqa_null_dist.pkl
```

**Build for all datasets:**
```bash
# Build null distributions for all 6 datasets
# Total time: ~1 hour
python -m commaqa.hc.build_null_dist hotpotqa dev_500
python -m commaqa.hc.build_null_dist 2wikimultihopqa dev_500
python -m commaqa.hc.build_null_dist musique dev_500
python -m commaqa.hc.build_null_dist nq dev_500
python -m commaqa.hc.build_null_dist trivia dev_500
python -m commaqa.hc.build_null_dist squad dev_500

# Verify null distributions were created
ls -lh data/hc_null_distributions/
# Should show 6 .pkl files (one per dataset)
```

**What this does:**
- Samples 100 queries from dev_500 set
- Retrieves 100 documents per query via BM25
- Collects BM25 scores from non-relevant documents (datasets have 87-93% noise)
- Computes score distribution statistics (mean, std, min, max)
- Saves distribution to `data/hc_null_distributions/{dataset}_null_dist.pkl`

**Why it's needed:**
HC computes p-values by comparing BM25 scores to the null distribution. Without this, HC cannot determine which documents are statistically significant.

### Step 10: Run HC on Dev Set

Now you're ready to run HC!

```bash
# Run HC with GPT on HotpotQA dev set (500 examples)
bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010
```

**What this script does:**

1. **Writes config files** with different hyperparameters:
   ```
   configs/hc_qa_gpt_hotpotqa_dev_500_p1_bm25_retrieval_count=30_gamma=0.1_min_hc=0.0.jsonnet
   ```

2. **Runs prediction** on dev_500 (500 examples):
   - Loads null distribution from `data/hc_null_distributions/hotpotqa_null_dist.pkl`
   - For each query:
     - Retrieves 30 candidates via BM25
     - Extracts BM25 scores
     - Computes HC statistic over top 10% (gamma=0.1)
     - Selects optimal k documents
     - Passes to GPT for QA
   - Saves predictions to `predictions/dev_500/hc_qa_gpt_hotpotqa_*/predictions.jsonl`

3. **Evaluates results**:
   - Computes F1 and EM metrics
   - Saves to `predictions/dev_500/hc_qa_gpt_hotpotqa_*/metrics.json`

4. **Summarizes results**:
   - Prints summary table with F1/EM for different hyperparameters

**Expected runtime:**
- ~30-60 minutes for 500 examples with GPT
- ~1-2 hours with FLAN-T5

### Step 11: Check Results

```bash
# View HC results
cat predictions/dev_500/hc_qa_gpt_hotpotqa_bm25_retrieval_count=30_gamma=0.1_min_hc=0.0/metrics.json

# Expected output:
# {
#   "f1": 0.XXX,
#   "exact_match": 0.XXX
# }

# Compare to IRCoT baseline (your previous results)
cat predictions/dev_500/ircot_qa_gpt_hotpotqa_*/metrics.json
# IRCoT baseline: F1=0.604, EM=0.446
```

### Step 12: Run on Other Datasets (Optional)

```bash
# Run on 2WikiMultiHopQA
bash run_retrieval_dev.sh hc_qa gpt 2wikimultihopqa 8010

# Run on Musique
bash run_retrieval_dev.sh hc_qa gpt musique 8010

# Run with FLAN-T5-XL
bash run_retrieval_dev.sh hc_qa flan-t5-xl hotpotqa 8010
```

### Step 13: Run on Test Set (After Dev Tuning)

Once satisfied with dev results:

```bash
# Run HC on full test set with best hyperparameters
bash run_retrieval_test.sh hc_qa gpt hotpotqa 8010

# Results saved to:
# predictions/test/hc_qa_gpt_hotpotqa_*/
```

---

## 🔍 Verification Commands

### Before Running HC

```bash
# 1. Check Elasticsearch is running
curl http://localhost:9200/_cluster/health
# Should return: "status":"green" or "yellow"

# 2. Check LLM server is running
curl http://localhost:8010/health
# Should return: {"status":"ok"} or similar

# 3. Check data exists
ls data/hotpotqa/dev_500.jsonl
ls data/corpus/wiki_all.jsonl

# 4. Check null distribution exists
ls data/hc_null_distributions/hotpotqa_null_dist.pkl

# 5. Verify HC module
python -c "from commaqa.hc import HigherCriticism; print('OK')"

# 6. Verify config file
ls base_configs/hc_qa_gpt_hotpotqa.jsonnet
```

### After Running HC

```bash
# 1. Check predictions were generated
ls predictions/dev_500/hc_qa_gpt_hotpotqa_*/predictions.jsonl

# 2. Check metrics were computed
cat predictions/dev_500/hc_qa_gpt_hotpotqa_*/metrics.json

# 3. Count predictions
wc -l predictions/dev_500/hc_qa_gpt_hotpotqa_*/predictions.jsonl
# Should show 500 lines

# 4. View sample prediction
head -n 1 predictions/dev_500/hc_qa_gpt_hotpotqa_*/predictions.jsonl | python -m json.tool

# 5. Compare to baseline
echo "HC Results:"
cat predictions/dev_500/hc_qa_gpt_hotpotqa_*/metrics.json
echo "IRCoT Baseline:"
cat predictions/dev_500/ircot_qa_gpt_hotpotqa_*/metrics.json
```

---

## 🐛 Troubleshooting

### Error: "Null distribution not found"

**Cause:** Forgot to run `build_null_dist.py`

**Fix:**
```bash
python -m commaqa.hc.build_null_dist hotpotqa dev_500
```

### Error: "Connection refused to localhost:9200"

**Cause:** Elasticsearch not running

**Fix:**
```bash
cd elasticsearch-7.10.2/
./bin/elasticsearch &
```

### Error: "Connection refused to localhost:8010"

**Cause:** LLM server not running

**Fix:**
```bash
# For GPT
export OPENAI_API_KEY="your-key"
python llm_server/run_server_openai.py --model gpt-4o-mini --port 8010

# For FLAN-T5
python llm_server/run_server_hf.py --model google/flan-t5-xl --port 8010 --device cuda
```

### Error: "No module named 'commaqa.hc'"

**Cause:** Wrong directory or Python path issue

**Fix:**
```bash
cd Adaptive-RAG
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "from commaqa.hc import HigherCriticism"
```

### Error: "Index not found"

**Cause:** Corpus not indexed to Elasticsearch

**Fix:**
```bash
python processing_scripts/index_corpus_to_elasticsearch.py \
    --corpus_file data/corpus/wiki_all.jsonl \
    --port 9200
```

### HC selecting 0 documents

**Cause:** Null distribution not representative or min_hc too high

**Fix:**
1. Rebuild null distribution with more queries:
   ```bash
   python -m commaqa.hc.build_null_dist hotpotqa dev_500 --num_queries 200
   ```
2. Lower min_hc threshold in `run.py`:
   ```python
   "hc_qa": {
       "min_hc": ["-1.0", "0.0"],  # Try negative values
   }
   ```

### Low F1/EM scores

**Possible causes:**
1. Hyperparameters not tuned - try different gamma values
2. Null distribution not representative - rebuild with more queries
3. BM25 retrieval issues - check Elasticsearch logs

**Fix:**
```bash
# Try different hyperparameters
# Edit run.py line ~811:
"hc_qa": {
    "bm25_retrieval_count": ["20", "30", "40"],
    "gamma": ["0.05", "0.1", "0.2"],
    "min_hc": ["0.0", "0.5"],
}

# Re-run dev script
bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010
```

---

## 📊 Expected Timeline

| Step | Time | Parallel? |
|------|------|-----------|
| Environment setup | 10-15 min | No |
| Download data | 30-60 min | No |
| Index corpus | 1-2 hours | No |
| Start servers | 5 min | Yes |
| Build null distributions (6 datasets) | 1 hour | Yes* |
| Run HC dev (1 dataset) | 30-60 min | No |
| Full comparison (6 datasets × 3 models) | 18+ hours | Yes* |

*Can run in parallel across multiple GPUs/machines

---

## 🎯 Quick Reference

### One-Command Test (After Setup)

```bash
# Complete test on HotpotQA:
python -m commaqa.hc.build_null_dist hotpotqa dev_500 && \
bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010 && \
cat predictions/dev_500/hc_qa_gpt_hotpotqa_*/metrics.json
```

### Important File Locations

```
data/hc_null_distributions/           # Null distributions (build first!)
base_configs/hc_qa_*.jsonnet          # HC config templates
commaqa/hc/                           # HC implementation
commaqa/inference/hc_retrieval.py     # HC retrieval participant
predictions/dev_500/hc_qa_*/          # HC results
```

### Key Parameters (in run.py)

```python
"hc_qa": {
    "bm25_retrieval_count": ["30"],  # Candidates to retrieve
    "gamma": ["0.1"],                # HC search window (top 10%)
    "min_hc": ["0.0"],               # Minimum HC threshold
}
```

---

## ✅ Final Checklist

Before running `bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010`:

- [ ] Repository cloned and checked out to `lihu/hc_implementation` branch
- [ ] Conda environment created with Python 3.8
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data downloaded (datasets and corpus)
- [ ] Elasticsearch running on port 9200
- [ ] Corpus indexed to Elasticsearch
- [ ] LLM server running on port 8010 (GPT or FLAN-T5)
- [ ] Null distribution built for hotpotqa (`python -m commaqa.hc.build_null_dist hotpotqa dev_500`)
- [ ] HC modules verified (`python -c "from commaqa.hc import HigherCriticism"`)
- [ ] Config file exists (`ls base_configs/hc_qa_gpt_hotpotqa.jsonnet`)

If all checks pass, run:
```bash
bash run_retrieval_dev.sh hc_qa gpt hotpotqa 8010
```

---

## 📝 Implementation Notes

**What makes this integration clean:**
- ✅ Original HC code in `hc_copied/` untouched
- ✅ No new dependencies (uses existing numpy via torch/scipy)
- ✅ Drop-in replacement for ircot_qa
- ✅ Follows existing patterns (oner_qa structure)
- ✅ Compatible with all existing scripts

**Performance expectations:**
- HC should perform similarly or better than IRCoT (F1=0.604, EM=0.446)
- HC excels when score distribution clearly separates relevant/non-relevant
- HC adapts k per query (0-30) vs fixed k=15 in baselines

---

**Ready to deploy!** Follow steps 1-10, then run the final command.
