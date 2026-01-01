#!/bin/bash
# Script to build null distribution for CrossEntityQA on runpod
# This is a ONE-TIME operation that needs to be run before using hc_qa with CrossEntityQA

set -e

echo "=============================================================================="
echo "Building Null Distribution for CrossEntityQA"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Verify the retriever server is running"
echo "  2. Build a null distribution from CrossEntityQA queries"
echo "  3. Save the distribution to processed_data/hc_null_distributions/"
echo ""
echo "Prerequisites:"
echo "  - Retriever server must be running on port 8000 (or specified port)"
echo "  - CrossEntityQA corpus must be indexed in Elasticsearch"
echo "  - CrossEntityQA/queries_with_ground_truth.jsonl must exist"
echo ""
echo "=============================================================================="
echo ""

# Configuration
RETRIEVER_HOST="${RETRIEVER_HOST:-http://127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
NUM_QUERIES="${NUM_QUERIES:-100}"
RETRIEVAL_PER_QUERY="${RETRIEVAL_PER_QUERY:-100}"

echo "Configuration:"
echo "  Retriever Host: $RETRIEVER_HOST"
echo "  Retriever Port: $RETRIEVER_PORT"
echo "  Number of Queries: $NUM_QUERIES"
echo "  Retrieval per Query: $RETRIEVAL_PER_QUERY"
echo ""

# Check if retriever is responding
echo "Checking retriever server..."
if curl -s --max-time 5 "$RETRIEVER_HOST:$RETRIEVER_PORT/health" > /dev/null 2>&1; then
    echo "✓ Retriever server is responding"
else
    echo "✗ ERROR: Retriever server is not responding at $RETRIEVER_HOST:$RETRIEVER_PORT"
    echo "  Please start the retriever server first:"
    echo "  cd retriever_server && python retriever_server.py 8000"
    exit 1
fi
echo ""

# Check if dataset exists
echo "Checking dataset..."
if [ -f "CrossEntityQA/queries_with_ground_truth.jsonl" ]; then
    QUERY_COUNT=$(wc -l < CrossEntityQA/queries_with_ground_truth.jsonl)
    echo "✓ Found CrossEntityQA dataset with $QUERY_COUNT queries"
else
    echo "✗ ERROR: CrossEntityQA/queries_with_ground_truth.jsonl not found"
    echo "  Please ensure the dataset is in the correct location"
    exit 1
fi
echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p processed_data/hc_null_distributions
echo "✓ Directory ready"
echo ""

# Build null distribution
echo "=============================================================================="
echo "Building null distribution (this may take 10-20 minutes)..."
echo "=============================================================================="
echo ""

python -m commaqa.hc.build_null_dist \
    crossentityqa \
    dev \
    --num_queries $NUM_QUERIES \
    --retrieval_per_query $RETRIEVAL_PER_QUERY \
    --retriever_host $RETRIEVER_HOST \
    --retriever_port $RETRIEVER_PORT \
    --output_dir processed_data/hc_null_distributions

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✓ SUCCESS: Null distribution built successfully!"
    echo "=============================================================================="
    echo ""
    echo "Output saved to: processed_data/hc_null_distributions/crossentityqa_null_dist.pkl"
    echo ""
    echo "You can now run hc_qa inference with:"
    echo "  bash run_retrieval_dev.sh hc_qa gpt crossentityqa 8010"
    echo ""
else
    echo ""
    echo "=============================================================================="
    echo "✗ ERROR: Null distribution building failed"
    echo "=============================================================================="
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  - Retriever server is running and accessible"
    echo "  - CrossEntityQA corpus is properly indexed"
    echo "  - Network connectivity is working"
    exit 1
fi
