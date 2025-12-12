#!/bin/bash

# Adaptive-RAG Services Startup Script
# This script starts all required services for running Adaptive-RAG

echo "=========================================="
echo "Starting Adaptive-RAG Services"
echo "=========================================="

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Please create one with your OPENAI_API_KEY"
fi

export NLTK_DATA=/workspace/nltk_data

# Activate conda environment
echo ""
echo "Activating conda environment..."
. /workspace/miniconda/etc/profile.d/conda.sh
conda activate adaptiverag

# -----------------------------
# Elasticsearch
# -----------------------------
echo ""
echo "Checking Elasticsearch status..."
if curl -s http://localhost:9200/ > /dev/null; then
    echo "✓ Elasticsearch is already running on port 9200"
else
    echo "Starting Elasticsearch on port 9200..."
    cd elasticsearch-7.10.2/
    ./bin/elasticsearch > ../elasticsearch.log 2>&1 &
    cd ..
    echo "Waiting for Elasticsearch to start (this may take 20-30 seconds)..."

    # Wait up to 60 seconds for Elasticsearch to be ready
    for i in {1..12}; do
        if curl -s http://localhost:9200/ > /dev/null 2>&1; then
            echo "✓ Elasticsearch started successfully"
            break
        fi
        if [ $i -eq 12 ]; then
            echo "⚠ Elasticsearch took longer than expected, but may still be starting..."
            echo "Check: curl localhost:9200/ to verify it's running"
        fi
        sleep 5
    done
fi

# -----------------------------
# Retriever Server
# -----------------------------
echo ""
echo "Checking Retriever Server status..."
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✓ Retriever Server is already running on port 8000"
else
    echo "Starting Retriever Server on port 8000..."
    uvicorn serve:app --port 8000 --app-dir retriever_server > retriever_server.log 2>&1 &
    sleep 5

    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✓ Retriever Server started successfully"
    else
        echo "✗ Failed to start Retriever Server. Check retriever_server.log"
        exit 1
    fi
fi

# -----------------------------
# LLM Server
# -----------------------------
echo ""
echo "Checking LLM Server status..."
if curl -s http://localhost:8010/docs > /dev/null 2>&1; then
    echo "✓ LLM Server is already running on port 8010"
else
    echo "Starting LLM Server on port 8010..."
    MODEL_NAME=flan-t5-xl uvicorn serve:app --port 8010 --app-dir llm_server > llm_server.log 2>&1 &
    sleep 5

    if curl -s http://localhost:8010/docs > /dev/null 2>&1; then
        echo "✓ LLM Server started successfully"
    else
        echo "✗ Failed to start LLM Server. Check llm_server.log"
        exit 1
    fi
fi

# -----------------------------
# OpenAI API Key
# -----------------------------
echo ""
echo "Checking OpenAI API Key..."
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-api-key-here" ]; then
    echo "⚠ Warning: OPENAI_API_KEY not set or using placeholder"
    echo "Please update .env file with your actual OpenAI API key"
else
    echo "✓ OpenAI API Key is configured"
fi

# -----------------------------
# Services Summary
# -----------------------------
echo ""
echo "=========================================="
echo "Services Status Summary"
echo "=========================================="
echo "Elasticsearch: http://localhost:9200"
echo "Retriever API: http://localhost:8000/docs"
echo "LLM Server API: http://localhost:8010/docs"
echo ""
echo "Ready to run retrieval strategies!"
echo "=========================================="

