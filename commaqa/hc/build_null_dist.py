"""
Build null distribution for HC-based retrieval on a dataset.

This script retrieves documents for non-relevant queries to build
a null distribution of BM25 scores. The null distribution is then
used by the HC retrieval participant to determine optimal thresholds.

Usage:
    python -m commaqa.hc.build_null_dist hotpotqa dev_500
    python -m commaqa.hc.build_null_dist 2wikimultihopqa dev_500
    python -m commaqa.hc.build_null_dist musique dev_500
    python -m commaqa.hc.build_null_dist nq dev_500
    python -m commaqa.hc.build_null_dist trivia dev_500
    python -m commaqa.hc.build_null_dist squad dev_500
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(corpus_name: str, split: str = "dev_500") -> List[Dict[str, Any]]:
    """Load dataset from file."""
    dataset_path = Path(f"processed_data/{corpus_name}/{split}_subsampled.jsonl")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = []
    with open(dataset_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def retrieve_bm25_scores(
    query: str,
    corpus_name: str,
    retrieval_count: int = 100,
    retriever_host: str = "http://127.0.0.1",
    retriever_port: int = 8000,
) -> List[float]:
    """Retrieve BM25 scores for a query."""
    url = f"{retriever_host}:{retriever_port}/retrieve"
    
    params = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "query_text": query,
        "max_hits_count": retrieval_count,
        "corpus_name": corpus_name,
        "document_type": "title_paragraph_text",
    }
    
    try:
        response = requests.post(url, json=params, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        retrieval = result.get("retrieval", [])
        
        scores = []
        for item in retrieval:
            if item.get("corpus_name") == corpus_name:
                score = item.get("score", 0.0)
                scores.append(score)
        
        return scores
    
    except Exception as e:
        logger.warning(f"Retrieval failed for query '{query[:50]}...': {e}")
        return []


def build_null_distribution(
    corpus_name: str,
    split: str = "dev_500",
    num_queries: int = 100,
    retrieval_per_query: int = 100,
    retriever_host: str = "http://127.0.0.1",
    retriever_port: int = 8000,
) -> Dict[str, Any]:
    """
    Build null distribution by retrieving documents for random queries.
    
    Strategy: Use questions from the dataset but retrieve from the corpus.
    Since most retrieved documents will be non-relevant (datasets have 87-93% noise),
    this provides a good approximation of the null distribution.
    """
    logger.info(f"Building null distribution for {corpus_name}")
    
    # Load dataset
    data = load_dataset(corpus_name, split)
    
    # Sample queries
    if len(data) > num_queries:
        import random
        random.seed(42)
        sampled_data = random.sample(data, num_queries)
    else:
        sampled_data = data
        logger.warning(f"Dataset has only {len(data)} examples, using all")
    
    # Collect BM25 scores
    all_scores = []
    
    for i, item in enumerate(sampled_data):
        query = item.get("question_text", "") or item.get("question", "")
        
        print(f"[LOOP] Item {i+1}: question='{query[:50] if query else 'EMPTY'}'")
        
        if not query.strip():
            print(f"[SKIP] Empty query, skipping")
            continue
        
        logger.info(f"Processing query {i+1}/{len(sampled_data)}: {query[:50]}...")
        print(f"[CALLING] About to call retrieve_bm25_scores")
    for i, item in enumerate(sampled_data):
        query = item.get("question_text", "") or item.get("question", "")
        
        if not query.strip():
            continue
        
        logger.info(f"Processing query {i+1}/{len(sampled_data)}: {query[:50]}...")
        
        scores = retrieve_bm25_scores(en(all_scores)} scores so far")
    
    if not all_scores:
        raise ValueError("Failed to collect any BM25 scores")
    
    # Convert to numpy array
    scores_array = np.array(all_scores, dtype=np.float32)
    
    # Compute statistics
    null_dist = {
        "scores": scores_array,
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "count": len(scores_array),
        "corpus_name": corpus_name,
        "split": split,
    }
    
    logger.info(f"Null distribution statistics:")
    logger.info(f"  Count: {null_dist['count']}")
    logger.info(f"  Mean: {null_dist['mean']:.3f}")
    logger.info(f"  Std: {null_dist['std']:.3f}")
    logger.info(f"  Min: {null_dist['min']:.3f}")
    logger.info(f"  Max: {null_dist['max']:.3f}")
    
    return null_dist


def main():
    parser = argparse.ArgumentParser(
        description="Build null distribution for HC-based retrieval"
    )
    parser.add_argument(
        "corpus_name",
        type=str,
        choices=["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"],
        help="Corpus name",
    )
    parser.add_argument(
        "split",
        type=str,
        default="dev_500",
        help="Dataset split (default: dev_500)",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=100,
        help="Number of queries to sample (default: 100)",
    )
    parser.add_argument(
        "--retrieval_per_query",
        type=int,
        default=100,
        help="Number of documents to retrieve per query (default: 100)",
    )
    parser.add_argument(
        "--retriever_host",
        type=str,
        default="http://127.0.0.1",
        help="Retriever host (default: http://127.0.0.1)",
    )
    parser.add_argument(
        "--retriever_port",
        type=int,
        default=8000,
        help="Retriever port (default: 8000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data/hc_null_distributions",
        help="Output directory (default: processed_data/hc_null_distributions)",
    )
    
    args = parser.parse_args()
    
    # Build null distribution
    null_dist = build_null_distribution(
        corpus_name=args.corpus_name,
        split=args.split,
        num_queries=args.num_queries,
        retrieval_per_query=args.retrieval_per_query,
        retriever_host=args.retriever_host,
        retriever_port=args.retriever_port,
    )
    
    # Save to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.corpus_name}_null_dist.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(null_dist, f)
    
    logger.info(f"Saved null distribution to {output_path}")


if __name__ == "__main__":
    main()
