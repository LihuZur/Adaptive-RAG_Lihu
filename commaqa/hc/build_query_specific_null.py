"""
Build query-specific null distributions for HC-based retrieval.

Instead of one global null distribution, this builds per-query statistics (μ, σ)
by sampling random documents for each query. This ensures p-values are uniform
even when entities overlap across queries.

Usage:
    python -m commaqa.hc.build_query_specific_null crossentityqa dev_500 \
        --num_queries 500 \
        --null_samples_per_query 500
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_queries(corpus_name: str, split: str) -> List[Dict]:
    """Load queries from dataset."""
    if corpus_name == "crossentityqa":
        query_file = "CrossEntityQA/queries_with_ground_truth.jsonl"
    else:
        query_file = f"processed_data/{corpus_name}/{split}_subsampled.jsonl"
    
    queries = []
    with open(query_file) as f:
        for line in f:
            queries.append(json.loads(line))
    
    return queries


def get_random_doc_scores(
    query_text: str,
    corpus: str,
    n_docs: int,
    retriever_host: str = "http://127.0.0.1",
    retriever_port: int = 9200
) -> np.ndarray:
    """
    Get BM25 scores for random documents.
    
    Strategy:
    1. Get random doc IDs using random_score function_score
    2. Query these docs with actual BM25 to get scores
    """
    # Step 1: Get random doc IDs
    try:
        response = requests.post(
            f'{retriever_host}:{retriever_port}/{corpus}/_search',
            headers={'Content-Type': 'application/json'},
            json={
                'query': {
                    'function_score': {
                        'query': {'match_all': {}},
                        'random_score': {'seed': np.random.randint(1000000)},
                        'boost_mode': 'replace'
                    }
                },
                '_source': False,  # Don't need content
                'size': n_docs
            },
            timeout=30
        )
        
        if response.status_code != 200:
            logger.warning(f"Random docs fetch failed: {response.status_code}")
            return np.array([])
        
        random_doc_ids = [hit['_id'] for hit in response.json()['hits']['hits']]
        
    except Exception as e:
        logger.warning(f"Error getting random docs: {e}")
        return np.array([])
    
    # Step 2: Get BM25 scores for these docs
    try:
        response = requests.post(
            f'{retriever_host}:{retriever_port}/{corpus}/_search',
            headers={'Content-Type': 'application/json'},
            json={
                'query': {
                    'multi_match': {
                        'query': query_text,
                        'fields': ['title', 'paragraph_text']
                    }
                },
                '_source': False,
                'size': n_docs * 3  # Get more to ensure we cover our random docs
            },
            timeout=30
        )
        
        if response.status_code != 200:
            logger.warning(f"BM25 query failed: {response.status_code}")
            return np.array([])
        
        # Build score map
        score_map = {hit['_id']: hit['_score'] for hit in response.json()['hits']['hits']}
        
        # Extract scores for our random docs
        scores = []
        for doc_id in random_doc_ids:
            if doc_id in score_map:
                scores.append(score_map[doc_id])
        
        return np.array(scores)
        
    except Exception as e:
        logger.warning(f"Error getting BM25 scores: {e}")
        return np.array([])


def build_query_specific_null(
    corpus_name: str,
    split: str = "dev_500",
    num_queries: int = None,
    null_samples_per_query: int = 500,
    retriever_host: str = "http://127.0.0.1",
    retriever_port: int = 9200,
) -> Dict[str, Dict[str, float]]:
    """
    Build query-specific null distribution statistics.
    
    For each query:
    - Sample N random documents
    - Compute their BM25 scores
    - Store μ (mean) and σ (std) for that query
    
    Returns:
        Dict mapping query_id -> {"mu": float, "sigma": float, "n_samples": int}
    """
    logger.info(f"Building query-specific null for {corpus_name}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Null samples per query: {null_samples_per_query}")
    
    # Load queries
    queries = load_queries(corpus_name, split)
    if num_queries:
        queries = queries[:num_queries]
    
    logger.info(f"Loaded {len(queries)} queries")
    
    # Build null stats for each query
    query_null_stats = {}
    
    for query_data in tqdm(queries, desc="Building query-specific nulls"):
        qid = query_data.get('qid', query_data.get('_id', 'unknown'))
        query_text = query_data.get('question', query_data.get('query_text', ''))
        
        # Get random doc scores
        scores = get_random_doc_scores(
            query_text=query_text,
            corpus=corpus_name,
            n_docs=null_samples_per_query,
            retriever_host=retriever_host,
            retriever_port=retriever_port
        )
        
        if len(scores) < 10:
            logger.warning(f"Query {qid}: Only got {len(scores)} scores, skipping")
            continue
        
        # Calculate statistics
        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        
        if sigma == 0:
            logger.warning(f"Query {qid}: Zero std deviation, skipping")
            continue
        
        query_null_stats[qid] = {
            "mu": mu,
            "sigma": sigma,
            "n_samples": len(scores),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
        
        if len(query_null_stats) % 50 == 0:
            logger.info(f"Processed {len(query_null_stats)} queries")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Built query-specific null statistics for {len(query_null_stats)} queries")
    
    # Summary statistics
    all_mus = [stats['mu'] for stats in query_null_stats.values()]
    all_sigmas = [stats['sigma'] for stats in query_null_stats.values()]
    
    logger.info(f"\nQuery-specific μ statistics:")
    logger.info(f"  Mean: {np.mean(all_mus):.3f}")
    logger.info(f"  Std: {np.std(all_mus):.3f}")
    logger.info(f"  Range: [{np.min(all_mus):.3f}, {np.max(all_mus):.3f}]")
    
    logger.info(f"\nQuery-specific σ statistics:")
    logger.info(f"  Mean: {np.mean(all_sigmas):.3f}")
    logger.info(f"  Std: {np.std(all_sigmas):.3f}")
    logger.info(f"  Range: [{np.min(all_sigmas):.3f}, {np.max(all_sigmas):.3f}]")
    logger.info(f"{'='*80}\n")
    
    return query_null_stats


def main():
    parser = argparse.ArgumentParser(description='Build query-specific null distributions')
    parser.add_argument('corpus_name', type=str, help='Corpus name (e.g., crossentityqa)')
    parser.add_argument('split', type=str, default='dev_500', help='Dataset split')
    parser.add_argument('--num_queries', type=int, default=None, help='Number of queries (default: all)')
    parser.add_argument('--null_samples_per_query', type=int, default=500, 
                       help='Random docs to sample per query')
    parser.add_argument('--retriever_host', type=str, default='http://127.0.0.1')
    parser.add_argument('--retriever_port', type=int, default=9200)
    parser.add_argument('--output_dir', type=str, default='processed_data/hc_null_distributions')
    
    args = parser.parse_args()
    
    # Build query-specific null
    query_null_stats = build_query_specific_null(
        corpus_name=args.corpus_name,
        split=args.split,
        num_queries=args.num_queries,
        null_samples_per_query=args.null_samples_per_query,
        retriever_host=args.retriever_host,
        retriever_port=args.retriever_port,
    )
    
    # Save to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.corpus_name}_query_specific_null.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(query_null_stats, f)
    
    logger.info(f"Saved query-specific null to: {output_file}")
    
    # Also save as JSON for inspection
    json_file = output_dir / f"{args.corpus_name}_query_specific_null.json"
    with open(json_file, 'w') as f:
        json.dump(query_null_stats, f, indent=2)
    
    logger.info(f"Saved JSON version to: {json_file}")


if __name__ == "__main__":
    main()
