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
    if not isinstance(query_text, str) or not query_text.strip():
        logger.warning(f"Empty or invalid query_text: {query_text!r}, skipping BM25 scoring.")
        return np.array([])
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
                    'bool': {
                        'should': [
                            {'match': {'title': query_text}},
                            {'match': {'paragraph_text': query_text}}
                        ]
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
    
    # Step 1: Map each entity to all relevant doc IDs for all queries about that entity
    def extract_entities(q):
        return set(q.get('entity_coverage', []) or q.get('sub_cluster_entities', []))

    def extract_relevant_doc_ids(q):
        rel_keys = ["positive_ctxs", "positive_paragraphs", "relevant_docs", "answers", "answer_paragraphs"]
        doc_ids = set()
        for k in rel_keys:
            if k in q and isinstance(q[k], list):
                for item in q[k]:
                    if isinstance(item, dict) and ("_id" in item or "id" in item):
                        doc_ids.add(item.get("_id", item.get("id")))
                    elif isinstance(item, str):
                        doc_ids.add(item)
        if "gt_doc_ids" in q:
            doc_ids.update(q["gt_doc_ids"])
        return doc_ids

    # Build entity -> set(all relevant doc ids for that entity)
    # Also build entity -> set(all top-K retrieved doc ids for that entity)
    entity_to_reldocs = {}
    entity_to_topkdocs = {}
    TOP_K = 100  # You can adjust this value for how "deep" negatives should be
    for q in tqdm(queries, desc="Precomputing entity doc sets"):
        entities = extract_entities(q)
        rel_docs = extract_relevant_doc_ids(q)
        # Get top-K retrieved doc ids for this query
        query_text = q.get('question', q.get('query_text', ''))
        try:
            response = requests.post(
                f'{retriever_host}:{retriever_port}/{corpus_name}/_search',
                headers={'Content-Type': 'application/json'},
                json={
                    'query': {
                        'bool': {
                            'should': [
                                {'match': {'title': query_text}},
                                {'match': {'paragraph_text': query_text}}
                            ]
                        }
                    },
                    '_source': False,
                    'size': TOP_K
                },
                timeout=30
            )
            hits = response.json()['hits']['hits']
            topk_doc_ids = set(hit['_id'] for hit in hits)
        except Exception as e:
            logger.warning(f"Error retrieving top-K docs for entity set: {entities} : {e}")
            topk_doc_ids = set()
        for ent in entities:
            if ent not in entity_to_reldocs:
                entity_to_reldocs[ent] = set()
            entity_to_reldocs[ent].update(rel_docs)
            if ent not in entity_to_topkdocs:
                entity_to_topkdocs[ent] = set()
            entity_to_topkdocs[ent].update(topk_doc_ids)

    query_null_stats = {}
    skipped_queries = []
    low_sample_queries = []
    for query_data in tqdm(queries, desc="Building brute-force farthest nulls"):
        qid = query_data.get('qid', query_data.get('query_id', query_data.get('_id', 'unknown')))
        query_text = query_data.get('question', query_data.get('query_text', ''))
        if not isinstance(query_text, str) or not query_text.strip():
            logger.warning(f"Query {qid}: Empty or invalid query_text, skipping.")
            skipped_queries.append(qid)
            continue
        entities = extract_entities(query_data)
        forbidden_doc_ids = set()
        for ent in entities:
            forbidden_doc_ids.update(entity_to_reldocs.get(ent, set()))
            forbidden_doc_ids.update(entity_to_topkdocs.get(ent, set()))

        # Try to get as many null samples as possible, fallback to smaller sample size if needed
        LARGE_POOL = max(2000, null_samples_per_query * 10)
        MIN_NULL_SAMPLES = 3
        all_scores = []
        all_doc_ids = []
        n_attempts = 0
        while len(all_scores) < LARGE_POOL and n_attempts < 20:
            candidate_scores = get_random_doc_scores(
                query_text=query_text,
                corpus=corpus_name,
                n_docs=LARGE_POOL,
                retriever_host=retriever_host,
                retriever_port=retriever_port
            )
            try:
                response = requests.post(
                    f'{retriever_host}:{retriever_port}/{corpus_name}/_search',
                    headers={'Content-Type': 'application/json'},
                    json={
                        'query': {
                            'function_score': {
                                'query': {'match_all': {}},
                                'random_score': {'seed': np.random.randint(1000000)},
                                'boost_mode': 'replace'
                            }
                        },
                        '_source': False,
                        'size': LARGE_POOL
                    },
                    timeout=30
                )
                hits = response.json()['hits']['hits']
                doc_ids = [hit['_id'] for hit in hits]
            except Exception as e:
                logger.warning(f"Error getting random doc ids for {qid}: {e}")
                break
            for idx, doc_id in enumerate(doc_ids):
                if doc_id not in forbidden_doc_ids and idx < len(candidate_scores):
                    all_scores.append(candidate_scores[idx])
                    all_doc_ids.append(doc_id)
                if len(all_scores) >= LARGE_POOL:
                    break
            n_attempts += 1
        # Fallback: if not enough, try with smaller sample size
        if len(all_scores) < MIN_NULL_SAMPLES:
            logger.warning(f"Query {qid}: Only got {len(all_scores)} null scores, skipping")
            skipped_queries.append(qid)
            continue
        # Take the bottom-N (lowest BM25) scores as the null, but if not enough, use all
        sorted_idx = np.argsort(all_scores)
        n_take = min(null_samples_per_query, len(all_scores))
        farthest_scores = [all_scores[i] for i in sorted_idx[:n_take]]
        mu = float(np.mean(farthest_scores))
        sigma = float(np.std(farthest_scores))
        logger.info(f"Query {qid}: null min={np.min(farthest_scores):.3f}, max={np.max(farthest_scores):.3f}, mean={mu:.3f}, std={sigma:.3f}, n={len(farthest_scores)}")
        if sigma == 0:
            logger.warning(f"Query {qid}: Zero std deviation, skipping")
            skipped_queries.append(qid)
            continue
        query_null_stats[qid] = {
            "mu": mu,
            "sigma": sigma,
            "n_samples": len(farthest_scores),
            "min": float(np.min(farthest_scores)),
            "max": float(np.max(farthest_scores)),
        }
        if len(farthest_scores) < null_samples_per_query:
            low_sample_queries.append((qid, len(farthest_scores)))
        if len(query_null_stats) % 50 == 0:
            logger.info(f"Processed {len(query_null_stats)} queries")

    logger.info(f"\n{'='*80}")
    logger.info(f"Built query-specific null statistics for {len(query_null_stats)} queries")
    if skipped_queries:
        logger.warning(f"Skipped {len(skipped_queries)} queries due to too few/null samples: {skipped_queries}")
    if low_sample_queries:
        logger.warning(f"Queries with low null samples (< requested): {low_sample_queries}")
    
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
