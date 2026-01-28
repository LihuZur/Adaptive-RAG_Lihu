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
    retriever_port: int = 9200,
    bm25_threshold: float = 0.1
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
        # Extract scores for our random docs, only if BM25 > threshold
        scores = []
        for doc_id in random_doc_ids:
            score = score_map.get(doc_id, 0.0)
            if score > bm25_threshold:
                scores.append(score)
        return np.array(scores)
    except Exception as e:
        logger.warning(f"Error getting BM25 scores: {e}")
        return np.array([])


def build_query_specific_null(
    corpus_name: str,
    split: str = "dev_500",
    num_queries: int = None,
    null_samples_per_query: int = 3000,
    retriever_host: str = "http://127.0.0.1",
    retriever_port: int = 9200,
    bm25_threshold: float = 0.1,
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
    
    # Only exclude direct ground-truth docs for each query
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

    # Precompute relevant doc ids for each query
    query_to_reldocs = {}
    for q in tqdm(queries, desc="Precomputing relevant doc sets"):
        qid = q.get('qid', q.get('query_id', q.get('_id', 'unknown')))
        rel_docs = extract_relevant_doc_ids(q)
        query_to_reldocs[qid] = rel_docs

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
        forbidden_doc_ids = query_to_reldocs.get(qid, set())

        # Try to get as many null samples as possible, fallback to smaller sample size if needed
        LARGE_POOL = max(2000, int(null_samples_per_query * 1.5))
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
                retriever_port=retriever_port,
                bm25_threshold=bm25_threshold
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
            candidate_pool_size = len([doc_id for doc_id in doc_ids if doc_id not in forbidden_doc_ids])
            logger.info(f"Query {qid}: candidate negative pool size after filtering: {candidate_pool_size}")
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
        # Use all sampled negatives above threshold (no sorting)
        n_take = min(null_samples_per_query, len(all_scores))
        used_scores = all_scores[:n_take]
        mu = float(np.mean(used_scores))
        sigma = float(np.std(used_scores))
        logger.info(f"Query {qid}: null min={np.min(used_scores):.3f}, max={np.max(used_scores):.3f}, mean={mu:.3f}, std={sigma:.3f}, n={len(used_scores)}")
        if sigma == 0:
            logger.warning(f"Query {qid}: Zero std deviation, skipping")
            skipped_queries.append(qid)
            continue
        query_null_stats[qid] = {
            "mu": mu,
            "sigma": sigma,
            "n_samples": len(used_scores),
            "min": float(np.min(used_scores)),
            "max": float(np.max(used_scores)),
        }
        if len(used_scores) < null_samples_per_query:
            low_sample_queries.append((qid, len(used_scores)))
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
    parser.add_argument('--null_samples_per_query', type=int, default=3000, 
                       help='Random docs to sample per query (default: 3000)')
    parser.add_argument('--retriever_host', type=str, default='http://127.0.0.1')
    parser.add_argument('--retriever_port', type=int, default=9200)
    parser.add_argument('--output_dir', type=str, default='processed_data/hc_null_distributions')
    parser.add_argument('--bm25_threshold', type=float, default=2.0, help='BM25 threshold for negatives (default 2.0, try higher for stronger negatives)')
    parser.add_argument('--auto_sweep', action='store_true', help='Try multiple BM25 thresholds and report best')
    parser.add_argument('--shared_pool', action='store_true', help='Use shared random pool for null and test (split 2N into N for null, N for test)')

    args = parser.parse_args()

    def run_and_report(thresh):
        if args.shared_pool:
            # Shared-pool: for each query, sample 2N, split into null and test, save both
            null_stats = {}
            test_scores_dict = {}
            queries = load_queries(args.corpus_name, args.num_queries or 1000000)
            for q in tqdm(queries, desc=f"Shared-pool nulls (BM25>{thresh})"):
                qid = q.get('qid', q.get('query_id', q.get('_id', 'unknown')))
                query_text = q.get('question', q.get('query_text', ''))
                scores = get_random_doc_scores(query_text, args.corpus_name, args.null_samples_per_query * 2, bm25_threshold=thresh)
                if len(scores) < args.null_samples_per_query * 2:
                    continue
                np.random.shuffle(scores)
                null_scores = scores[:args.null_samples_per_query]
                test_scores = scores[args.null_samples_per_query:args.null_samples_per_query*2]
                mu = float(np.mean(null_scores))
                sigma = float(np.std(null_scores))
                null_stats[qid] = {"mu": mu, "sigma": sigma, "n_samples": len(null_scores)}
                test_scores_dict[qid] = test_scores.tolist()
            # Save nulls
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{args.corpus_name}_sharedpool_null_bm25_{thresh}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(null_stats, f)
            # Save test scores for verification
            test_file = output_dir / f"{args.corpus_name}_sharedpool_test_bm25_{thresh}.pkl"
            with open(test_file, 'wb') as f:
                pickle.dump(test_scores_dict, f)
            all_mus = [stats['mu'] for stats in null_stats.values()]
            all_sigmas = [stats['sigma'] for stats in null_stats.values()]
            mean_mu = np.mean(all_mus)
            mean_sigma = np.mean(all_sigmas)
            ratio = mean_sigma / mean_mu if mean_mu != 0 else 0
            print(f"[SharedPool] BM25>{thresh}: mean μ={mean_mu:.3f}, mean σ={mean_sigma:.3f}, σ/μ={ratio:.3f}")
            return {'threshold': thresh, 'mean_mu': mean_mu, 'mean_sigma': mean_sigma, 'ratio': ratio, 'method': 'sharedpool'}
        else:
            query_null_stats = build_query_specific_null(
                corpus_name=args.corpus_name,
                split=args.split,
                num_queries=args.num_queries,
                null_samples_per_query=args.null_samples_per_query,
                retriever_host=args.retriever_host,
                retriever_port=args.retriever_port,
                bm25_threshold=thresh,
            )
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{args.corpus_name}_query_specific_null_bm25_{thresh}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(query_null_stats, f)
            json_file = output_dir / f"{args.corpus_name}_query_specific_null_bm25_{thresh}.json"
            with open(json_file, 'w') as f:
                json.dump(query_null_stats, f, indent=2)
            all_mus = [stats['mu'] for stats in query_null_stats.values()]
            all_sigmas = [stats['sigma'] for stats in query_null_stats.values()]
            mean_mu = np.mean(all_mus)
            mean_sigma = np.mean(all_sigmas)
            ratio = mean_sigma / mean_mu if mean_mu != 0 else 0
            print(f"BM25>{thresh}: mean μ={mean_mu:.3f}, mean σ={mean_sigma:.3f}, σ/μ={ratio:.3f}")
            return {'threshold': thresh, 'mean_mu': mean_mu, 'mean_sigma': mean_sigma, 'ratio': ratio, 'method': 'standard'}

    if args.auto_sweep:
        # Only run standard method, sweep over 1.5, 2.0, 2.5
        thresholds = [1.5, 2.0, 2.5]
        results = []
        print(f"\n=== Building nulls for method: standard ===")
        args.shared_pool = False
        for thresh in thresholds:
            print(f"\n--- BM25 threshold {thresh} ---")
            res = run_and_report(thresh)
            res['method'] = 'standard'
            results.append(res)
        print("\n=== Summary of BM25 threshold sweep (standard) ===")
        for r in results:
            print(f"standard | BM25>{r['threshold']}: mean μ={r['mean_mu']:.3f}, mean σ={r['mean_sigma']:.3f}, σ/μ={r['ratio']:.3f}")
        best = max(results, key=lambda r: r['ratio'])
        print(f"\nBest by σ/μ ratio: standard | BM25>{best['threshold']} (σ/μ={best['ratio']:.3f})")
        # Shared pool sweep is commented out due to issues
        # for method in ['sharedpool']:
        #     print(f"\n=== Building nulls for method: {method} ===")
        #     args.shared_pool = True
        #     for thresh in thresholds:
        #         print(f"\n--- BM25 threshold {thresh} ---")
        #         res = run_and_report(thresh)
        #         res['method'] = method
        #         results.append(res)
        # print("\n=== Summary of BM25 threshold/method sweep ===")
        # for r in results:
        #     print(f"{r['method']} | BM25>{r['threshold']}: mean μ={r['mean_mu']:.3f}, mean σ={r['mean_sigma']:.3f}, σ/μ={r['ratio']:.3f}")
        # best = max(results, key=lambda r: r['ratio'])
        # print(f"\nBest by σ/μ ratio: {best['method']} | BM25>{best['threshold']} (σ/μ={best['ratio']:.3f})")
    else:
        # Build query-specific null
        query_null_stats = build_query_specific_null(
            corpus_name=args.corpus_name,
            split=args.split,
            num_queries=args.num_queries,
            null_samples_per_query=args.null_samples_per_query,
            retriever_host=args.retriever_host,
            retriever_port=args.retriever_port,
            bm25_threshold=args.bm25_threshold,
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
