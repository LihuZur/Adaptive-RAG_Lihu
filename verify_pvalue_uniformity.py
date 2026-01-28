"""
Verify p-value uniformity for null distribution.

This script tests whether p-values are uniform under H0 (null hypothesis).
For a correct null distribution, p-values should form a flat histogram from 0 to 1.

Usage:
    python verify_pvalue_uniformity.py --dataset crossentityqa --method global
    python verify_pvalue_uniformity.py --dataset crossentityqa --method query_specific
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import json
from pathlib import Path
from scipy import stats
from typing import List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from commaqa.hc.higher_criticism import HigherCriticism
from commaqa.hc.null_distribution import NullDistribution


def load_queries(dataset: str, n_queries: int = 50) -> List[dict]:
    """Load queries from dataset."""
    if dataset == "crossentityqa":
        query_file = "CrossEntityQA/queries_with_ground_truth.jsonl"
    else:
        query_file = f"processed_data/{dataset}/dev_500_subsampled.jsonl"
    
    queries = []
    with open(query_file) as f:
        for i, line in enumerate(f):
            if i >= n_queries:
                break
            queries.append(json.loads(line))
    
    return queries


def retrieve_random_docs(query_text: str, n_docs: int, corpus: str) -> np.ndarray:
    """Retrieve random documents and get their BM25 scores, filtering by BM25 threshold."""
    # Use global variable for threshold if set, else default
    bm25_threshold = globals().get('BM25_THRESHOLD', 0.1)
    LARGE_POOL = max(2000, n_docs * 10)
    response = requests.post(
        f'http://127.0.0.1:9200/{corpus}/_search',
        headers={'Content-Type': 'application/json'},
        json={
            'query': {
                'function_score': {
                    'query': {'match_all': {}},
                    'random_score': {'seed': np.random.randint(1000000)},
                    'boost_mode': 'replace'
                }
            },
            'size': LARGE_POOL
        }
    )
    if response.status_code != 200:
        raise Exception(f"ES query failed: {response.status_code}")
    doc_ids = [hit['_id'] for hit in response.json()['hits']['hits']]
    response2 = requests.post(
        f'http://127.0.0.1:9200/{corpus}/_search',
        headers={'Content-Type': 'application/json'},
        json={
            'query': {
                'multi_match': {
                    'query': query_text,
                    'fields': ['title', 'paragraph_text']
                }
            },
            'size': LARGE_POOL * 2
        }
    )
    score_map = {hit['_id']: hit['_score'] for hit in response2.json()['hits']['hits']}
    # Only include docs with BM25 > threshold
    scores = []
    for doc_id in doc_ids:
        score = score_map.get(doc_id, 0.0)
        if score > bm25_threshold:
            scores.append(score)
    # Sort by BM25 score (lowest first, i.e., farthest negatives)
    if scores:
        sorted_scores = sorted(scores)
        return np.array(sorted_scores[:n_docs])
    else:
        return np.array([])


def test_global_null(dataset: str, n_queries: int = 50) -> Tuple[np.ndarray, dict]:
    """Test current global null distribution approach."""
    print(f"\n{'='*80}")
    print(f"Testing GLOBAL null distribution for {dataset}")
    print(f"{'='*80}\n")
    
    # Load global null distribution
    null_dist_path = f"processed_data/hc_null_distributions/{dataset}_null_dist.pkl"
    if not Path(null_dist_path).exists():
        print(f"ERROR: Null distribution file not found: {null_dist_path}")
        print("This file exists on runpod. Either:")
        print("  1. Run this script on runpod")
        print("  2. Run with --method query_specific to test locally")
        return np.array([]), {'error': 'file_not_found'}
    
    with open(null_dist_path, 'rb') as f:
        null_dist = pickle.load(f)
    
    print(f"Loaded global null dist:")
    print(f"  n_samples: {null_dist.n_samples}")
    print(f"  mean: {null_dist.mean:.3f}")
    print(f"  std: {null_dist.std:.3f}")
    print(f"  range: [{null_dist.min_val:.3f}, {null_dist.max_val:.3f}]")
    
    # Create HC instance
    hc = HigherCriticism(null_distribution=null_dist)
    
    # Load queries
    queries = load_queries(dataset, n_queries)
    print(f"\nLoaded {len(queries)} queries")
    
    # For each query, get random docs and compute p-values
    all_pvalues = []
    
    for i, query_data in enumerate(queries):
        query_text = query_data.get('question', query_data.get('query_text', ''))
        print(f"Query {i+1}/{len(queries)}: {query_text[:50]}...")
        
        # Get 100 random docs
        scores = retrieve_random_docs(query_text, n_docs=100, corpus=dataset)
        
        if len(scores) == 0:
            print("  No scores retrieved, skipping")
            continue
        
        # Compute p-values using global null
        p_values = hc.compute_p_values(scores)
        all_pvalues.extend(p_values)
        
        print(f"  Got {len(scores)} scores, p-values range: [{p_values.min():.3f}, {p_values.max():.3f}]")
    
    all_pvalues = np.array(all_pvalues)
    
    # Statistical tests
    stats_results = {
        'n_samples': len(all_pvalues),
        'mean': float(np.mean(all_pvalues)),
        'std': float(np.std(all_pvalues)),
        'ks_statistic': None,
        'ks_pvalue': None,
    }
    
    # Kolmogorov-Smirnov test for uniformity
    ks_stat, ks_pval = stats.kstest(all_pvalues, 'uniform')
    stats_results['ks_statistic'] = float(ks_stat)
    stats_results['ks_pvalue'] = float(ks_pval)
    
    print(f"\n{'='*80}")
    print("Statistical Tests:")
    print(f"  Mean: {stats_results['mean']:.3f} (should be ~0.5 for uniform)")
    print(f"  Std: {stats_results['std']:.3f} (should be ~0.289 for uniform)")
    print(f"  KS statistic: {stats_results['ks_statistic']:.4f} (smaller = more uniform)")
    print(f"  KS p-value: {stats_results['ks_pvalue']:.4f} (>0.05 = can't reject uniformity)")
    
    if ks_pval < 0.05:
        print(f"\n  ❌ REJECTED: p-values are NOT uniform (p={ks_pval:.4f} < 0.05)")
    else:
        print(f"\n  ✓ ACCEPTED: p-values appear uniform (p={ks_pval:.4f} >= 0.05)")
    
    print(f"{'='*80}\n")
    
    return all_pvalues, stats_results


def test_query_specific_null(dataset: str, n_queries: int = 50, n_null_samples: int = 500) -> Tuple[np.ndarray, dict]:
    """Test query-specific null distribution approach."""
    print(f"\n{'='*80}")
    print(f"Testing QUERY-SPECIFIC null distribution for {dataset}")
    print(f"{'='*80}\n")
    
    # Load queries
    queries = load_queries(dataset, n_queries)
    print(f"Loaded {len(queries)} queries")
    # Load precomputed per-query nulls
    null_path = f"processed_data/hc_null_distributions/{dataset}_query_specific_null.pkl"
    with open(null_path, 'rb') as f:
        query_null_stats = pickle.load(f)
    print(f"Loaded precomputed query-specific nulls from {null_path}\n")

    all_pvalues = []
    low_test_doc_queries = []

    for i, query_data in enumerate(queries):
        query_text = query_data.get('question', query_data.get('query_text', ''))
        qid = query_data.get('qid', query_data.get('query_id', query_data.get('_id', 'unknown')))
        print(f"Query {i+1}/{len(queries)}: {query_text[:50]}...")

        # Get as many test docs as possible (up to n_null_samples)
        test_scores = retrieve_random_docs(query_text, n_docs=n_null_samples, corpus=dataset)
        print(f"  Got {len(test_scores)} test docs for this query.")
        if len(test_scores) == 0:
            print("  No test docs, skipping.")
            low_test_doc_queries.append((qid, 0))
            continue
        if len(test_scores) < n_null_samples:
            print(f"  Warning: Only {len(test_scores)} test docs (requested {n_null_samples})")
            low_test_doc_queries.append((qid, len(test_scores)))

        # Use precomputed μ, σ for this query
        null_stats = query_null_stats.get(qid)
        if null_stats is None:
            print(f"  No precomputed null for qid={qid}, skipping")
            continue
        mu_q = null_stats['mu']
        sigma_q = null_stats['sigma']
        if sigma_q == 0:
            print("  Zero std in precomputed null, skipping")
            continue

        # Debug: print null stats and test scores for first few queries
        if i < 5:
            print(f"    [DEBUG] Null μ={mu_q:.4f}, σ={sigma_q:.4f}")
            print(f"    [DEBUG] Test BM25 scores (first 10): {test_scores[:10]}")
            print(f"    [DEBUG] Test BM25 min={np.min(test_scores):.4f}, max={np.max(test_scores):.4f}, mean={np.mean(test_scores):.4f}")

        # Z-score normalize the test scores using precomputed null
        z_scores = (test_scores - mu_q) / sigma_q
        p_values = 1 - stats.norm.cdf(z_scores)
        all_pvalues.extend(p_values)

        print(f"  μ={mu_q:.2f}, σ={sigma_q:.2f}, p-values range: [{p_values.min():.3f}, {p_values.max():.3f}]")

    if low_test_doc_queries:
        print(f"\nSummary: Queries with low test doc counts (qid, count): {low_test_doc_queries}")
    
    all_pvalues = np.array(all_pvalues)
    
    # Statistical tests
    stats_results = {
        'n_samples': len(all_pvalues),
        'mean': float(np.mean(all_pvalues)),
        'std': float(np.std(all_pvalues)),
        'ks_statistic': None,
        'ks_pvalue': None,
    }
    
    # Kolmogorov-Smirnov test for uniformity
    ks_stat, ks_pval = stats.kstest(all_pvalues, 'uniform')
    stats_results['ks_statistic'] = float(ks_stat)
    stats_results['ks_pvalue'] = float(ks_pval)
    
    print(f"\n{'='*80}")
    print("Statistical Tests:")
    print(f"  Mean: {stats_results['mean']:.3f} (should be ~0.5 for uniform)")
    print(f"  Std: {stats_results['std']:.3f} (should be ~0.289 for uniform)")
    print(f"  KS statistic: {stats_results['ks_statistic']:.4f} (smaller = more uniform)")
    print(f"  KS p-value: {stats_results['ks_pvalue']:.4f} (>0.05 = can't reject uniformity)")
    
    if ks_pval < 0.05:
        print(f"\n  ❌ REJECTED: p-values are NOT uniform (p={ks_pval:.4f} < 0.05)")
    else:
        print(f"\n  ✓ ACCEPTED: p-values appear uniform (p={ks_pval:.4f} >= 0.05)")
    
    print(f"{'='*80}\n")
    
    return all_pvalues, stats_results


def plot_results(pvalues_dict: dict, output_path: str = "pvalue_uniformity_test.png"):
    """Plot p-value histograms and Q-Q plots."""
    fig, axes = plt.subplots(2, len(pvalues_dict), figsize=(6*len(pvalues_dict), 10))
    
    if len(pvalues_dict) == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (method, pvalues) in enumerate(pvalues_dict.items()):
        # Histogram
        ax = axes[0, col]
        ax.hist(pvalues, bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='r', linestyle='--', label='Uniform (y=1.0)')
        ax.set_xlabel('P-value')
        ax.set_ylabel('Density')
        ax.set_title(f'{method}\nHistogram (n={len(pvalues)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[1, col]
        stats.probplot(pvalues, dist='uniform', plot=ax)
        ax.set_title(f'{method}\nQ-Q Plot vs Uniform')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")


def main():

    parser = argparse.ArgumentParser(description='Verify p-value uniformity')
    parser.add_argument('--dataset', type=str, default='crossentityqa', help='Dataset name')
    parser.add_argument('--method', type=str, choices=['global', 'query_specific', 'both'], default='both')
    parser.add_argument('--n_queries', type=int, default=50, help='Number of queries to test')
    parser.add_argument('--n_null_samples', type=int, default=500, help='Null samples per query (query_specific only)')
    parser.add_argument('--bm25_threshold', type=float, default=2.0, help='BM25 threshold for negatives (default 2.0, try higher for stronger negatives)')
    parser.add_argument('--auto_sweep', action='store_true', help='Try multiple BM25 thresholds and report best')

    args = parser.parse_args()

    def run_and_report(bm25_threshold):
        global BM25_THRESHOLD
        BM25_THRESHOLD = bm25_threshold
        pvalues_dict = {}
        if args.method in ['query_specific', 'both']:
            pvalues, stats_res = test_query_specific_null(args.dataset, args.n_queries, args.n_null_samples)
            pvalues_dict[f'BM25>{bm25_threshold}'] = pvalues
        # Plot
        plot_results(pvalues_dict, output_path=f'pvalue_uniformity_test_bm25_{bm25_threshold}.png')
        print(f"\n[BM25>{bm25_threshold}] Mean: {np.mean(pvalues):.3f}, Std: {np.std(pvalues):.3f}, KS p-value: {stats_res['ks_pvalue']:.4f}")
        return stats_res['ks_pvalue'], np.mean(pvalues), np.std(pvalues)

    if args.auto_sweep:
        thresholds = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        results = []
        for thresh in thresholds:
            print(f"\n=== Running for BM25 threshold {thresh} ===")
            ks_pval, mean, std = run_and_report(thresh)
            results.append({'threshold': thresh, 'ks_pval': ks_pval, 'mean': mean, 'std': std})
        print("\n=== Summary of BM25 threshold sweep ===")
        for r in results:
            print(f"BM25>{r['threshold']}: KS p-value={r['ks_pval']:.4f}, mean={r['mean']:.3f}, std={r['std']:.3f}")
        best = max(results, key=lambda r: r['ks_pval'])
        print(f"\nBest threshold by KS p-value: BM25>{best['threshold']} (KS p-value={best['ks_pval']:.4f})")
    else:
        # Set global threshold for use in retrieve_random_docs
        global BM25_THRESHOLD
        BM25_THRESHOLD = args.bm25_threshold
        pvalues_dict = {}
        if args.method in ['query_specific', 'both']:
            pvalues, stats_res = test_query_specific_null(args.dataset, args.n_queries, args.n_null_samples)
            pvalues_dict['Query-Specific Null'] = pvalues
        # Plot
        plot_results(pvalues_dict)


if __name__ == "__main__":
    main()
