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
    """Retrieve random documents and get their BM25 scores."""
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
            'size': n_docs
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"ES query failed: {response.status_code}")
    
    # Now get BM25 scores for these docs with actual query
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
            'size': n_docs * 2  # Get more to ensure we have our random docs
        }
    )
    
    # Extract scores for our random docs
    scores = []
    score_map = {hit['_id']: hit['_score'] for hit in response2.json()['hits']['hits']}
    for doc_id in doc_ids:
        if doc_id in score_map:
            scores.append(score_map[doc_id])
    
    return np.array(scores) if scores else np.array([])


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

    for i, query_data in enumerate(queries):
        query_text = query_data.get('question', query_data.get('query_text', ''))
        qid = query_data.get('qid', query_data.get('query_id', query_data.get('_id', 'unknown')))
        print(f"Query {i+1}/{len(queries)}: {query_text[:50]}...")

        # Get N random docs to test null
        test_scores = retrieve_random_docs(query_text, n_docs=n_null_samples, corpus=dataset)

        if len(test_scores) < 10:
            print("  Too few test scores, skipping")
            continue

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

        # Z-score normalize the test scores using precomputed null
        z_scores = (test_scores - mu_q) / sigma_q
        p_values = 1 - stats.norm.cdf(z_scores)
        all_pvalues.extend(p_values)

        print(f"  μ={mu_q:.2f}, σ={sigma_q:.2f}, p-values range: [{p_values.min():.3f}, {p_values.max():.3f}]")
    
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
    
    args = parser.parse_args()
    
    pvalues_dict = {}
    
    if args.method in ['global', 'both']:
        pvalues, stats_res = test_global_null(args.dataset, args.n_queries)
        if len(pvalues) > 0:
            pvalues_dict['Global Null'] = pvalues
    
    if args.method in ['query_specific', 'both']:
        pvalues, stats_res = test_query_specific_null(args.dataset, args.n_queries, args.n_null_samples)
        pvalues_dict['Query-Specific Null'] = pvalues
    
    # Plot
    plot_results(pvalues_dict)


if __name__ == "__main__":
    main()
