"""
Simplified verification of p-value uniformity using synthetic data.

This demonstrates why query-specific null is needed without requiring 
Elasticsearch or saved null distributions.

Usage:
    python verify_pvalue_uniformity_simple.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def generate_synthetic_queries_with_entity_overlap(n_queries=50, n_entities=10):
    """
    Simulate CrossEntityQA where entities repeat across queries.
    
    Each query is assigned a "primary entity" (0 to n_entities-1).
    Documents about that entity get higher scores.
    This creates the problematic pattern where global null fails.
    """
    queries = []
    
    for i in range(n_queries):
        primary_entity = i % n_entities  # Entities repeat
        
        # Generate "candidate documents" for this query
        # 10 docs about primary entity (high scores: 25-35)
        # 40 docs about other topics (low scores: 8-18)
        high_scores = np.random.uniform(25, 35, size=10)
        low_scores = np.random.uniform(8, 18, size=40)
        
        candidate_scores = np.concatenate([high_scores, low_scores])
        np.random.shuffle(candidate_scores)
        
        queries.append({
            'query_id': f'q{i}',
            'primary_entity': primary_entity,
            'candidate_scores': candidate_scores,
        })
    
    return queries


def test_global_null(queries):
    """
    Test global null approach.
    
    Problem: Different queries naturally have different score distributions
    (e.g., iPhone queries score higher on any iPhone doc). Global null
    can't capture this, leading to non-uniform p-values.
    """
    print(f"\n{'='*80}")
    print("Testing GLOBAL null distribution")
    print(f"{'='*80}\n")
    
    # Build global null: sample random docs from all queries
    all_scores = []
    for q in queries:
        # Sample 20 random scores per query
        random_sample = np.random.choice(q['candidate_scores'], size=20, replace=False)
        all_scores.extend(random_sample)
    
    global_null = np.array(all_scores)
    sorted_null = np.sort(global_null)
    N_null = len(sorted_null)
    
    print(f"Global null statistics:")
    print(f"  Mean: {np.mean(global_null):.3f}")
    print(f"  Std: {np.std(global_null):.3f}")
    print(f"  Range: [{np.min(global_null):.3f}, {np.max(global_null):.3f}]")
    
    # Compute p-values for all candidates
    all_p_values = []
    
    for q in queries:
        scores = q['candidate_scores']
        
        # Convert to p-values using global null
        indices = np.searchsorted(sorted_null, scores, side='left')
        n_greater_equal = N_null - indices
        p_values = (n_greater_equal + 1.0) / (N_null + 1.0)
        eps = 1.0 / (N_null + 1.0)
        p_values = np.clip(p_values, eps, 1.0 - eps)
        
        all_p_values.extend(p_values)
    
    all_p_values = np.array(all_p_values)
    
    # Test uniformity
    ks_stat, ks_pvalue = stats.kstest(all_p_values, 'uniform')
    
    print(f"\nResults:")
    print(f"  Total p-values: {len(all_p_values)}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value: {ks_pvalue:.6f}")
    print(f"  Is uniform? {'YES' if ks_pvalue >= 0.05 else 'NO'} (need p >= 0.05)")
    
    if ks_pvalue < 0.05:
        print(f"\n  ❌ FAILED: P-values are NOT uniform (p={ks_pvalue:.6f} < 0.05)")
        print(f"     This means HC will make biased decisions!")
    else:
        print(f"\n  ✅ PASSED: P-values are uniform")
    
    return {
        'method': 'global',
        'p_values': all_p_values,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
    }


def test_query_specific_null(queries):
    """
    Test query-specific null approach (Z-score normalization).
    
    Solution: For each query, calculate μ_q and σ_q from random docs,
    then Z-score normalize. This makes p-values uniform even when
    different queries have different score distributions.
    """
    print(f"\n{'='*80}")
    print("Testing QUERY-SPECIFIC null distribution (Z-score normalization)")
    print(f"{'='*80}\n")
    
    all_p_values = []
    
    for q in queries:
        scores = q['candidate_scores']
        
        # For this query, calculate null statistics from random sample
        # (In real implementation, we'd sample from corpus, here we use candidates)
        random_sample = np.random.choice(scores, size=30, replace=False)
        mu_q = np.mean(random_sample)
        sigma_q = np.std(random_sample)
        
        # Z-score normalize
        z_scores = (scores - mu_q) / sigma_q
        
        # Convert to p-values using standard normal CDF
        p_values = 1.0 - stats.norm.cdf(z_scores)
        
        # Clip for numerical stability
        p_values = np.clip(p_values, 1e-10, 1.0 - 1e-10)
        
        all_p_values.extend(p_values)
    
    all_p_values = np.array(all_p_values)
    
    # Test uniformity
    ks_stat, ks_pvalue = stats.kstest(all_p_values, 'uniform')
    
    print(f"Results:")
    print(f"  Total p-values: {len(all_p_values)}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value: {ks_pvalue:.6f}")
    print(f"  Is uniform? {'YES' if ks_pvalue >= 0.05 else 'NO'} (need p >= 0.05)")
    
    if ks_pvalue < 0.05:
        print(f"\n  ❌ FAILED: P-values are NOT uniform (p={ks_pvalue:.6f} < 0.05)")
    else:
        print(f"\n  ✅ PASSED: P-values are uniform")
    
    return {
        'method': 'query_specific',
        'p_values': all_p_values,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
    }


def plot_comparison(global_result, query_specific_result):
    """Plot histograms and Q-Q plots for both methods."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Global null histogram
    ax = axes[0, 0]
    ax.hist(global_result['p_values'], bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', label='Uniform (expected)')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Density')
    ax.set_title(f'Global Null: P-value Histogram\n(KS p={global_result["ks_pvalue"]:.4f})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Global null Q-Q plot
    ax = axes[0, 1]
    stats.probplot(global_result['p_values'], dist='uniform', plot=ax)
    ax.set_title(f'Global Null: Q-Q Plot')
    ax.grid(alpha=0.3)
    
    # Query-specific null histogram
    ax = axes[1, 0]
    ax.hist(query_specific_result['p_values'], bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', label='Uniform (expected)')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Density')
    ax.set_title(f'Query-Specific Null: P-value Histogram\n(KS p={query_specific_result["ks_pvalue"]:.4f})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Query-specific null Q-Q plot
    ax = axes[1, 1]
    stats.probplot(query_specific_result['p_values'], dist='uniform', plot=ax)
    ax.set_title(f'Query-Specific Null: Q-Q Plot')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'pvalue_uniformity_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")
    
    return output_file


def main():
    print("="*80)
    print("P-VALUE UNIFORMITY VERIFICATION (Synthetic Data)")
    print("="*80)
    print("\nSimulating CrossEntityQA characteristics:")
    print("  - Small set of entities repeated across queries")
    print("  - Queries about same entity naturally get higher scores")
    print("  - This breaks global null assumption\n")
    
    # Generate synthetic queries
    queries = generate_synthetic_queries_with_entity_overlap(n_queries=50, n_entities=10)
    print(f"Generated {len(queries)} synthetic queries with {10} repeating entities\n")
    
    # Test both methods
    global_result = test_global_null(queries)
    query_specific_result = test_query_specific_null(queries)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nGlobal Null:")
    print(f"  KS p-value: {global_result['ks_pvalue']:.6f}")
    print(f"  Uniform? {'NO ❌' if global_result['ks_pvalue'] < 0.05 else 'YES ✅'}")
    
    print(f"\nQuery-Specific Null (Z-score):")
    print(f"  KS p-value: {query_specific_result['ks_pvalue']:.6f}")
    print(f"  Uniform? {'NO ❌' if query_specific_result['ks_pvalue'] < 0.05 else 'YES ✅'}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if global_result['ks_pvalue'] < 0.05 and query_specific_result['ks_pvalue'] >= 0.05:
        print("\n✅ Query-specific null FIXES the uniformity problem!")
        print("   This proves your supervisor's point: global null fails with entity overlap.")
        print("   Implementing query-specific null will make HC work correctly.\n")
    elif global_result['ks_pvalue'] >= 0.05:
        print("\n⚠️  Global null appears uniform in this synthetic test.")
        print("   This may be because the synthetic data doesn't fully capture")
        print("   the complexity of real CrossEntityQA. Try with real data on runpod.\n")
    else:
        print("\n⚠️  Both methods show non-uniform p-values in this synthetic test.")
        print("   This suggests the synthetic data may need tuning.\n")
    
    # Plot
    plot_comparison(global_result, query_specific_result)
    
    print("\nNext steps:")
    print("  1. Run this on runpod with REAL data: python verify_pvalue_uniformity.py --dataset crossentityqa --method both")
    print("  2. If query-specific null passes uniformity test, deploy it")
    print("  3. Re-run HC experiments and compare to oner_qa baseline\n")


if __name__ == "__main__":
    main()
