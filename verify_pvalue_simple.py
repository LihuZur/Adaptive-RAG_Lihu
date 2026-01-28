"""
Simple verification that query-specific null produces uniform p-values.
Uses synthetic data, no Elasticsearch needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("\n" + "="*80)
print("Demonstrating: Global vs Query-Specific Null Distribution")
print("="*80 + "\n")

# Simulate 3 queries with different "entity bias"
queries = [
    {"name": "Query A (high entity score)", "entity_bias": 25.0, "noise_std": 3.0},
    {"name": "Query B (medium entity score)", "entity_bias": 15.0, "noise_std": 3.0},
    {"name": "Query C (low entity score)", "entity_bias": 8.0, "noise_std": 3.0},
]

# Global null: mean=15, std=5 (doesn't fit any query well)
global_null_mean = 15.0
global_null_std = 5.0

print("Setup:")
print(f"  Global null: mean={global_null_mean}, std={global_null_std}")
print(f"  3 queries with different entity biases\n")

# Collect p-values
global_pvalues = []
query_specific_pvalues = []

for query in queries:
    print(f"{query['name']}:")
    print(f"  True null for this query: mean={query['entity_bias']}, std={query['noise_std']}")
    
    # Generate 100 random docs (pure noise for this query)
    noise_scores = np.random.normal(query['entity_bias'], query['noise_std'], 100)
    
    # Method 1: Global null (WRONG)
    z_scores_global = (noise_scores - global_null_mean) / global_null_std
    p_vals_global = 1 - stats.norm.cdf(z_scores_global)
    global_pvalues.extend(p_vals_global)
    
    print(f"  Global method: p-values range [{p_vals_global.min():.3f}, {p_vals_global.max():.3f}], mean={p_vals_global.mean():.3f}")
    
    # Method 2: Query-specific null (CORRECT)
    mu_q = np.mean(noise_scores)
    sigma_q = np.std(noise_scores)
    z_scores_specific = (noise_scores - mu_q) / sigma_q
    p_vals_specific = 1 - stats.norm.cdf(z_scores_specific)
    query_specific_pvalues.extend(p_vals_specific)
    
    print(f"  Query-specific: p-values range [{p_vals_specific.min():.3f}, {p_vals_specific.max():.3f}], mean={p_vals_specific.mean():.3f}")
    print()

global_pvalues = np.array(global_pvalues)
query_specific_pvalues = np.array(query_specific_pvalues)

# Statistical tests
print("="*80)
print("Statistical Tests for Uniformity:")
print("="*80 + "\n")

# Test 1: Global method
ks_stat_global, ks_pval_global = stats.kstest(global_pvalues, 'uniform')
print("Global Null Method:")
print(f"  Mean: {global_pvalues.mean():.3f} (should be ~0.5)")
print(f"  Std: {global_pvalues.std():.3f} (should be ~0.289)")
print(f"  KS statistic: {ks_stat_global:.4f}")
print(f"  KS p-value: {ks_pval_global:.4f}")
if ks_pval_global < 0.05:
    print(f"  ❌ FAILED: p-values are NOT uniform (p={ks_pval_global:.4f} < 0.05)")
else:
    print(f"  ✓ PASSED: p-values are uniform (p={ks_pval_global:.4f} >= 0.05)")
print()

# Test 2: Query-specific method
ks_stat_specific, ks_pval_specific = stats.kstest(query_specific_pvalues, 'uniform')
print("Query-Specific Null Method:")
print(f"  Mean: {query_specific_pvalues.mean():.3f} (should be ~0.5)")
print(f"  Std: {query_specific_pvalues.std():.3f} (should be ~0.289)")
print(f"  KS statistic: {ks_stat_specific:.4f}")
print(f"  KS p-value: {ks_pval_specific:.4f}")
if ks_pval_specific < 0.05:
    print(f"  ❌ FAILED: p-values are NOT uniform (p={ks_pval_specific:.4f} < 0.05)")
else:
    print(f"  ✓ PASSED: p-values are uniform (p={ks_pval_specific:.4f} >= 0.05)")
print()

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Global method - histogram
axes[0, 0].hist(global_pvalues, bins=20, density=True, alpha=0.7, edgecolor='black')
axes[0, 0].axhline(1.0, color='r', linestyle='--', label='Uniform (y=1.0)')
axes[0, 0].set_xlabel('P-value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Global Null: Histogram (KS p={ks_pval_global:.3f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Global method - Q-Q plot
stats.probplot(global_pvalues, dist='uniform', plot=axes[0, 1])
axes[0, 1].set_title('Global Null: Q-Q Plot vs Uniform')
axes[0, 1].grid(True, alpha=0.3)

# Query-specific - histogram
axes[1, 0].hist(query_specific_pvalues, bins=20, density=True, alpha=0.7, edgecolor='black', color='green')
axes[1, 0].axhline(1.0, color='r', linestyle='--', label='Uniform (y=1.0)')
axes[1, 0].set_xlabel('P-value')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title(f'Query-Specific Null: Histogram (KS p={ks_pval_specific:.3f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Query-specific - Q-Q plot
stats.probplot(query_specific_pvalues, dist='uniform', plot=axes[1, 1])
axes[1, 1].set_title('Query-Specific Null: Q-Q Plot vs Uniform')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pvalue_comparison.png', dpi=150)
print("\nSaved visualization to: pvalue_comparison.png")
print("\nConclusion:")
print("  The query-specific null method produces uniform p-values under H0,")
print("  while the global null method fails due to entity bias in the queries.")
