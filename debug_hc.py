"""Debug HC computation with actual CrossEntityQA scores."""
import numpy as np
import pickle
from commaqa.hc.higher_criticism import HigherCriticism

# Load null distribution
with open('processed_data/hc_null_distributions/crossentityqa_null_dist.pkl', 'rb') as f:
    null_dist = pickle.load(f)

print("Null Distribution:")
print(f"  n_samples: {null_dist.n_samples}")
print(f"  mean: {null_dist.mean:.3f}")
print(f"  std: {null_dist.std:.3f}")
print(f"  min: {null_dist.min_val:.3f}")
print(f"  max: {null_dist.max_val:.3f}")
print()

# Actual retrieval scores
actual_scores = np.array([28.099, 24.378, 22.249, 21.216, 21.086, 20.659, 20.426, 19.921, 19.557, 19.315])

print("Actual Retrieval Scores:")
for i, score in enumerate(actual_scores):
    print(f"  [{i}] {score:.3f}")
print()

# Create HC instance
hc = HigherCriticism(null_distribution=null_dist)

# Compute p-values
p_values = hc.compute_p_values(actual_scores)
print("P-values:")
for i, (score, pval) in enumerate(zip(actual_scores, p_values)):
    print(f"  [{i}] score={score:.3f} -> p={pval:.6f}")
print()

# Test different gamma values
for gamma in [0.3, 0.5, 0.7, 1.0]:
    print(f"Testing gamma={gamma}:")
    
    # Compute HC statistic
    hc_stat, best_idx = hc.compute_hc_statistic(actual_scores, gamma=gamma)
    print(f"  HC statistic: {hc_stat:.3f}, best_idx: {best_idx}")
    
    # Compute threshold
    result = hc.compute_hc_threshold(actual_scores, gamma=gamma, min_hc=0.0)
    print(f"  Result: k={result.k}, threshold={result.threshold:.3f}, hc={result.hc_statistic:.3f}")
    print()
