"""
Higher Criticism (HC) Module for Adaptive-RAG

Adapted from HC-RAG implementation for BM25-based retrieval.
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple, Dict
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class HCThresholdResult(NamedTuple):
    """
    Result from HC threshold computation.

    Attributes:
        threshold: BM25 score threshold for document selection
        hc_statistic: Maximum HC statistic value
        k: Number of documents to retrieve (0 for empty set)
    """
    threshold: float
    hc_statistic: float
    k: int


class HigherCriticism:
    """
    Higher Criticism statistical test for adaptive document retrieval.

    Uses HC statistics to identify which retrieved documents are
    statistically significant (likely relevant) vs random noise.
    
    Adapted for BM25 scores from Elasticsearch retrieval.
    """

    def __init__(
        self, 
        null_distribution: Optional['NullDistribution'] = None,
        query_specific_null: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize Higher Criticism module.

        Args:
            null_distribution: Pre-computed global null distribution for BM25 scores (legacy)
            query_specific_null: Dict mapping qid -> {"mu": float, "sigma": float} (preferred)
        """
        self.null_distribution = null_distribution
        self.query_specific_null = query_specific_null
        logger.debug("HigherCriticism module initialized")

        if null_distribution is not None:
            logger.debug(f"  Loaded global null distribution with {null_distribution.n_samples} samples")
        
        if query_specific_null is not None:
            logger.debug(f"  Loaded query-specific null for {len(query_specific_null)} queries")

    def compute_p_values(
        self, 
        scores: np.ndarray,
        query_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute p-values for BM25 scores.
        
        If query_specific_null is available and query_id is provided, uses
        Z-score normalization: Z = (score - μ_q) / σ_q, then p = 1 - Φ(Z).
        
        Otherwise falls back to global null distribution.

        Args:
            scores: Array of observed BM25 scores
            query_id: Query identifier (required for query-specific null)

        Returns:
            Array of p-values (same shape as scores)
        """
        # Try query-specific null first (preferred)
        if self.query_specific_null is not None and query_id is not None:
            if query_id not in self.query_specific_null:
                logger.warning(f"Query {query_id} not in query-specific null, falling back to global")
            else:
                return self._compute_p_values_query_specific(scores, query_id)
        
        # Fall back to global null distribution
        if self.null_distribution is None:
            raise ValueError("No null distribution available (neither query-specific nor global).")
        
        return self._compute_p_values_global(scores)
    
    def _compute_p_values_global(self, scores: np.ndarray) -> np.ndarray:
        """Compute p-values using global null distribution (legacy method)."""
        null_scores = self.null_distribution.scores
        N_null = len(null_scores)

        # Sort null distribution once for binary search
        sorted_null = np.sort(null_scores)

        # Vectorized p-value computation
        indices = np.searchsorted(sorted_null, scores, side='left')
        n_greater_equal = N_null - indices

        # Monte Carlo calibration
        p_values = (n_greater_equal + 1.0) / (N_null + 1.0)

        # Clip to prevent 0/1 in HC denominator
        eps = 1.0 / (N_null + 1.0)
        p_values = np.clip(p_values, eps, 1.0 - eps)

        return p_values
    
    def _compute_p_values_query_specific(self, scores: np.ndarray, query_id: str) -> np.ndarray:
        """
        Compute p-values using query-specific null (Z-score normalization).
        
        For query Q with null statistics μ_q and σ_q:
        1. Z-score normalize: Z = (score - μ_q) / σ_q
        2. Convert to p-value: p = 1 - Φ(Z) where Φ is standard normal CDF
        
        This ensures p-values are uniform under H0 (null hypothesis).
        """
        query_stats = self.query_specific_null[query_id]
        mu = query_stats['mu']
        sigma = query_stats['sigma']
        
        # Z-score normalization
        z_scores = (scores - mu) / sigma
        
        # Convert to p-values using standard normal CDF
        # p = P(Z > z) = 1 - Φ(z)
        p_values = 1.0 - stats.norm.cdf(z_scores)
        
        # Clip to prevent numerical issues in HC denominator
        eps = 1e-10
        p_values = np.clip(p_values, eps, 1.0 - eps)
        
        return p_values

    def compute_hc_statistic(
        self,
        scores: np.ndarray,
        gamma: float = 0.1,
        query_id: Optional[str] = None
    ) -> Tuple[float, int]:
        """
        Compute Higher Criticism statistic.

        HC statistic measures the maximum deviation between observed p-values
        and uniform distribution, focusing on small p-values (likely relevant docs).

        Args:
            scores: Array of BM25 scores (will be sorted descending)
            gamma: Fraction of top scores to search for HC maximum (0 < gamma <= 1)
            query_id: Query identifier (for query-specific null)

        Returns:
            Tuple of (hc_statistic, best_index)
        """
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        if len(scores) == 0:
            return 0.0, 0

        # Sort scores in descending order (highest first)
        sorted_scores = np.sort(scores)[::-1]

        # Compute p-values for sorted scores
        p_values = self.compute_p_values(sorted_scores, query_id=query_id)

        n = len(p_values)
        n_gamma = min(n, max(1, int(np.floor(gamma * n))))

        # Compute HC statistic for each position
        hc_values = []
        for i in range(1, n_gamma + 1):
            expected_quantile = i / n
            p_i = p_values[i - 1]
            numerator = expected_quantile - p_i
            denominator = np.sqrt(p_i * (1 - p_i) / n)

            if denominator > 0:
                hc_i = numerator / denominator
                hc_values.append(hc_i)
            else:
                hc_values.append(0.0)

        if hc_values:
            max_hc = float(np.max(hc_values))
            best_idx = int(np.argmax(hc_values))
        else:
            max_hc = 0.0
            best_idx = 0

        return max_hc, best_idx

    def compute_hc_threshold(
        self,
        scores: np.ndarray,
        gamma: float = 0.1,
        min_hc: float = 0.0,
        allow_empty: bool = True,
        query_id: Optional[str] = None
    ) -> HCThresholdResult:
        """
        Compute HC-based threshold for adaptive retrieval.

        The threshold is determined by the position where HC is maximized.
        Selection rule: retrieve the top-k documents where k is determined by HC.

        Args:
            scores: Array of BM25 scores from retrieval
            gamma: Fraction of top scores to search for HC maximum
            min_hc: Minimum HC statistic to accept any documents
            allow_empty: If True, can return empty set when HC < min_hc
            query_id: Query identifier (for query-specific null)

        Returns:
            HCThresholdResult with threshold, hc_statistic, and k
        """
        if len(scores) == 0:
            return HCThresholdResult(threshold=-np.inf, hc_statistic=0.0, k=0)

        # Sort scores descending
        sorted_scores = np.sort(scores)[::-1]
        n = len(sorted_scores)

        # Compute HC statistic and find best cutoff
        hc_stat, best_idx = self.compute_hc_statistic(sorted_scores, gamma=gamma, query_id=query_id)

        # Check HC gate: if below minimum, return empty set
        if allow_empty and hc_stat < min_hc:
            logger.debug(
                f"HC statistic {hc_stat:.3f} below min_hc={min_hc:.3f}, "
                f"returning empty set"
            )
            return HCThresholdResult(threshold=-np.inf, hc_statistic=float(hc_stat), k=0)

        # Convert index to k
        k = max(0, min(n, best_idx + 1))

        # Ensure at least 1 document if allow_empty=False
        if not allow_empty and k == 0:
            k = 1

        # Determine threshold based on k
        if k > 0:
            threshold = float(sorted_scores[k - 1])
        else:
            threshold = -np.inf

        return HCThresholdResult(threshold=threshold, hc_statistic=float(hc_stat), k=int(k))
