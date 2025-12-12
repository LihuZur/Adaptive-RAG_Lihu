"""Null Distribution for Higher Criticism with BM25 scores."""

import numpy as np
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class NullDistribution:
    """Container for null distribution of BM25 scores."""
    scores: np.ndarray
    mean: float
    std: float
    min_val: float
    max_val: float
    n_samples: int

    def __repr__(self):
        return (f"NullDistribution(n={self.n_samples}, "
                f"mean={self.mean:.4f}, std={self.std:.4f})")

    def save(self, path: str):
        """Save to disk."""
        path = Path(path).with_suffix('.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved null distribution to {path}")

    @staticmethod
    def load(path: str) -> "NullDistribution":
        """Load from disk."""
        path = Path(path).with_suffix('.pkl')
        with open(path, 'rb') as f:
            dist = pickle.load(f)
        logger.info(f"Loaded null distribution from {path}")
        return dist


class BM25NullDistribution:
    """
    Build null distribution from BM25 scores of non-relevant documents.
    
    Adapted for Adaptive-RAG's Elasticsearch-based retrieval system.
    """

    def __init__(
        self,
        retriever_host: str,
        retriever_port: int,
        corpus_name: str,
        seed: int = 42
    ):
        """
        Initialize BM25 null distribution builder.

        Args:
            retriever_host: Elasticsearch retriever host
            retriever_port: Elasticsearch retriever port
            corpus_name: Name of the corpus (e.g., 'hotpotqa')
            seed: Random seed
        """
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.corpus_name = corpus_name
        self.seed = seed
        np.random.seed(seed)

    def build_from_dataset(
        self,
        dataset_path: str,
        n_negatives_per_query: int = 50,
        max_queries: Optional[int] = None
    ) -> NullDistribution:
        """
        Build null distribution from dataset.

        Args:
            dataset_path: Path to dataset JSONL file
            n_negatives_per_query: Number of negative documents to sample per query
            max_queries: Maximum number of queries to process (None = all)

        Returns:
            NullDistribution object
        """
        import jsonlines
        import requests

        logger.info(f"Building null distribution from {dataset_path}")
        logger.info(f"  Negatives per query: {n_negatives_per_query}")
        logger.info(f"  Max queries: {max_queries if max_queries else 'all'}")

        all_scores = []
        n_processed = 0

        with jsonlines.open(dataset_path) as reader:
            for query_data in reader:
                if max_queries and n_processed >= max_queries:
                    break

                query_text = query_data.get('question', '')
                if not query_text:
                    continue

                # Get relevant titles for this query
                relevant_titles = set()
                if 'titles' in query_data:
                    relevant_titles = set(query_data['titles'])
                elif 'context' in query_data:
                    # Extract titles from context
                    for para in query_data['context']:
                        if isinstance(para, list) and len(para) > 0:
                            relevant_titles.add(para[0])

                # Retrieve documents for this query
                try:
                    params = {
                        "retrieval_method": "retrieve_from_elasticsearch",
                        "query_text": query_text,
                        "max_hits_count": n_negatives_per_query * 2,  # Get more to filter
                        "corpus_name": self.corpus_name,
                        "document_type": "title_paragraph_text",
                    }
                    url = f"{self.retriever_host.rstrip('/')}:{self.retriever_port}/retrieve"
                    response = requests.post(url, json=params, timeout=30)

                    if response.ok:
                        retrieval = response.json()["retrieval"]

                        # Filter to only negative (non-relevant) documents
                        negative_scores = []
                        for item in retrieval:
                            title = item.get("title", "")
                            score = item.get("score", 0.0)

                            # Check if this is a negative (not in relevant titles)
                            if title not in relevant_titles:
                                negative_scores.append(score)

                            # Stop once we have enough negatives
                            if len(negative_scores) >= n_negatives_per_query:
                                break

                        all_scores.extend(negative_scores)
                        n_processed += 1

                        if (n_processed) % 100 == 0:
                            logger.info(f"  Processed {n_processed} queries, collected {len(all_scores)} scores")

                except Exception as e:
                    logger.warning(f"  Failed to retrieve for query: {e}")
                    continue

        if not all_scores:
            raise ValueError("No scores collected! Check retriever connection and dataset.")

        scores_array = np.array(all_scores, dtype=np.float32)

        logger.info(f"Built null distribution from {n_processed} queries")
        logger.info(f"  Total scores: {len(scores_array)}")
        logger.info(f"  Mean: {np.mean(scores_array):.4f}")
        logger.info(f"  Std: {np.std(scores_array):.4f}")
        logger.info(f"  Range: [{np.min(scores_array):.4f}, {np.max(scores_array):.4f}]")

        return NullDistribution(
            scores=scores_array,
            mean=float(np.mean(scores_array)),
            std=float(np.std(scores_array)),
            min_val=float(np.min(scores_array)),
            max_val=float(np.max(scores_array)),
            n_samples=len(scores_array)
        )

    def build_quick(self, n_samples: int = 10000) -> NullDistribution:
        """
        Build a quick null distribution using random query-document pairs.
        
        This is a fallback method that doesn't require dataset processing.
        Uses random words as queries to get random BM25 scores.

        Args:
            n_samples: Number of random samples to collect

        Returns:
            NullDistribution object
        """
        import requests

        logger.info(f"Building quick null distribution with {n_samples} samples")

        all_scores = []
        random_words = ["the", "a", "is", "of", "and", "to", "in", "for", "on", "with",
                       "at", "by", "from", "as", "it", "was", "are", "be", "this", "that"]

        n_collected = 0
        attempts = 0
        max_attempts = n_samples * 3

        while n_collected < n_samples and attempts < max_attempts:
            # Generate random query
            query_words = np.random.choice(random_words, size=np.random.randint(2, 5), replace=True)
            query_text = " ".join(query_words)

            try:
                params = {
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": query_text,
                    "max_hits_count": 20,
                    "corpus_name": self.corpus_name,
                    "document_type": "title_paragraph_text",
                }
                url = f"{self.retriever_host.rstrip('/')}:{self.retriever_port}/retrieve"
                response = requests.post(url, json=params, timeout=10)

                if response.ok:
                    retrieval = response.json()["retrieval"]
                    for item in retrieval:
                        score = item.get("score", 0.0)
                        all_scores.append(score)
                        n_collected += 1
                        if n_collected >= n_samples:
                            break

            except:
                pass

            attempts += 1

            if (n_collected) % 1000 == 0 and n_collected > 0:
                logger.info(f"  Collected {n_collected}/{n_samples} scores")

        if not all_scores:
            raise ValueError("Failed to collect scores! Check retriever connection.")

        scores_array = np.array(all_scores[:n_samples], dtype=np.float32)

        logger.info(f"Built quick null distribution")
        logger.info(f"  Total scores: {len(scores_array)}")
        logger.info(f"  Mean: {np.mean(scores_array):.4f}")
        logger.info(f"  Std: {np.std(scores_array):.4f}")

        return NullDistribution(
            scores=scores_array,
            mean=float(np.mean(scores_array)),
            std=float(np.std(scores_array)),
            min_val=float(np.min(scores_array)),
            max_val=float(np.max(scores_array)),
            n_samples=len(scores_array)
        )
