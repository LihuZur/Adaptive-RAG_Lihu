"""
Entity List F1 Metric for CrossEntityQA
Compares entity lists separated by " and " regardless of order.
Computes precision, recall, and F1 based on set overlap.
"""
import re
from typing import List, Tuple, Set
import ftfy

from metrics.metric import Metric


def normalize_entity(entity: str) -> str:
    """Normalize a single entity by removing extra whitespace and lowercasing."""
    return entity.strip().lower()


def extract_entities(text: str) -> Set[str]:
    """
    Extract entities from text by splitting on " and " or commas.
    Handles both formats for robustness.
    Returns a set of normalized entities.
    """
    if not text:
        return set()
    
    # Fix encoding issues
    text = ftfy.fix_text(text)
    
    # Try splitting by " and " first (preferred format)
    if " and " in text:
        entities = text.split(" and ")
    # Fallback to comma splitting if no " and " found
    elif "," in text:
        entities = text.split(",")
    else:
        # Single entity
        entities = [text]
    
    # Normalize each entity
    normalized = set()
    for entity in entities:
        norm = normalize_entity(entity)
        if norm:  # Skip empty strings
            normalized.add(norm)
    
    return normalized


def compute_entity_f1(predicted: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for entity lists.
    
    Args:
        predicted: Predicted answer string with entities separated by " and " or commas
        ground_truth: Ground truth string with entities separated by " and "
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_entities = extract_entities(predicted)
    gt_entities = extract_entities(ground_truth)
    
    if len(pred_entities) == 0 and len(gt_entities) == 0:
        return 1.0, 1.0, 1.0
    
    if len(pred_entities) == 0 or len(gt_entities) == 0:
        return 0.0, 0.0, 0.0
    
    # Compute overlap
    overlap = pred_entities & gt_entities
    num_overlap = len(overlap)
    
    if num_overlap == 0:
        return 0.0, 0.0, 0.0
    
    precision = num_overlap / len(pred_entities)
    recall = num_overlap / len(gt_entities)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1


def compute_exact_match(predicted: str, ground_truth: str) -> int:
    """
    Compute exact match: 1 if entity sets are identical, 0 otherwise.
    Order doesn't matter.
    """
    pred_entities = extract_entities(predicted)
    gt_entities = extract_entities(ground_truth)
    return int(pred_entities == gt_entities)


class EntityListF1Metric(Metric):
    """
    Metric for evaluating entity list predictions.
    Splits by " and " (or commas as fallback), compares as sets.
    """
    
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):
        """
        Evaluate prediction against ground truth(s).
        
        Args:
            predicted_answer: Single prediction string
            ground_truth_answers: List of ground truth strings (typically just one for CrossEntityQA)
        """
        # Handle wrapped formats
        if isinstance(predicted_answer, list):
            predicted_answer = predicted_answer[0]
        if isinstance(ground_truth_answers[0], tuple):
            ground_truth_answers = [i for i in ground_truth_answers[0]]
        
        # Fix encoding
        predicted_answer = ftfy.fix_text(predicted_answer)
        ground_truth_answers = [ftfy.fix_text(e) for e in ground_truth_answers]
        
        assert isinstance(predicted_answer, str)
        assert isinstance(ground_truth_answers, (Tuple, List))
        
        # Compute metrics against each ground truth, take max
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_em = 0
        
        for gt in ground_truth_answers:
            precision, recall, f1 = compute_entity_f1(predicted_answer, gt)
            em = compute_exact_match(predicted_answer, gt)
            
            if f1 > best_f1:
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_em = em
        
        self._total_em += best_em
        self._total_precision += best_precision
        self._total_recall += best_recall
        self._total_f1 += best_f1
        self._count += 1

    def get_metric(self, reset: bool = False) -> dict:
        """Return aggregated metrics."""
        if self._count == 0:
            return {"em": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
        
        em = self._total_em / self._count
        precision = self._total_precision / self._count
        recall = self._total_recall / self._count
        f1 = self._total_f1 / self._count
        
        if reset:
            self.reset()
        
        return {
            "em": round(em, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "count": self._count
        }

    def reset(self):
        """Reset all counters."""
        self._total_em = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._total_f1 = 0.0
        self._count = 0
