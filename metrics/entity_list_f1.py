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


def extract_entity_tokens(entity: str) -> Set[str]:
    """
    Extract meaningful tokens from an entity name.
    Splits by spaces and removes very common words.
    """
    # Normalize first
    entity = normalize_entity(entity)
    
    # Remove ordinal numbers (1st, 2nd, etc.), common title words, and company suffixes
    stop_words = {'of', 'the', 'a', 'an', 'and', '1st', '2nd', '3rd', '4th', '5th', 
                  '6th', '7th', '8th', '9th', '10th', 'earl', 'duke', 'marquess',
                  'count', 'baron', 'sir', 'lord', 'lady', 'film', 'bank', 'group',
                  'corp', 'inc', 'ltd', 'llc', 'company', 'corporation', 'incorporated',
                  'co', 'limited', 'plc', 'gmbh', 'ag'}
    
    tokens = entity.split()
    # Keep tokens that are meaningful (3+ chars and not stop words)
    meaningful_tokens = set()
    for token in tokens:
        # Remove commas, parentheses, dots and other punctuation
        token = token.strip(',.()[]').replace('.', '')
        # Skip years (4-digit numbers)
        if token.isdigit() and len(token) == 4:
            continue
        if len(token) >= 2 and token not in stop_words:  # Lowered to 2 chars for "X", "xAI" etc.
            meaningful_tokens.add(token)
    
    return meaningful_tokens


def extract_entities(text: str) -> List[Set[str]]:
    """
    Extract entities from text by splitting on " and " or commas.
    Returns a list of token sets, one per entity.
    """
    if not text:
        return []
    
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
    
    # Extract token sets for each entity
    entity_token_sets = []
    for entity in entities:
        tokens = extract_entity_tokens(entity)
        if tokens:  # Skip empty token sets
            entity_token_sets.append(tokens)
    
    return entity_token_sets


def entity_match(pred_tokens: Set[str], gt_tokens: Set[str], threshold: float = 0.5) -> bool:
    """
    Check if two entities match based on token overlap.
    Uses subset matching if prediction is fully contained in GT, otherwise uses Jaccard similarity.
    
    Args:
        pred_tokens: Token set from predicted entity
        gt_tokens: Token set from ground truth entity
        threshold: Minimum Jaccard similarity to consider a match
    
    Returns:
        True if entities match, False otherwise
    """
    if not pred_tokens or not gt_tokens:
        return False
    
    # If all prediction tokens appear in ground truth, it's a match (subset matching)
    # This handles cases like "Duel" matching "Duel (1971 film)"
    if pred_tokens.issubset(gt_tokens):
        return True
    
    # Otherwise, compute Jaccard similarity: intersection / union
    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens
    
    if not union:
        return False
    
    similarity = len(intersection) / len(union)
    return similarity >= threshold


def compute_entity_f1(predicted: str, ground_truth: str, threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for entity lists using fuzzy token matching.
    
    Args:
        predicted: Predicted answer string with entities separated by " and " or commas
        ground_truth: Ground truth string with entities separated by " and "
        threshold: Minimum token overlap similarity for matching
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_entities = extract_entities(predicted)
    gt_entities = extract_entities(ground_truth)
    
    if len(pred_entities) == 0 and len(gt_entities) == 0:
        return 1.0, 1.0, 1.0
    
    if len(pred_entities) == 0 or len(gt_entities) == 0:
        return 0.0, 0.0, 0.0
    
    # Match predicted entities to ground truth using greedy matching
    matched_gt = set()
    matched_pred = 0
    
    for pred_tokens in pred_entities:
        # Find best matching GT entity
        best_match = None
        best_similarity = 0.0
        
        for i, gt_tokens in enumerate(gt_entities):
            if i in matched_gt:
                continue
            
            if entity_match(pred_tokens, gt_tokens, threshold):
                intersection = pred_tokens & gt_tokens
                union = pred_tokens | gt_tokens
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = i
        
        if best_match is not None:
            matched_gt.add(best_match)
            matched_pred += 1
    
    if matched_pred == 0:
        return 0.0, 0.0, 0.0
    
    precision = matched_pred / len(pred_entities)
    recall = len(matched_gt) / len(gt_entities)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_exact_match(predicted: str, ground_truth: str, threshold: float = 0.5) -> int:
    """
    Compute exact match: 1 if all entities match (using fuzzy matching), 0 otherwise.
    """
    pred_entities = extract_entities(predicted)
    gt_entities = extract_entities(ground_truth)
    
    if len(pred_entities) != len(gt_entities):
        return 0
    
    # Try to match all entities
    matched_gt = set()
    
    for pred_tokens in pred_entities:
        found_match = False
        for i, gt_tokens in enumerate(gt_entities):
            if i in matched_gt:
                continue
            
            if entity_match(pred_tokens, gt_tokens, threshold):
                matched_gt.add(i)
                found_match = True
                break
        
        if not found_match:
            return 0
    
    # All entities matched
    return 1


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
