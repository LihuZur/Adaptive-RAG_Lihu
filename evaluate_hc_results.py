#!/usr/bin/env python3
"""
Script to evaluate and compare HC configurations across different hyperparameters.
Usage: python evaluate_hc_results.py <strategy> <model> <dataset> <split>
Example: python evaluate_hc_results.py hc_qa gpt hotpotqa test
"""

import os
import sys
import json
import glob
from typing import List, Dict, Tuple

def parse_config_from_path(path: str) -> Dict[str, str]:
    """Extract hyperparameters from prediction folder path."""
    # Example path: predictions/test/hc_qa_gpt_hotpotqa____prompt_set_1___bm25_retrieval_count__30___gamma__0.2___min_hc__0.0/
    parts = path.split('___')
    config = {}
    
    for part in parts:
        if '__' in part:
            key_value = part.split('__')
            if len(key_value) >= 2:
                key = key_value[0]
                value = '__'.join(key_value[1:])  # Handle cases with multiple __
                if key == 'bm25_retrieval_count':
                    config['retrieval'] = value
                elif key == 'gamma':
                    config['gamma'] = value
                elif key == 'min_hc':
                    config['min_hc'] = value
    
    return config

def load_evaluation_metrics(eval_file: str) -> Dict[str, float]:
    """Load metrics from evaluation_metrics JSON file."""
    try:
        with open(eval_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def find_hc_predictions(strategy: str, model: str, dataset: str, split: str) -> List[Tuple[Dict, Dict]]:
    """Find all HC prediction folders and their metrics."""
    base_pattern = f"predictions/{split}/{strategy}_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__*"
    prediction_dirs = glob.glob(base_pattern)
    
    results = []
    for pred_dir in prediction_dirs:
        config = parse_config_from_path(pred_dir)
        
        # Find evaluation metrics file
        eval_pattern = f"{pred_dir}/evaluation_metrics__{dataset}_to_{dataset}__{split}_subsampled.json"
        eval_files = glob.glob(eval_pattern)
        
        if eval_files:
            metrics = load_evaluation_metrics(eval_files[0])
            if metrics:
                results.append((config, metrics))
    
    return results

def print_results_table(results: List[Tuple[Dict, Dict]]):
    """Print formatted table of HC results."""
    if not results:
        print("No results found!")
        return
    
    # Sort by EM (descending)
    sorted_results = sorted(results, key=lambda x: x[1].get('em', 0), reverse=True)
    
    print("\n" + "="*90)
    print("HC Results - All Metrics (sorted by EM)")
    print("="*90)
    print(f"{'Rank':<6}{'Retr':<7}{'Gamma':<7}{'MinHC':<7}{'EM':<7}{'F1':<7}{'Prec':<7}{'Recall':<7}{'SP_EM':<7}{'SP_F1':<7}")
    print("-"*90)
    
    for rank, (config, metrics) in enumerate(sorted_results, 1):
        retrieval = config.get('retrieval', 'N/A')
        gamma = config.get('gamma', 'N/A')
        min_hc = config.get('min_hc', 'N/A')
        
        em = metrics.get('em', 0)
        f1 = metrics.get('f1', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        sp_em = metrics.get('sp_em', 0)
        sp_f1 = metrics.get('sp_f1', 0)
        
        star = "⭐" if rank == 1 else ""
        
        print(f"{rank:<6}{retrieval:<7}{gamma:<7}{min_hc:<7}{em:<7.3f}{f1:<7.3f}"
              f"{precision:<7.3f}{recall:<7.3f}{sp_em:<7.3f}{sp_f1:<7.3f}  {star}")
    
    print("="*90)
    
    # Print best config details
    best_config, best_metrics = sorted_results[0]
    print(f"\n🏆 BEST CONFIG: retrieval={best_config.get('retrieval', 'N/A')}, "
          f"gamma={best_config.get('gamma', 'N/A')}, min_hc={best_config.get('min_hc', 'N/A')}")
    print(f"   Answer: EM={best_metrics.get('em', 0):.3f}, F1={best_metrics.get('f1', 0):.3f}, "
          f"Precision={best_metrics.get('precision', 0):.3f}, Recall={best_metrics.get('recall', 0):.3f}")
    if 'sp_em' in best_metrics and 'sp_f1' in best_metrics:
        print(f"   Supporting Facts: SP_EM={best_metrics.get('sp_em', 0):.3f}, "
              f"SP_F1={best_metrics.get('sp_f1', 0):.3f}")
    print()

def main():
    if len(sys.argv) != 5:
        print("Usage: python evaluate_hc_results.py <strategy> <model> <dataset> <split>")
        print("Example: python evaluate_hc_results.py hc_qa gpt hotpotqa test")
        sys.exit(1)
    
    strategy = sys.argv[1]
    model = sys.argv[2]
    dataset = sys.argv[3]
    split = sys.argv[4]
    
    print(f"\nSearching for {strategy} {model} {dataset} {split} results...")
    
    results = find_hc_predictions(strategy, model, dataset, split)
    
    if not results:
        print(f"\nNo results found in predictions/{split}/ for {strategy}_{model}_{dataset}")
        print("Make sure the predictions have been generated and evaluated.")
        sys.exit(1)
    
    print(f"Found {len(results)} configurations\n")
    print_results_table(results)

if __name__ == "__main__":
    main()
