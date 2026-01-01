"""
Preprocessing script to add ground truth to CrossEntityQA queries.

For each query_id:
1. Find all relevant passage_ids from qrels.tsv
2. For each passage_id, get the entity_label from corpus.jsonl
3. Aggregate entity_labels with commas between them
4. Add as "ground_truth" field to queries.jsonl
"""

import json
from collections import defaultdict
from tqdm import tqdm

# File paths
QRELS_PATH = "CrossEntityQA/qrels.tsv"
CORPUS_PATH = "CrossEntityQA/corpus.jsonl"
QUERIES_PATH = "CrossEntityQA/queries.jsonl"
OUTPUT_PATH = "CrossEntityQA/queries_with_ground_truth.jsonl"


def load_qrels(qrels_path):
    """Load qrels.tsv and create a mapping from query_id to list of passage_ids."""
    print(f"Loading qrels from {qrels_path}...")
    query_to_passages = defaultdict(list)
    
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                passage_id = parts[2]
                query_to_passages[query_id].append(passage_id)
    
    print(f"Loaded {len(query_to_passages)} queries with relevant passages")
    return query_to_passages


def load_corpus(corpus_path):
    """Load corpus.jsonl and create a mapping from passage_id to entity_label."""
    print(f"Loading corpus from {corpus_path}...")
    passage_to_entity = {}
    
    with open(corpus_path, 'r') as f:
        for line in tqdm(f, desc="Loading corpus"):
            if line.strip():
                doc = json.loads(line)
                passage_id = doc['passage_id']
                entity_label = doc['entity_label']
                passage_to_entity[passage_id] = entity_label
    
    print(f"Loaded {len(passage_to_entity)} passages")
    return passage_to_entity


def generate_ground_truth(passage_ids, passage_to_entity):
    """
    Generate ground truth by aggregating entity_labels from passage_ids.
    
    Args:
        passage_ids: List of passage_ids for a query
        passage_to_entity: Dict mapping passage_id to entity_label
    
    Returns:
        String with entity_labels joined by ", "
    """
    entity_labels = []
    for passage_id in passage_ids:
        if passage_id in passage_to_entity:
            entity_label = passage_to_entity[passage_id]
            entity_labels.append(entity_label)
        else:
            print(f"Warning: passage_id {passage_id} not found in corpus")
    
    return ", ".join(entity_labels)


def process_queries(queries_path, output_path, query_to_passages, passage_to_entity):
    """Process queries.jsonl and add ground_truth field."""
    print(f"Processing queries from {queries_path}...")
    
    processed_count = 0
    missing_count = 0
    
    with open(queries_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, desc="Processing queries"):
            if line.strip():
                query = json.loads(line)
                query_id = query['query_id']
                
                # Get relevant passage_ids for this query
                if query_id in query_to_passages:
                    passage_ids = query_to_passages[query_id]
                    ground_truth = generate_ground_truth(passage_ids, passage_to_entity)
                    query['ground_truth'] = ground_truth
                    processed_count += 1
                else:
                    print(f"Warning: query_id {query_id} not found in qrels")
                    query['ground_truth'] = ""
                    missing_count += 1
                
                fout.write(json.dumps(query) + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Queries with ground truth: {processed_count}")
    print(f"Queries missing from qrels: {missing_count}")
    print(f"Output written to: {output_path}")


def main():
    # Load qrels mapping
    query_to_passages = load_qrels(QRELS_PATH)
    
    # Load corpus mapping
    passage_to_entity = load_corpus(CORPUS_PATH)
    
    # Process queries and add ground truth
    process_queries(QUERIES_PATH, OUTPUT_PATH, query_to_passages, passage_to_entity)
    
    # Show example
    print("\n" + "="*80)
    print("Example verification:")
    print("="*80)
    with open(OUTPUT_PATH, 'r') as f:
        first_query = json.loads(f.readline())
        print(f"Query ID: {first_query['query_id']}")
        print(f"Query Text: {first_query['query_text']}")
        print(f"Ground Truth: {first_query['ground_truth']}")


if __name__ == "__main__":
    main()
