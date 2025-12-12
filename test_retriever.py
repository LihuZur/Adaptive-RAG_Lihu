"""Quick test script to debug retriever connection."""
import requests
import json

url = "http://127.0.0.1:8000/retrieve"
params = {
    "retrieval_method": "retrieve_from_elasticsearch",
    "query_text": "What is the capital of France?",
    "max_hits_count": 5,
    "corpus_name": "hotpotqa",
    "document_type": "title_paragraph_text",
}

print(f"Testing URL: {url}")
print(f"Params: {json.dumps(params, indent=2)}")
print("-" * 50)

try:
    response = requests.post(url, json=params, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print("-" * 50)
    
    if response.ok:
        result = response.json()
        print(f"Response keys: {list(result.keys())}")
        retrieval = result.get("retrieval", [])
        print(f"Retrieved {len(retrieval)} items")
        
        if retrieval:
            print(f"\nFirst item keys: {list(retrieval[0].keys())}")
            print(f"First item corpus_name: {retrieval[0].get('corpus_name')}")
            print(f"First item score: {retrieval[0].get('score')}")
            
            scores = []
            for item in retrieval:
                if item.get("corpus_name") == "hotpotqa":
                    score = item.get("score", 0.0)
                    scores.append(score)
            
            print(f"\nExtracted {len(scores)} scores: {scores}")
        else:
            print("No retrieval results!")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
