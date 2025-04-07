import os
import json
import glob
from typing import List, Dict

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    docs = []
    for file_path in glob.glob(os.path.join(data_dir, '**/*.json'), recursive=True):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract disease case section
        main_keys = [k for k in data if not k.startswith("input")]

        # Extract all inputs as separate chunks
        for input_key in [f"input{i}" for i in range(1, 7)]:
            if input_key in data and data[input_key].strip():
                docs.append({
                    "type": input_key,
                    "content": data[input_key],
                    "source_file": file_path
                })

        # Extract reasoning rules and evidence
        for key in main_keys:
            section = data[key]
            for rule, evidence_dict in section.items():
                evidence = list(evidence_dict.keys())
                docs.append({
                    "type": "rule",
                    "content": f"Rule: {rule}. Evidence: {'; '.join(evidence)}",
                    "source_file": file_path
                })
    return docs


def prepare_documents_with_metadata(docs: List[Dict[str, str]]) -> List[Dict[str, Dict]]:
    """
    Takes raw docs (from load_documents) and adds metadata structure.
    
    Args:
        docs: List of dictionaries with 'type', 'content', and 'source_file'.

    Returns:
        List of dictionaries with 'text' and 'metadata' keys.
    """
    rag_ready_docs = []
    for doc in docs:
        rag_ready_docs.append({
            "text": doc["content"],
            "metadata": {
                "type": doc["type"],
                "source_file": doc["source_file"]
            }
        })
    return rag_ready_docs

data_dir = 'Finished'
docs = load_documents(data_dir)
docs=prepare_documents_with_metadata(docs)
print(len(docs))
# for _ in range(1000,2000):
#     print(docs[_])
#     print()



