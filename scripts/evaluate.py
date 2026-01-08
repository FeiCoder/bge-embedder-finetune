import argparse
import json
import os
import numpy as np
import torch
from FlagEmbedding import FlagModel
from tqdm import tqdm
from typing import List, Dict, Set

def load_corpus(corpus_file: str):
    """
    Load corpus from jsonl file.
    Returns:
        corpus_ids: List[str] - list of document IDs
        corpus_texts: List[str] - list of document texts
        id2index: Dict[str, int] - mapping from doc ID to index
    """
    print(f"Loading corpus from {corpus_file}...")
    corpus_ids = []
    corpus_texts = []
    id2index = {}
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            data = json.loads(line)
            docid = data.get('docid')
            text = data.get('text', '')
            
            if docid:
                id2index[docid] = len(corpus_ids)
                corpus_ids.append(docid)
                corpus_texts.append(text)
                
    print(f"Loaded {len(corpus_ids)} documents.")
    return corpus_ids, corpus_texts, id2index

def load_test_data(test_file: str):
    """
    Load test data (queries and ground truth) from jsonl file.
    Returns:
        queries: List[str] - list of query texts
        qrels: Dict[int, Set[str]] - mapping from query index to set of relevant doc IDs
    """
    print(f"Loading test queries from {test_file}...")
    queries = []
    qrels = {} # query_index -> Set[doc_id]
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            query_text = data.get('query', '').strip()
            # Handle positive passages
            # Data format: 'positive_passages': [{'docid': '...', 'text': '...'}]
            pos_passages = data.get('positive_passages', [])
            
            relevant_docs = set()
            for p in pos_passages:
                if isinstance(p, dict):
                    docid = p.get('docid')
                    if docid:
                        relevant_docs.add(docid)
                elif isinstance(p, str): # Fallback if it's just a list of IDs
                    relevant_docs.add(p)
            
            if query_text and relevant_docs:
                qrels[len(queries)] = relevant_docs
                queries.append(query_text)
                
    print(f"Loaded {len(queries)} queries with relevant documents.")
    return queries, qrels

def compute_metrics(q_embs, p_embs, qrels: Dict[int, Set[str]], id2index: Dict[str, int], corpus_ids: List[str], k_values=[10]):
    """
    Compute Recall@k and NDCG@k.
    """
    print("Computing metrics...")
    
    # Normalize for cosine similarity
    # FlagModel usually does normalization if specified, but let's ensure it.
    # Actually corpus_embeddings from FlagModel might strictly be dot product compatible if normalized.
    # We will assume they are normalized or we normalize them here.
    
    # Use torch for GPU acceleration of search if possible
    if torch.cuda.is_available():
        q_embs_tensor = torch.tensor(q_embs).cuda()
        p_embs_tensor = torch.tensor(p_embs).cuda()
    else:
        q_embs_tensor = torch.tensor(q_embs)
        p_embs_tensor = torch.tensor(p_embs)
        
    # Chunking query search to avoid OOM if many queries (though 800 is fine)
    batch_size = 100
    num_queries = len(q_embs)
    
    all_recalls = {k: [] for k in k_values}
    all_ndcgs = {k: [] for k in k_values}
    
    for i in tqdm(range(0, num_queries, batch_size), desc="Searching"):
        end_idx = min(i + batch_size, num_queries)
        q_batch = q_embs_tensor[i:end_idx]
        
        # Dot product
        scores = torch.matmul(q_batch, p_embs_tensor.t())
        
        # Get top max_k indices
        max_k = max(k_values)
        topk_scores, topk_indices = torch.topk(scores, k=max_k)
        
        topk_indices = topk_indices.cpu().numpy()
        
        # Calculate metrics for this batch
        for j, q_idx_in_batch in enumerate(range(i, end_idx)):
            # Global query index
            q_idx = q_idx_in_batch 
            if q_idx not in qrels:
                continue
                
            relevant_docids = qrels[q_idx]
            
            retrieved_indices = topk_indices[j]
            retrieved_docids = [corpus_ids[idx] for idx in retrieved_indices]
            
            for k in k_values:
                # Recall@k
                hits = 0
                retrieved_k = retrieved_docids[:k]
                for docid in retrieved_k:
                    if docid in relevant_docids:
                        hits += 1
                recall = hits / len(relevant_docids) if len(relevant_docids) > 0 else 0.0
                all_recalls[k].append(recall)
                
                # NDCG@k
                dcg = 0.0
                idcg = 0.0
                
                # Check rel at each position
                rel_at_k = [1 if docid in relevant_docids else 0 for docid in retrieved_k]
                
                for rank, rel in enumerate(rel_at_k):
                    if rel:
                        dcg += 1.0 / np.log2(rank + 2)
                        
                # Ideal DCG
                num_rel = len(relevant_docids)
                for rank in range(min(num_rel, k)):
                    idcg += 1.0 / np.log2(rank + 2)
                    
                ndcg = dcg / idcg if idcg > 0 else 0.0
                all_ndcgs[k].append(ndcg)

    metrics = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = np.mean(all_recalls[k])
        metrics[f"NDCG@{k}"] = np.mean(all_ndcgs[k])
        
    return metrics


def parallel_encode(model: FlagModel, texts: List[str], batch_size: int, max_len: int = 512):
    """
    Encode texts using DataParallel for multi-GPU support.
    """
    # Safety check for device
    if hasattr(model, "device"):
        device = model.device 
    elif hasattr(model.model, "device"):
        device = model.model.device
    else:
        # Fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_model = model.model
    
    # Explicitly move to cuda:0 to satisfy DataParallel requirements
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        original_model.to(device)
    else:
        device = torch.device("cpu")
    
    # Wrap model with DataParallel if available and multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference...")
        if not isinstance(original_model, torch.nn.DataParallel):
            parallel_model = torch.nn.DataParallel(original_model)
        else:
            parallel_model = original_model
    else:
        parallel_model = original_model
    
    all_embeddings = []
    
    # Interleaved tokenization and inference
    for start_index in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[start_index : start_index + batch_size]
        
        # Tokenize
        inputs = model.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_len
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use parallel_model here, not model.model
            outputs = parallel_model(**inputs)
            
            # Extract embeddings safely
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                embeddings = outputs[0][:, 0]
                
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        all_embeddings.append(embeddings.cpu().numpy())

    
    return np.concatenate(all_embeddings, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--corpus_file", type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--query_instruction", type=str, default="为这个句子生成表示以用于检索相关文章：", help="Instruction for queries")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding")
    
    args = parser.parse_args()
    
    print(f"Evaluating model: {args.model_name_or_path}")
    
    # 1. Load Data
    corpus_ids, corpus_texts, id2index = load_corpus(args.corpus_file)
    queries, qrels = load_test_data(args.test_file)
    
    # 2. Load Model
    model = FlagModel(
        args.model_name_or_path, 
        query_instruction_for_retrieval=args.query_instruction,
        use_fp16=True
    )
    
    # 3. Encode Corpus
    print("Encoding corpus...")
    # Use custom parallel encode instead of model.encode_corpus
    corpus_embeddings = parallel_encode(model, corpus_texts, batch_size=args.batch_size)
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
    
    # 4. Encode Queries
    print("Encoding queries...")
    # Use custom parallel encode to utilize large batch/multi-gpu even for queries (though 800 is small)
    query_embeddings = parallel_encode(model, queries, batch_size=args.batch_size)
    print(f"Query embeddings shape: {query_embeddings.shape}")
    
    # 5. Compute Metrics
    metrics = compute_metrics(query_embeddings, corpus_embeddings, qrels, id2index, corpus_ids)
    
    print("-" * 30)
    print(f"Results for {args.model_name_or_path}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
