import json
import os
from tqdm import tqdm

# selecet 100k samples from mldr for finetuning BGE
sample_size = 100000

def process_mldr_data(input_file, output_file, sample_size=100000):
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(tqdm(f_in)):
            if i >= sample_size:
                break
            try:
                data = json.loads(line)
                
                query = data.get('query', '').strip()
                positive_passages = data.get('positive_passages', [])
                negative_passages = data.get('negative_passages', [])
                
                if not query or not positive_passages:
                    continue
                
                # Extract text from passages
                pos_texts = [p.get('text', '').strip() for p in positive_passages if p.get('text', '').strip()]
                neg_texts = [p.get('text', '').strip() for p in negative_passages if p.get('text', '').strip()]
                
                if not pos_texts:
                    continue
                
                # BGE finetuning format
                out_record = {
                    "query": query,
                    "pos": pos_texts,
                    "neg": neg_texts
                }
                
                f_out.write(json.dumps(out_record, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print("Error decoding JSON line, skipping.")
                continue

if __name__ == "__main__":
    # Define paths
    input_path = "/data/zf/Datasets/mldr-v1.0-zh/train.jsonl"
    output_path = "data/finetune_data.jsonl"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    process_mldr_data(input_path, output_path, sample_size=sample_size)
