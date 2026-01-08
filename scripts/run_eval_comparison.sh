#!/bin/bash

# Environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Paths
BASE_MODEL="/data/zf/models/BAAI/bge-base-zh-v1.5/"
FINETUNED_MODEL="output/bge-base-zh-v1.5-finetuned"
CORPUS_FILE="/data/zf/Datasets/mldr-v1.0-zh/corpus.jsonl"
TEST_FILE="/data/zf/Datasets/mldr-v1.0-zh/test.jsonl"
BATCH_SIZE=2048

echo "========================================================"
echo "Evaluating Base Model: $BASE_MODEL"
echo "========================================================"
python scripts/evaluate.py \
    --model_name_or_path "$BASE_MODEL" \
    --corpus_file "$CORPUS_FILE" \
    --test_file "$TEST_FILE" \
    --batch_size $BATCH_SIZE

echo ""
echo "========================================================"
echo "Evaluating Finetuned Model: $FINETUNED_MODEL"
echo "========================================================"
# Check if finetuned model exists
if [ -d "$FINETUNED_MODEL" ]; then
    python scripts/evaluate.py \
        --model_name_or_path "$FINETUNED_MODEL" \
        --corpus_file "$CORPUS_FILE" \
        --test_file "$TEST_FILE" \
        --batch_size $BATCH_SIZE
else
    echo "Finetuned model directory not found at $FINETUNED_MODEL"
    echo "Please run scripts/run_finetune.sh first."
fi
