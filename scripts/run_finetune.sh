#!/bin/bash

# 设置环境变量，使用所有4块 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 配置路径
MODEL_PATH="/data/zf/models/BAAI/bge-base-zh-v1.5/"
TRAIN_DATA="data/finetune_data.jsonl"
OUTPUT_DIR="output/bge-base-zh-v1.5-finetuned"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 运行微调
# 注意：BGE v1.5 建议对 query 使用 instruction
# 使用 4 GPUs 进行分布式训练，增大 batch size 利用显存
torchrun --nproc_per_node 4 \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --train_group_size 8 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --fp16 \
    --logging_steps 10 \
    --save_steps 500 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章："
