#!/usr/bin/env bash

export PYTORCH_SDP_KERNEL=math

MODEL="EleutherAI/pythia-14m"

# 0: input embeddings on GPU 0
#CUDA_VISIBLE_DEVICES=0 python run_transformer_cli.py -m "$MODEL" -c embed_in &

# # QKV layers on GPUs 1–6
# for L in 0 1 2 3 4 5; do
#   GPU=$((L+1))
#   echo "Launching layer $L on GPU $GPU"
#   CUDA_VISIBLE_DEVICES=$GPU python run_transformer_cli.py \
#     -m "$MODEL" \
#     -c attention.query_key_value.weight \
#     -l "$L" &
# done

#CUDA_VISIBLE_DEVICES=0 python run_transformer_cli.py -m "$MODEL" -c attention.dense.weight -l "5" &

for L in 0 1 2 3 4; do
  GPU=$((L+1))
  echo "Launching layer $L on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU python run_transformer_cli.py \
    -m "$MODEL" \
    -c attention.dense.weight \
    -l "$L" &
done

# wait for all QKV + input-embed jobs
#wait

# run output embedding
#CUDA_VISIBLE_DEVICES=0 python run_transformer_cli.py -m "$MODEL" -c embed_out