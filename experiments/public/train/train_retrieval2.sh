#!/bin/bash
# NOTE: replace ... with actual paths
# export LD_LIBRARY_PATH=...
# export PATH=...
# echo "conda location: $(which conda)"
# echo "Python location: $(which python)"
# echo "Python version: $(python --version)"

set -e  # Stop on error
trap 'echo "❌ Script failed at line $LINENO"; exit 1' ERR

echo "==> Environment"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""


# export HF_DATASETS_CACHE=...
# export HF_HOME=...
export WANDB_DISABLED=false
export WANDB_PROJECT=vlm2vec_train
export WANDB_API_KEY=151b985aec8f2669c89875abb20b1c822ecdb9ad
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_PROJECT=...
export WANDB_RUN_GROUP=2Dec_AddTail_Replace_EOSInit_freezeTail
export MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
# export MODEL_NAME=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
export WANDB_NAME="${WANDB_RUN_GROUP}-${MODEL_NAME}"
export EXP_NAME=$WANDB_NAME
export EXP_DIR=/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/train/$WANDB_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

export DATA_BASEDIR=/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/data/vlm2vec_train/MMEB-train

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd /home/infres/zzhu-24/PRIM/VLM2Vec/

cmd="CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29121 --max_restarts=0 train.py 
    --lora
    --lora_r 16
    --model_name $MODEL_NAME
    --bf16
    --pooling eos
    --normalize True
    --temperature 0.02
    --dataloader_num_workers 1
    --dataset_config /home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/train/retrieval.yaml
    --data_basedir "$DATA_BASEDIR"
    --run_name $EXP_NAME
    --output_dir $EXP_DIR
    --grad_cache True
    --per_device_train_batch_size 16
    --gc_q_chunk_size 8
    --gc_p_chunk_size 8
    --interleave_batch_size 0
    --lr_scheduler_type linear
    --learning_rate 1e-5 
    --max_steps 6000
    --warmup_steps 100
    --save_steps 50
    --logging_steps 1
    --save_safetensors True
    --remove_unused_columns False
    --resume_from auto
    --plus_one_token True
    --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd