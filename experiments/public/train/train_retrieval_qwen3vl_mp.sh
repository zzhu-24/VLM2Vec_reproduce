#!/bin/bash
# 模型并行训练脚本 - 专门针对 Qwen3-VL-4B-Instruct
# 使用模型分块（模型并行），将模型分配到 2 个 GPU 上

set -e  # Stop on error
trap 'echo "❌ Script failed at line $LINENO"; exit 1' ERR

echo "==> Environment"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

export WANDB_DISABLED=false
export WANDB_PROJECT=vlm2vec_train
export WANDB_API_KEY=151b985aec8f2669c89875abb20b1c822ecdb9ad
export WANDB_RUN_GROUP=12Jan_Qwen3VL4b_mp_2gpu
export MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct
export WANDB_NAME="${WANDB_RUN_GROUP}-${MODEL_NAME}"
export EXP_NAME=$WANDB_NAME
export EXP_DIR=/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/train/$WANDB_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

export DATA_BASEDIR=/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/data/vlm2vec_train/MMEB-train

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd /home/infres/zzhu-24/PRIM/VLM2Vec/

# 模型并行：使用 2 个 GPU，不使用 torchrun（因为不是数据并行）
# 直接运行 Python 脚本，脚本内部会处理模型并行
cmd="CUDA_VISIBLE_DEVICES=0,1 python train_qwen3vl_mp.py 
    --lora
    --lora_r 16
    --model_name $MODEL_NAME
    --bf16
    --pooling eos
    --normalize True
    --temperature 0.02
    --dataloader_num_workers 2
    --dataset_config /home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/train/retrieval.yaml
    --data_basedir "$DATA_BASEDIR"
    --run_name $EXP_NAME
    --output_dir $EXP_DIR
    --grad_cache True
    --per_device_train_batch_size 2
    --gc_q_chunk_size 1
    --gc_p_chunk_size 1
    --interleave_batch_size 0
    --lr_scheduler_type linear
    --learning_rate 1e-5 
    --max_steps 6000
    --warmup_steps 100
    --save_steps 500
    --logging_steps 1
    --save_safetensors True
    --remove_unused_columns False
    --resume_from auto
    --delete_L 36
    --delete_n 0
    --joint_training_layers -1
    --eval_layers -1
    --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
