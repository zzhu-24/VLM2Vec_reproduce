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

export HF_ENDPOINT="https://hf-mirror.com"

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export HF_HUB_ENABLE_HF_TRANSFER=1

export WANDB_DISABLED=false
export WANDB_PROJECT=vlm2vec_train
export WANDB_API_KEY=151b985aec8f2669c89875abb20b1c822ecdb9ad
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_PROJECT=...
export WANDB_RUN_GROUP=23Jan_Qwen3VL4B_rmv_28_14_more_datasets
export MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct
# export MODEL_NAME=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
export WANDB_NAME="${WANDB_RUN_GROUP}-${MODEL_NAME}"
export EXP_NAME=$WANDB_NAME
export EXP_DIR=/home/zhuzhehua/2025/VLM2Vec_reproduce/experiments/public/exps/train/$WANDB_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

export DATA_BASEDIR=/home/zhuzhehua/2025/VLM2Vec_reproduce/experiments/public/data/vlm2vec_train/MMEB-train

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd ~/2025/VLM2Vec_reproduce

cmd="torchrun --nproc_per_node=1 --master_port=2208 --max_restarts=0 train.py 
    --lora
    --lora_r 16
    --model_name $MODEL_NAME
    --bf16
    --pooling eos
    --normalize True
    --temperature 0.02
    --dataloader_num_workers 2
    --dataset_config /home/zhuzhehua/2025/VLM2Vec_reproduce/experiments/public/train/train_image.yaml
    --data_basedir "$DATA_BASEDIR"
    --run_name $EXP_NAME
    --output_dir $EXP_DIR
    --grad_cache True
    --per_device_train_batch_size 16
    --gc_q_chunk_size 4
    --gc_p_chunk_size 4
    --interleave_batch_size 0
    --lr_scheduler_type linear
    --learning_rate 1e-5 
    --max_steps 6000
    --warmup_steps 100
    --save_steps 500
    --logging_steps 1
    --save_safetensors True
    --remove_unused_columns False
    --delete_L 28
    --delete_n 14
    --joint_training_layers -1
    --eval_layers -1
    --report_to none 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd