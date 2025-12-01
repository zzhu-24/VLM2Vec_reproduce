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
export WANDB_RUN_GROUP=02Dec_Delete1-10
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

cmd="CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=2207 --max_restarts=0 train.py 
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
    --delete_L 10
    --delete_n 10
    --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd










# ===========================================
# VLM2Vec Evaluation: COCO & FashionIQ only
# Per-layer embedding traversal version
# ===========================================

set -e  # Stop on error
trap 'echo "❌ Script failed at line $LINENO"; exit 1' ERR

echo "==> Environment"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# ==============================================================================
# Configuration
# ==============================================================================

CUDA_VISIBLE_DEVICES="0"
BATCH_SIZE=24

MODEL_TYPE="${WANDB_RUN_GROUP}-Qwen"

CHECKPOINT_LIST=(
    "checkpoint-50"
    "checkpoint-100"
    "checkpoint-300"
    "checkpoint-500"
    "checkpoint-1000"
    "checkpoint-1500"
    "checkpoint-3000"
    "checkpoint-6000"
)

DATA_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/data/vlm2vec_eval/MMEB-V2"
# OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/vlm2vec_retrieval"
OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/eval_after_train/DEBUG_${MODEL_TYPE}"
# # colpali cannot use average token

LAYER_START=1
LAYER_END=18

# ==> Model specs: format "MODEL_NAME;BACKBONE;OUTPUT_DIR"
declare -a MODEL_SPECS
# MODEL_SPECS+=("VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B")
# MODEL_SPECS+=("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct")
# MODEL_SPECS+=("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct")
MODEL_SPECS+=("Qwen/Qwen2-VL-2B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/qwen-Qwen2-VL-2B-Instruct")
# MODEL_SPECS+=("Qwen/Qwen2-VL-7B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/qwen-Qwen2-VL-7B-Instruct")
# MODEL_SPECS+=("TIGER-Lab/VLM2Vec-Qwen2VL-7B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-Qwen2VL-7B")
# MODEL_SPECS+=("code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret")
# MODEL_SPECS+=("vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3")

# ==============================================================================
# Main Loop
# ==============================================================================
for spec in "${MODEL_SPECS[@]}"; do
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH <<< "$spec"

  echo "================================================="
  echo "🚀 Evaluating Model: $MODEL_NAME"
  echo "================================================="

  DATA_CONFIG_PATH="experiments/public/eval/retrieval.yaml"

  echo "  ➤ Dataset config: $DATA_CONFIG_PATH"
  echo "  ➤ Output base path: $BASE_OUTPUT_PATH"
  echo "  ➤ Layer range: $LAYER_START to $((LAYER_END))"
  echo ""

  start_time_model=$(date +%s)

  for CKPT in "${CHECKPOINT_LIST[@]}"; do
    echo "==============================================="
    echo "🔎 Using checkpoint: $CKPT"
    echo "==============================================="

    # Path for checkpoint
    CKPT_PATH="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/train/${MODEL_TYPE}/Qwen2-VL-2B-Instruct/${CKPT}"


    for L in $(seq $LAYER_START $LAYER_END); do
      OUTPUT_PATH="$BASE_OUTPUT_PATH/$CKPT/layer_${L}/retrieval"
      mkdir -p "$OUTPUT_PATH"

      echo "  ▶ Layer $L → output: $OUTPUT_PATH"
      start_time_layer=$(date +%s)

      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python eval.py \
        --lora True \
        --lora_r 16 \
        --pooling eos \
        --normalize true \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --model_backbone "$MODEL_BACKBONE" \
        --model_name "$MODEL_NAME" \
        --dataset_config "$DATA_CONFIG_PATH" \
        --encode_output_path "$OUTPUT_PATH" \
        --data_basedir "$DATA_BASEDIR" \
        --qry_chosen_layer "$L" \
        --tgt_chosen_layer "$L" \
        --checkpoint_path "$CKPT_PATH" \
        &> "$OUTPUT_PATH/eval.log"

      end_time_layer=$(date +%s)
      elapsed_layer=$((end_time_layer - start_time_layer))
      echo "    ✅ Finished layer $L in ${elapsed_layer}s"

      # ===============================
      # Cleanup except for the first run
      # ===============================
      if [ "$first_run" = false ]; then
          echo "    ➤ Cleaning temporary files under $OUTPUT_PATH"
          find "$OUTPUT_PATH" -maxdepth 1 -type f \( \
              -name "*_tgt" -o \
              -name "*_qry" -o \
              -name "*_pred.jsonl" -o \
              -name "*_info.jsonl" \
          \) -delete
      else
          echo "    ➤ First iteration: skipping cleanup"
          first_run=false
      fi
    done
  done

  end_time_model=$(date +%s)
  elapsed_model=$((end_time_model - start_time_model))
  echo "-------------------------------------------------"
  echo "✅ Finished $MODEL_NAME (layers ${LAYER_START}-${LAYER_END}) in ${elapsed_model}s"
  echo "-------------------------------------------------"
done

echo "🎯 All per-layer retrieval evaluations completed."
