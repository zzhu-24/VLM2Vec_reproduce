#!/bin/bash
# ===========================================
# VLM2Vec Evaluation: COCO & FashionIQ only
# Per-layer embedding traversal version
# ===========================================

set -e  # Stop on error
trap 'echo "Script failed at line $LINENO"; exit 1' ERR

echo "==> Environment"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

CUDA_VISIBLE_DEVICES="0"
BATCH_SIZE=24
MODEL_TYPE="5Jan_Qwen3VL4b_rmv_21_7-Qwen"
BASE_MODEL="Qwen3-VL-4B-Instruct"
CHECKPOINT_LIST=(
  "checkpoint-1000"
  "checkpoint-2000"
  "checkpoint-3000"
  "checkpoint-4000"
  # "checkpoint-4500"
  # "checkpoint-4650"
)
DATA_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/data/vlm2vec_eval/MMEB-V2"
# OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/vlm2vec_retrieval"
OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/eval_after_train/DEBUG_${MODEL_TYPE}"

# ==> Model specs: format "MODEL_NAME;BACKBONE;OUTPUT_DIR"
declare -a MODEL_SPECS
MODEL_SPECS+=("Qwen/${BASE_MODEL};qwen3_vl;$OUTPUT_BASEDIR/qwen-${BASE_MODEL}")
# MODEL_SPECS+=("VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B")
# MODEL_SPECS+=("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct")
# MODEL_SPECS+=("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct")
# MODEL_SPECS+=("Qwen/Qwen2-VL-2B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/qwen-Qwen2-VL-2B-Instruct")
# MODEL_SPECS+=("Qwen/Qwen2-VL-7B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/qwen-Qwen2-VL-7B-Instruct")
# MODEL_SPECS+=("TIGER-Lab/VLM2Vec-Qwen2VL-7B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-Qwen2VL-7B")
# MODEL_SPECS+=("code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret")
# MODEL_SPECS+=("vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3")

for spec in "${MODEL_SPECS[@]}"; do
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH <<< "$spec"

  echo "================================================="
  echo "Evaluating Model: $MODEL_NAME"
  echo "================================================="

  DATA_CONFIG_PATH="experiments/public/eval/retrieval.yaml"

  echo "  ➤ Dataset config: $DATA_CONFIG_PATH"
  echo "  ➤ Output base path: $BASE_OUTPUT_PATH"
  echo ""

  start_time_model=$(date +%s)

  for CKPT in "${CHECKPOINT_LIST[@]}"; do
    echo "==============================================="
    echo "Using checkpoint: $CKPT"
    echo "==============================================="

    CKPT_PATH="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/train/${MODEL_TYPE}/${BASE_MODEL}/${CKPT}"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$CKPT/retrieval"
    mkdir -p "$OUTPUT_PATH"

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
      --delete_L 21 \
      --delete_n 7 \
      --eval_layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29\
      --checkpoint_path "$CKPT_PATH" \
      &> "$OUTPUT_PATH/eval.log"

    # Cleanup
    echo " ➤ Cleaning temporary files under $OUTPUT_PATH"
    find "$OUTPUT_PATH" -maxdepth 2 -type f \( \
        -name "*_tgt" -o \
        -name "*_qry" -o \
        -name "*_pred.jsonl" -o \
        -name "*_info.jsonl" \
    \) -delete
  done

  end_time_model=$(date +%s)
  elapsed_model=$((end_time_model - start_time_model))
  echo "-------------------------------------------------"
  echo "Finished $MODEL_NAME in ${elapsed_model}s"
  echo "-------------------------------------------------"
done

echo "All retrieval evaluations completed."
