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
BATCH_SIZE=8

# MODEL_TYPE="17Dec_Qwen3VL4B_rmv_35_7-Qwen"
# MODEL_TYPE="5Jan_Qwen3VL4b_rmv_28_7-Qwen"
# MODEL_TYPE="9Jan_Qwen3VL4b_rmv_14_7-Qwen"
# MODEL_TYPE="17Dec_Qwen3VL2B_original-Qwen"
# MODEL_TYPE="original-Qwen"
# MODEL_TYPE="18Jan_Qwen3VL4B_rmv_28_26_24_22_20_18_16-Qwen"
# MODEL_TYPE="Qwen3VL4B_uploaded"
# MODEL_TYPE="21Jan_Qwen3VL4B_rmv_28_14-Qwen"
# MODEL_TYPE="23Jan_Qwen3VL4B_rmv_28_14_more_datasets-Qwen"
# MODEL_TYPE="28Jan_Qwen3VL4B_rmv_30_23-Qwen"
MODEL_TYPE="3Feb_Qwen3VL4B_rmv_30_23_lora24-Qwen"

# EVAL_TYPE="eval_after_train"
# DATA_CONFIG_PATH="experiments/public/eval/retrieval.yaml"
# EVAL_TYPE="eval_new_datasets"
# DATA_CONFIG_PATH="experiments/public/eval/retrieval_new_datasets.yaml"
# EVAL_TYPE="eval_cls_datasets"
# DATA_CONFIG_PATH="experiments/public/eval/image_cls.yaml"
EVAL_TYPE="eval_all"
DATA_CONFIG_PATH="experiments/public/eval/image_all.yaml"



BASE_MODEL="Qwen3-VL-4B-Instruct"
CHECKPOINT_LIST=(
  # "checkpoint-1000"
  # "checkpoint-2000"
  # "checkpoint-3000"
  # "checkpoint-4000"
  # "checkpoint-4500"
  # "checkpoint-5000"
  # "checkpoint-5500"
  # "checkpoint-6000"
  "checkpoint-9000"
)
DATA_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/data/vlm2vec_eval/MMEB-V2"
# OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/vlm2vec_retrieval"
OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/${EVAL_TYPE}/${MODEL_TYPE}"

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
      --lora_r 24 \
      --pooling eos \
      --normalize true \
      --per_device_eval_batch_size "$BATCH_SIZE" \
      --model_backbone "$MODEL_BACKBONE" \
      --model_name "$MODEL_NAME" \
      --dataset_config "$DATA_CONFIG_PATH" \
      --encode_output_path "$OUTPUT_PATH" \
      --data_basedir "$DATA_BASEDIR" \
      --delete_L 30 \
      --delete_n 23 \
      --eval_layers -1 \
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
