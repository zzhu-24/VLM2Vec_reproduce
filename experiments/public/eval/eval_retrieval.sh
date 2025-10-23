#!/bin/bash
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
# GPU Availability Check
# ==============================================================================
echo "==> Checking available GPUs..."

# Check if any GPU is visible
if ! command -v nvidia-smi &> /dev/null; then
  echo "❌ nvidia-smi not found! Make sure CUDA drivers are installed."
  exit 1
fi

# List available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | awk '{print $1}' | xargs)
if [ -z "$AVAILABLE_GPUS" ]; then
  echo "❌ No GPUs detected on this node!"
  exit 1
fi

echo "Detected GPUs: $AVAILABLE_GPUS"

# Optionally check if the selected GPUs are busy
BUSY_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk '$2 > 50 {print $1}')
if [ -n "$BUSY_GPUS" ]; then
  echo "⚠️ Warning: Some GPUs are currently under high utilization (>50%): $BUSY_GPUS"
else
  echo "✅ GPUs are free and ready to use."
fi

echo ""

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="0"
BATCH_SIZE=24
DATA_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/data/vlm2vec_eval/MMEB-V2"
# OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/vlm2vec_retrieval"
OUTPUT_BASEDIR="/home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/exps/token_average"
# # colpali cannot use average token

LAYER_START=1
LAYER_END=28
# # colpali baseline layer index 0-18

# ==> Model specs: format "MODEL_NAME;BACKBONE;OUTPUT_DIR"
declare -a MODEL_SPECS
# MODEL_SPECS+=("VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B")
# MODEL_SPECS+=("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct")
# MODEL_SPECS+=("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct")
MODEL_SPECS+=("Qwen/Qwen2-VL-2B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/qwen-Qwen2-VL-2B-Instruct")
MODEL_SPECS+=("Qwen/Qwen2-VL-7B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/qwen-Qwen2-VL-7B-Instruct")
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

  for L in $(seq $LAYER_START $LAYER_END); do
    OUTPUT_PATH="$BASE_OUTPUT_PATH/layer_${L}/retrieval"
    mkdir -p "$OUTPUT_PATH"

    echo "  ▶ Layer $L → output: $OUTPUT_PATH"
    start_time_layer=$(date +%s)

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python eval.py \
      --pooling average \
      --normalize true \
      --per_device_eval_batch_size "$BATCH_SIZE" \
      --model_backbone "$MODEL_BACKBONE" \
      --model_name "$MODEL_NAME" \
      --dataset_config "$DATA_CONFIG_PATH" \
      --encode_output_path "$OUTPUT_PATH" \
      --data_basedir "$DATA_BASEDIR" \
      --qry_chosen_layer "$L" \
      --tgt_chosen_layer "$L" \
      --attn_implementation eager \
      &> "$OUTPUT_PATH/eval.log"

    end_time_layer=$(date +%s)
    elapsed_layer=$((end_time_layer - start_time_layer))
    echo "    ✅ Finished layer $L in ${elapsed_layer}s"
  done

  end_time_model=$(date +%s)
  elapsed_model=$((end_time_model - start_time_model))
  echo "-------------------------------------------------"
  echo "✅ Finished $MODEL_NAME (layers ${LAYER_START}-${LAYER_END}) in ${elapsed_model}s"
  echo "-------------------------------------------------"
done

echo "🎯 All per-layer retrieval evaluations completed."
