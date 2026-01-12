#!/bin/bash
# 检查 Hugging Face 缓存文件脚本
# 用于确认需要拷贝到新集群的文件

set -e

HF_CACHE_DIR="${HOME}/.cache/huggingface"
HF_HUB_DIR="${HF_CACHE_DIR}/hub"

echo "=========================================="
echo "Hugging Face 缓存文件检查"
echo "=========================================="
echo ""

# 检查模型缓存
echo "1. 检查模型缓存：Qwen3-VL-4B-Instruct"
echo "----------------------------------------"
MODEL_DIR="${HF_HUB_DIR}/models--Qwen--Qwen3-VL-4B-Instruct"
if [ -d "$MODEL_DIR" ]; then
    echo "✓ 模型目录存在: $MODEL_DIR"
    echo "  目录大小: $(du -sh "$MODEL_DIR" | cut -f1)"
    echo ""
    
    # 检查关键文件
    if [ -d "$MODEL_DIR/blobs" ]; then
        BLOB_COUNT=$(find "$MODEL_DIR/blobs" -type f | wc -l)
        BLOB_SIZE=$(du -sh "$MODEL_DIR/blobs" | cut -f1)
        echo "  ✓ blobs 目录存在: $BLOB_COUNT 个文件, 总大小 $BLOB_SIZE"
    else
        echo "  ✗ blobs 目录不存在"
    fi
    
    if [ -d "$MODEL_DIR/snapshots" ]; then
        SNAPSHOT_COUNT=$(find "$MODEL_DIR/snapshots" -type d -mindepth 1 -maxdepth 1 | wc -l)
        echo "  ✓ snapshots 目录存在: $SNAPSHOT_COUNT 个快照"
        
        # 列出快照
        for snapshot in "$MODEL_DIR/snapshots"/*; do
            if [ -d "$snapshot" ]; then
                snapshot_name=$(basename "$snapshot")
                file_count=$(find "$snapshot" -type l | wc -l)
                echo "    - 快照: $snapshot_name ($file_count 个符号链接)"
            fi
        done
    else
        echo "  ✗ snapshots 目录不存在"
    fi
    
    if [ -f "$MODEL_DIR/refs/main" ]; then
        REF_CONTENT=$(cat "$MODEL_DIR/refs/main")
        echo "  ✓ refs/main 存在，指向: $REF_CONTENT"
    else
        echo "  ✗ refs/main 不存在"
    fi
else
    echo "✗ 模型目录不存在: $MODEL_DIR"
fi
echo ""

# 检查数据集缓存
echo "2. 检查数据集缓存：TIGER-Lab/MMEB-train"
echo "----------------------------------------"
DATASET_DIR="${HF_HUB_DIR}/datasets--TIGER-Lab--MMEB-train"
if [ -d "$DATASET_DIR" ]; then
    echo "✓ 数据集目录存在: $DATASET_DIR"
    echo "  目录大小: $(du -sh "$DATASET_DIR" | cut -f1)"
    echo ""
    
    # 检查关键文件
    if [ -d "$DATASET_DIR/blobs" ]; then
        BLOB_COUNT=$(find "$DATASET_DIR/blobs" -type f | wc -l)
        BLOB_SIZE=$(du -sh "$DATASET_DIR/blobs" | cut -f1)
        echo "  ✓ blobs 目录存在: $BLOB_COUNT 个文件, 总大小 $BLOB_SIZE"
    else
        echo "  ✗ blobs 目录不存在"
    fi
    
    if [ -d "$DATASET_DIR/snapshots" ]; then
        SNAPSHOT_COUNT=$(find "$DATASET_DIR/snapshots" -type d -mindepth 1 -maxdepth 1 | wc -l)
        echo "  ✓ snapshots 目录存在: $SNAPSHOT_COUNT 个快照"
        
        # 列出快照
        for snapshot in "$DATASET_DIR/snapshots"/*; do
            if [ -d "$snapshot" ]; then
                snapshot_name=$(basename "$snapshot")
                dir_count=$(find "$snapshot" -type d -mindepth 1 | wc -l)
                echo "    - 快照: $snapshot_name ($dir_count 个子目录)"
            fi
        done
    else
        echo "  ✗ snapshots 目录不存在"
    fi
    
    if [ -f "$DATASET_DIR/refs/main" ]; then
        REF_CONTENT=$(cat "$DATASET_DIR/refs/main")
        echo "  ✓ refs/main 存在，指向: $REF_CONTENT"
    else
        echo "  ✗ refs/main 不存在"
    fi
else
    echo "✗ 数据集目录不存在: $DATASET_DIR"
fi
echo ""

# 检查数据集元数据
echo "3. 检查数据集元数据缓存"
echo "----------------------------------------"
DATASETS_DIR="${HF_CACHE_DIR}/datasets"
if [ -d "$DATASETS_DIR" ]; then
    echo "✓ 数据集元数据目录存在: $DATASETS_DIR"
    MMEB_FILES=$(find "$DATASETS_DIR" -name "*TIGER-Lab*MMEB*" -o -name "*mmeb*" 2>/dev/null | wc -l)
    echo "  相关文件数量: $MMEB_FILES"
    if [ "$MMEB_FILES" -gt 0 ]; then
        echo "  相关文件列表:"
        find "$DATASETS_DIR" -name "*TIGER-Lab*MMEB*" -o -name "*mmeb*" 2>/dev/null | head -10 | sed 's/^/    - /'
    fi
else
    echo "✗ 数据集元数据目录不存在: $DATASETS_DIR"
fi
echo ""

# 检查评估数据集缓存
echo "4. 检查评估数据集缓存：ziyjiang/MMEB_Test_Instruct"
echo "----------------------------------------"
EVAL_DATASET_DIR="${HF_HUB_DIR}/datasets--ziyjiang--MMEB_Test_Instruct"
if [ -d "$EVAL_DATASET_DIR" ]; then
    echo "✓ 评估数据集目录存在: $EVAL_DATASET_DIR"
    echo "  目录大小: $(du -sh "$EVAL_DATASET_DIR" | cut -f1)"
    echo ""
    
    if [ -d "$EVAL_DATASET_DIR/blobs" ]; then
        BLOB_COUNT=$(find "$EVAL_DATASET_DIR/blobs" -type f | wc -l)
        BLOB_SIZE=$(du -sh "$EVAL_DATASET_DIR/blobs" | cut -f1)
        echo "  ✓ blobs 目录存在: $BLOB_COUNT 个文件, 总大小 $BLOB_SIZE"
    else
        echo "  ✗ blobs 目录不存在"
    fi
    
    if [ -f "$EVAL_DATASET_DIR/refs/main" ]; then
        REF_CONTENT=$(cat "$EVAL_DATASET_DIR/refs/main")
        echo "  ✓ refs/main 存在，指向: $REF_CONTENT"
    else
        echo "  ✗ refs/main 不存在"
    fi
else
    echo "✗ 评估数据集目录不存在: $EVAL_DATASET_DIR"
fi
echo ""

# 检查评估数据集实际数据缓存
echo "5. 检查评估数据集实际数据缓存"
echo "----------------------------------------"
EVAL_DATA_DIR="${HF_CACHE_DIR}/datasets/ziyjiang___mmeb_test_instruct"
if [ -d "$EVAL_DATA_DIR" ]; then
    echo "✓ 评估数据集实际数据目录存在: $EVAL_DATA_DIR"
    echo "  目录大小: $(du -sh "$EVAL_DATA_DIR" | cut -f1)"
    SUBDIR_COUNT=$(find "$EVAL_DATA_DIR" -type d -mindepth 1 -maxdepth 1 | wc -l)
    echo "  子目录数量: $SUBDIR_COUNT"
else
    echo "✗ 评估数据集实际数据目录不存在: $EVAL_DATA_DIR"
fi
echo ""

# 总结
echo "=========================================="
echo "总结"
echo "=========================================="
TOTAL_SIZE=0
if [ -d "$MODEL_DIR" ]; then
    MODEL_SIZE=$(du -sm "$MODEL_DIR" | cut -f1)
    TOTAL_SIZE=$((TOTAL_SIZE + MODEL_SIZE))
fi
if [ -d "$DATASET_DIR" ]; then
    DATASET_SIZE=$(du -sm "$DATASET_DIR" | cut -f1)
    TOTAL_SIZE=$((TOTAL_SIZE + DATASET_SIZE))
fi
if [ -d "$EVAL_DATASET_DIR" ]; then
    EVAL_DATASET_SIZE=$(du -sm "$EVAL_DATASET_DIR" | cut -f1)
    TOTAL_SIZE=$((TOTAL_SIZE + EVAL_DATASET_SIZE))
fi
if [ -d "$EVAL_DATA_DIR" ]; then
    EVAL_DATA_SIZE=$(du -sm "$EVAL_DATA_DIR" | cut -f1)
    TOTAL_SIZE=$((TOTAL_SIZE + EVAL_DATA_SIZE))
fi

echo "需要拷贝的总大小: 约 ${TOTAL_SIZE}MB ($(echo "scale=2; $TOTAL_SIZE/1024" | bc)GB)"
echo ""
echo "需要拷贝的目录："
echo "  1. ${MODEL_DIR} (模型缓存)"
echo "  2. ${DATASET_DIR} (训练数据集缓存)"
if [ -d "$EVAL_DATASET_DIR" ]; then
    echo "  3. ${EVAL_DATASET_DIR} (评估数据集缓存)"
fi
if [ -d "$EVAL_DATA_DIR" ]; then
    echo "  4. ${EVAL_DATA_DIR} (评估数据集实际数据)"
fi
echo "  5. ${DATASETS_DIR} (数据集元数据)"
echo ""
echo "详细迁移指南请查看: MIGRATION_GUIDE.md"
