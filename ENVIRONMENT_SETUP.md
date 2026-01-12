# 环境设置指南

本文档说明如何在目标集群上设置 VLM2Vec 运行环境。

## 文件说明

项目提供了三个 requirements 文件：

1. **`requirements.txt`** - 推荐使用
   - 包含所有核心包和依赖，带有注释和分组
   - 排除了 Jupyter 相关包（可选安装）
   - 适合大多数情况

2. **`requirements_full.txt`** - 完整环境
   - 从 conda 环境直接导出的完整包列表（187个包）
   - 包含所有依赖，包括 Jupyter 相关包
   - 确保环境完全一致

3. **`requirements_minimal.txt`** - 最小安装
   - 仅包含核心包，依赖会自动安装
   - 适合快速测试或空间受限的情况

## 安装步骤

### 方法1：使用 requirements.txt（推荐）

```bash
# 创建新的 conda 环境
conda create -n vlm2vec python=3.10.18 -y
conda activate vlm2vec

# 安装 PyTorch（根据目标集群的 CUDA 版本选择）
# CUDA 12.1
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 或者 CUDA 11.8
# pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 安装 Flash Attention（可能需要从源码编译）
# 如果预编译版本不可用，需要从源码安装：
# pip install flash-attn --no-build-isolation
```

### 方法2：使用 requirements_full.txt（完全一致）

```bash
# 创建新的 conda 环境
conda create -n vlm2vec python=3.10.18 -y
conda activate vlm2vec

# 直接安装所有包（包括 Jupyter）
pip install -r requirements_full.txt
```

### 方法3：使用 requirements_minimal.txt（最小安装）

```bash
# 创建新的 conda 环境
conda create -n vlm2vec python=3.10.18 -y
conda activate vlm2vec

# 安装 PyTorch
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 安装核心包
pip install -r requirements_minimal.txt
```

## 特殊包说明

### Flash Attention

Flash Attention 可能需要从源码编译，特别是如果目标集群的 CUDA 版本与预编译版本不匹配。

```bash
# 方法1：尝试安装预编译版本
pip install flash-attn==2.8.3

# 方法2：如果失败，从源码编译
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install .

# 或者使用 pip 从源码安装
pip install flash-attn --no-build-isolation
```

### Qwen VL Utils

```bash
pip install qwen-vl-utils==0.0.8
# 如果需要 decord 支持
pip install "qwen-vl-utils[decord]==0.0.8"
```

## 验证安装

```bash
# 激活环境
conda activate vlm2vec

# 验证关键包
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import flash_attn; print('Flash Attention installed')"
python -c "import qwen_vl_utils; print('Qwen VL Utils installed')"
```

## 环境信息

- **Python 版本**: 3.10.18
- **PyTorch 版本**: 2.4.0+cu121
- **CUDA 版本**: 12.1（根据目标集群调整）
- **Transformers 版本**: 4.57.0

## 常见问题

### 1. CUDA 版本不匹配

如果目标集群的 CUDA 版本不同，需要调整 PyTorch 安装：

```bash
# CUDA 11.8
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.4.0 torchvision==0.19.0
```

### 2. Flash Attention 编译失败

确保安装了编译工具：

```bash
# 安装编译依赖
sudo apt-get install build-essential
pip install ninja packaging wheel

# 然后重新安装 flash-attn
pip install flash-attn --no-build-isolation
```

### 3. 依赖冲突

如果遇到依赖冲突，建议使用 `requirements_full.txt` 以确保所有版本完全一致。

## 与 Hugging Face 缓存配合使用

安装完环境后，还需要拷贝 Hugging Face 缓存文件（参考 `MIGRATION_GUIDE.md`）。
