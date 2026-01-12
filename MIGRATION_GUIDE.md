# Hugging Face 缓存迁移指南

由于新集群无法访问 Hugging Face，需要手动拷贝以下缓存文件。

## 需要拷贝的文件和目录

### 1. 模型缓存：Qwen3-VL-4B-Instruct

**源路径：** `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/`

**目标路径（新集群）：** `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/`

**需要拷贝的内容：**
- **整个目录**（约 6.6GB）
  - `blobs/` 目录：包含所有模型文件（权重、配置文件等）
  - `snapshots/` 目录：包含符号链接
  - `refs/` 目录：包含引用文件
  - `.no_exist/` 目录（如果存在）

**重要文件：**
- `blobs/` 目录下的所有文件（包括模型权重文件）
- `snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/` 目录及其所有符号链接
- `refs/main` 文件（指向快照版本）

### 2. 数据集缓存：TIGER-Lab/MMEB-train

**源路径：** `~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/`

**目标路径（新集群）：** `~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/`

**需要拷贝的内容：**
- **整个目录**（约 146MB）
  - `blobs/` 目录：包含所有数据集文件
  - `snapshots/` 目录：包含符号链接
  - `refs/` 目录：包含引用文件
  - `.no_exist/` 目录（如果存在）

**重要文件：**
- `blobs/` 目录下的所有文件
- `snapshots/76dd0a440b6d4c02776830a804443fffbb2d0bfa/` 目录及其所有符号链接
- `refs/main` 文件（指向快照版本）

### 3. 评估数据集缓存：ziyjiang/MMEB_Test_Instruct

**源路径：** `~/.cache/huggingface/hub/datasets--ziyjiang--MMEB_Test_Instruct/`

**目标路径（新集群）：** `~/.cache/huggingface/hub/datasets--ziyjiang--MMEB_Test_Instruct/`

**需要拷贝的内容：**
- **整个目录**（约 4.9MB）
  - `blobs/` 目录：包含数据集元数据文件
  - `snapshots/` 目录：包含符号链接
  - `refs/` 目录：包含引用文件
  - `.no_exist/` 目录（如果存在）

**重要说明：** 评估脚本（eval.py）使用的数据集（如 MSCOCO_i2t、MSCOCO_t2i）都来自这个 Hugging Face 数据集。

### 4. 评估数据集实际数据缓存

**源路径：** `~/.cache/huggingface/datasets/ziyjiang___mmeb_test_instruct/`

**目标路径（新集群）：** `~/.cache/huggingface/datasets/ziyjiang___mmeb_test_instruct/`

**需要拷贝的内容：**
- **整个目录**（约 1.4GB）
  - 包含所有评估数据集的 Arrow 格式缓存文件
  - 包括 MSCOCO_i2t、MSCOCO_t2i 等子集的实际数据

### 5. 训练数据集元数据缓存

**源路径：** `~/.cache/huggingface/datasets/`

**目标路径（新集群）：** `~/.cache/huggingface/datasets/`

**需要拷贝的内容：**
- 所有以 `TIGER-Lab___mmeb-train` 开头的锁文件和元数据文件
- 特别是 `MSCOCO_t2i` 和 `MSCOCO_i2t` 相关的缓存文件

## 拷贝命令示例

### 方法1：使用 rsync（推荐，保留符号链接）

```bash
# 在源集群上执行
# 1. 拷贝模型缓存
rsync -avz --progress \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/ \
  user@new-cluster:~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/

# 2. 拷贝训练数据集缓存
rsync -avz --progress \
  ~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/ \
  user@new-cluster:~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/

# 3. 拷贝评估数据集缓存（hub目录）
rsync -avz --progress \
  ~/.cache/huggingface/hub/datasets--ziyjiang--MMEB_Test_Instruct/ \
  user@new-cluster:~/.cache/huggingface/hub/datasets--ziyjiang--MMEB_Test_Instruct/

# 4. 拷贝评估数据集实际数据缓存
rsync -avz --progress \
  ~/.cache/huggingface/datasets/ziyjiang___mmeb_test_instruct/ \
  user@new-cluster:~/.cache/huggingface/datasets/ziyjiang___mmeb_test_instruct/

# 5. 拷贝数据集元数据
rsync -avz --progress \
  ~/.cache/huggingface/datasets/ \
  user@new-cluster:~/.cache/huggingface/datasets/
```

### 方法2：使用 tar + scp（适合一次性传输）

```bash
# 在源集群上打包
cd ~/.cache/huggingface/hub/
tar -czf qwen3-vl-4b-cache.tar.gz models--Qwen--Qwen3-VL-4B-Instruct/
tar -czf mmeb-train-cache.tar.gz datasets--TIGER-Lab--MMEB-train/
tar -czf mmeb-test-cache.tar.gz datasets--ziyjiang--MMEB_Test_Instruct/

cd ~/.cache/huggingface/
tar -czf mmeb-test-data-cache.tar.gz datasets/ziyjiang___mmeb_test_instruct/
tar -czf datasets-metadata.tar.gz datasets/

# 传输到新集群
scp qwen3-vl-4b-cache.tar.gz user@new-cluster:~/.cache/huggingface/hub/
scp mmeb-train-cache.tar.gz user@new-cluster:~/.cache/huggingface/hub/
scp mmeb-test-cache.tar.gz user@new-cluster:~/.cache/huggingface/hub/
scp mmeb-test-data-cache.tar.gz user@new-cluster:~/.cache/huggingface/
scp datasets-metadata.tar.gz user@new-cluster:~/.cache/huggingface/

# 在新集群上解压
cd ~/.cache/huggingface/hub/
tar -xzf qwen3-vl-4b-cache.tar.gz
tar -xzf mmeb-train-cache.tar.gz
tar -xzf mmeb-test-cache.tar.gz

cd ~/.cache/huggingface/
tar -xzf mmeb-test-data-cache.tar.gz
tar -xzf datasets-metadata.tar.gz
```

## 验证拷贝是否成功

在新集群上执行以下命令验证：

```bash
# 检查模型缓存
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs/ | head -10

# 检查训练数据集缓存
ls -la ~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/snapshots/
ls -la ~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/blobs/ | head -10

# 检查评估数据集缓存
ls -la ~/.cache/huggingface/hub/datasets--ziyjiang--MMEB_Test_Instruct/snapshots/
ls -la ~/.cache/huggingface/datasets/ziyjiang___mmeb_test_instruct/ | head -10

# 检查文件大小
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/
du -sh ~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-train/
```

## 注意事项

1. **符号链接**：确保使用 `rsync -a` 或 `tar` 时保留符号链接，不要使用 `cp -r` 直接拷贝，否则会破坏符号链接结构。

2. **权限**：确保新集群上的用户有权限访问这些文件。

3. **环境变量**：在新集群上确保设置了正确的 Hugging Face 缓存路径：
   ```bash
   export HF_HOME=~/.cache/huggingface
   export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
   ```

4. **测试加载**：拷贝完成后，建议先测试加载模型和数据集：
   ```python
   from transformers import AutoModel, AutoProcessor
   from datasets import load_dataset
   
   # 测试模型加载
   model = AutoModel.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
   processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
   
   # 测试数据集加载
   dataset = load_dataset("TIGER-Lab/MMEB-train", "MSCOCO_t2i", split="train", streaming=False)
   ```

5. **完整目录结构**：确保拷贝整个目录，包括所有子目录和文件，不要遗漏任何内容。

## 文件大小总结

- **模型缓存**（Qwen3-VL-4B-Instruct）：约 6.6GB
  - 包含：模型权重、配置文件、tokenizer、processor 等所有文件
- **训练数据集缓存**（MMEB-train）：约 146MB
  - 包含：训练数据集的元数据和索引文件
- **评估数据集缓存**（MMEB_Test_Instruct - hub目录）：约 4.9MB
  - 包含：评估数据集的元数据文件
- **评估数据集实际数据缓存**（MMEB_Test_Instruct - datasets目录）：约 1.4GB
  - 包含：所有评估数据集（MSCOCO_i2t、MSCOCO_t2i 等）的实际数据
- **数据集元数据缓存**：约几MB
  - 包含：数据集的锁文件和缓存信息

**总计：约 8.15GB**

## 重要说明

1. **Processor 和 Tokenizer**：Qwen3-VL 的 processor 和 tokenizer 文件已经包含在模型缓存中（`models--Qwen--Qwen3-VL-4B-Instruct`），不需要单独拷贝。

2. **符号链接结构**：Hugging Face 使用符号链接来组织文件，`snapshots/` 目录中的文件是指向 `blobs/` 目录的符号链接。拷贝时必须保留这种结构，否则模型无法正确加载。

3. **数据集子集**：虽然训练脚本只使用了 `MSCOCO_t2i` 和 `MSCOCO_i2t` 两个子集，但建议拷贝整个 `datasets--TIGER-Lab--MMEB-train` 目录，以确保完整性。

4. **验证完整性**：拷贝完成后，运行 `check_hf_cache.sh` 脚本验证文件是否完整。
