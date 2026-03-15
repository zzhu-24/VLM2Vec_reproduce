# Personal Research Repository Based on VLM2Vec

This repository is a **extension and experimental fork** of [TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). In this repo, we tried various strategies to accelerate the training of VLM2Vec.

The original VLM2Vec project provides a unified multimodal embedding framework and the MMEB benchmark series; on top of that, this repo adds custom training scripts, configurations, and experimental workflows.

> **Important Disclaimer**
> - All academic contributions (models, methods, benchmark design, etc.) belong to the original VLM2Vec authors.
> - This repository is purely a personal experimental and development fork and does **not** represent any official stance of the original authors.

---

## Environment & Quick Start

The following is a **non-strict** example of how you might set up an environment:

```bash
conda create -n vlm2vec python=3.10 -y
conda activate vlm2vec

pip install -r requirements.txt  # if present
```

Actual dependencies and versions may vary depending on your experiments; please adjust according to errors and your specific needs.

---

## Training & Evaluation (Illustrative)

- **Training example**: check scripts under `experiments/public/train/`, for example:

```bash
bash experiments/public/train/xxx/train_xxx.sh
```

- **Evaluation / retrieval example**: check scripts under `experiments/public/eval/` or `experiments/public/exps/*/retrieval`:

```bash
bash experiments/public/eval/xxx/eval_xxx.sh
```

> Since this repo is tailored to personal experiments, script names and paths may be informal. Please inspect the actual directory structure and adapt as needed.

---

## Relation to Upstream & Acknowledgements

- **Origin**: This repo evolves from the upstream `TIGER-AI-Lab/VLM2Vec` codebase and contains most or all of its core implementation.
- **Modifications**: Mainly around training scripts, experiment configurations, and some data-processing flows; the core method described in the original papers is not changed.
- **Acknowledgements**: Many thanks to the VLM2Vec authors for releasing high-quality code and data, which makes this line of research possible.

If you use this codebase for research, please **always cite the original VLM2Vec papers** (see below).

---

## Citation (Original VLM2Vec Work)

```text
@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}

@article{meng2025vlm2vecv2,
  title={VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents},
  author={Rui Meng and Ziyan Jiang and Ye Liu and Mingyi Su and Xinyi Yang and Yuepeng Fu and Can Qin and Zeyuan Chen and Ran Xu and Caiming Xiong and Yingbo Zhou and Wenhu Chen and Semih Yavuz},
  journal={arXiv preprint arXiv:2507.04590},
  year={2025}
}
```

If you build new methods or downstream applications on top of this fork, please add your own citations or project references accordingly.
