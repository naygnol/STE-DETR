# STE-DETR

**STE-DETR: A Separate Task Expert Detector for Object Detection** 官方实验代码。

仓库：https://github.com/naygnol/STE-DETR

[English](README.md) | 简体中文

## 与论文对应关系

| 论文模块 | 代码位置 | 默认超参 |
|---|---|---|
| STED | `detr_layers.py` 双分支 cross-attn | — |
| 融合系数 μ | `dino_layers.py` | 0.6 |
| LFOM | `base_detr.py` | λ=1.75 |
| CLS-IoU | `detr_head.py` 中 `IA_BCE_loss` | α=0.25 |

## 复现主实验（COCO AP 50.8）

环境：2×RTX 4090，PyTorch 1.13 + CUDA 11.7，12 epoch，lr=1e-4，每卡 batch=2。

```bash
conda create -n ste_detr python=3.8 -y && conda activate ste_detr
# 安装 PyTorch / mmcv / mmengine 见英文 README

git clone https://github.com/naygnol/STE-DETR.git
cd STE-DETR && pip install -e . && pip install -r requirements.txt

# 数据放到 data/coco/

bash tools/dist_train.sh \
  configs/dino/dino-4scale_r50_8xb2-12e_coco.py 2 \
  --work-dir work_dirs/ste_detr_dino_r50_12e

python tools/test.py \
  configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
  work_dirs/ste_detr_dino_r50_12e/epoch_12.pth
```

目标指标：AP **50.8** / AP<sub>S</sub> **33.3**（与论文主表一致）。

VisDrone 需自行转为 COCO 格式，输入尺度 640×640，`num_classes=10`。

## 许可

Apache-2.0（基于 MMDetection）。
