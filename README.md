# STE-DETR

Official experimental code for **STE-DETR: A Separate Task Expert Detector for Object Detection**.

Paper PDF: `STE_DETR__A_Separate_Task_Expert_Detector_for_Object_Detection.pdf`  
Code: https://github.com/naygnol/STE-DETR

Built upon [MMDetection](https://github.com/open-mmlab/mmdetection) (v3.0.0) + [DINO](https://arxiv.org/abs/2203.03605).

English | [简体中文](README_zh-CN.md)

---

## Method ↔ Code Mapping

| Paper module | Code location | Default hyper-parameter |
|---|---|---|
| **STED** (dual-path task experts) | `mmdet/models/layers/transformer/detr_layers.py` (`cross_attn_cls` / `cross_attn_box`) | — |
| Fusion factor **μ** | `mmdet/models/layers/transformer/dino_layers.py` | `μ = 0.6` |
| **LFOM** (C-LFOM / B-LFOM) | `mmdet/models/detectors/base_detr.py` (`rego_cls` / `rego_box`) | enlarge ratio **λ = 1.75** |
| Glimpse transformer | `mmdet/models/Add/transformer.py` | — |
| **CLS-IoU loss** (`IA_BCE_loss`, σ/α=0.25) | `mmdet/models/dense_heads/detr_head.py`, used in `dino_head.py` | `α = 0.25` |

---

## Environment (same as paper)

Paper setting:

- 2 × NVIDIA RTX 4090
- PyTorch 1.13.0 + CUDA 11.7
- MMDetection 3.0 / MMEngine / MMCV 2.0.x
- AdamW, lr = 1e-4, weight decay = 1e-4
- 12 epochs, LR decay at epoch 11
- Per-GPU batch size = 2 (total batch = 4)

```bash
conda create -n ste_detr python=3.8 -y
conda activate ste_detr

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

pip install -U openmim
mim install "mmcv>=2.0.0rc4,<2.1.0"
pip install "mmengine>=0.7.1,<1.0.0"

git clone https://github.com/naygnol/STE-DETR.git
cd STE-DETR
pip install -e .
pip install -r requirements.txt
```

---

## Datasets

### MS COCO 2017 (main results)

```text
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

Default `data_root = 'data/coco/'`.

### VisDrone 2019 (small-object validation)

Paper also reports VisDrone results (input 640×640). Convert VisDrone to COCO format, then create a config based on `configs/dino/dino-4scale_r50_8xb2-12e_coco.py` with:

- `data_root` pointing to VisDrone
- `num_classes = 10`
- train / test image scale `(640, 640)`

A dedicated VisDrone config will be added if needed; reviewers can reproduce the **main COCO Table** with the commands below.

---

## Reproduce main COCO experiment (AP 50.8)

### Train (2 GPUs, matches paper)

```bash
bash tools/dist_train.sh \
  configs/dino/dino-4scale_r50_8xb2-12e_coco.py 2 \
  --work-dir work_dirs/ste_detr_dino_r50_12e
```

Expected wall time: ~2–3 days on 2×4090.

### Test

```bash
python tools/test.py \
  configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
  work_dirs/ste_detr_dino_r50_12e/epoch_12.pth \
  --work-dir work_dirs/ste_detr_dino_r50_12e
```

### Single-GPU fallback

```bash
python tools/train.py \
  configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
  --work-dir work_dirs/ste_detr_dino_r50_12e \
  --auto-scale-lr
```

---

## Results (should match paper Table)

COCO `val2017`, ResNet-50, 12 epochs:

| Model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|---|---:|---:|---:|---:|---:|---:|
| DINO-DETR (baseline) | 49.0 | 66.6 | 53.5 | 32.0 | 52.3 | 63.0 |
| **STE-DETR (Ours)** | **50.8** | **68.4** | **55.1** | **33.3** | **54.2** | **66.2** |

VisDrone 2019 (paper): Baseline 31.8 AP → Ours **39.6** AP.

> Pretrained weights are large; upload to Google Drive / Hugging Face and put the link here for reviewers.

---

## Ablation switches (for reviewers)

| Ablation | How to approximate in code |
|---|---|
| w/o LFOM | set `self.use_rego = False` in `base_detr.py` |
| w/o STED | use unmodified DINO decoder (single-path); see MMDetection DINO baseline |
| λ search | change `rego_scales_*` in `base_detr.py` (`1.5 / 1.75 / 2.0`) |
| μ search | change `mu` in `dino_layers.py` (`0.4 / 0.5 / 0.6 / 0.7`) |
| σ in CLS-IoU | `IA_BCE_loss(..., alpha=0.25, ...)` in `dino_head.py` |

---

## Project layout

```text
mmdet/models/
├── detectors/base_detr.py              # LFOM
├── dense_heads/{detr,dino}_head.py     # CLS-IoU (IA_BCE_loss)
├── layers/transformer/
│   ├── detr_layers.py                  # STED dual experts
│   └── dino_layers.py                  # μ fusion
└── Add/transformer.py                  # LFOM glimpse transformer
configs/dino/                           # training configs
tools/train.py / tools/test.py
```

---

## License & Acknowledgement

Apache-2.0 (same as MMDetection). Thanks to MMDetection and DINO.

---

## Citation

```bibtex
@article{stedetr,
  title={STE-DETR: A Separate Task Expert Detector for Object Detection},
  author={Long, Yan and Xu, Chenjun and Yang, Xiaobao and Sun, Wei and Han, Haozhe and Zhang, Weiwei and Chai, Ruiyang},
  journal={Under Review},
  year={2026}
}
```
