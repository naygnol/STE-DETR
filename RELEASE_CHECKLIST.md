# Reproduce checklist (for GitHub release)

Use this file before `git push`.

## Include

- [x] Source code under `mmdet/`
- [x] Configs under `configs/`
- [x] Training / testing tools under `tools/`
- [x] `README.md` / `README_zh-CN.md`
- [x] `LICENSE`, `NOTICE`, `CITATION.cff`
- [x] `requirements*` / `setup.py`

## Exclude (already in `.gitignore`)

- [x] `save/` (local training logs)
- [x] `work_dirs/`
- [x] `*.pth` / `*.pt` / `*.ckpt`
- [x] `data/` (COCO images / annotations)
- [x] `.idea/`, `__pycache__/`, `*.log`

## Before release

1. GitHub username is `naygnol`; repository name is `STE-DETR`
   (https://github.com/naygnol/STE-DETR). Paper method name is STE-DETR.
2. Fill author names in Citation / README bibtex before camera-ready.
3. (Optional) Upload checkpoints and add download links in README Results.
4. For double-blind review: hide personal identity until acceptance.
