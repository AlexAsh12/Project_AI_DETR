# DETR: End-to-End Object Detection from Scratch

Implementation of **DETR (DEtection TRansformer)** from scratch in PyTorch, trained on **PASCAL VOC2007** (20 object classes). Built as Assignment 2 for the CSE4007 Artificial Intelligence course (Hanyang University), with results submitted to a Kaggle leaderboard.

## Overview

DETR reformulates object detection as a direct set prediction problem, removing hand-designed components like anchor generation and non-maximum suppression. This project implements the full pipeline from scratch:

1. **Architecture** — CNN backbone + Transformer encoder-decoder with learned object queries
2. **Loss functions** — Hungarian matching (via `scipy.optimize.linear_sum_assignment`) combining classification, L1, and GIoU costs
3. **Training & optimization** — end-to-end training and tuning for detection performance on VOC2007

## Repository Contents

| File | Description |
|---|---|
| `assignment2_DETR.ipynb` | Main notebook — architecture, training loop, inference, evaluation, Kaggle submission generation |
| `assignment2_detr.py` | Script export of the notebook |
| `Project_AI_DETR.pdf` | Written report covering architecture, methodology, and results |

## Architecture Notes

- Backbone: ResNet-50 (HuggingFace pretrained checkpoint used as a reference/initialization point)
- Positional embeddings added to queries and keys, **not** to values, in attention layers
- Post-norm Transformer encoder layers
- Decoder iterates with `enumerate(self.layers)` to support auxiliary losses at each decoder layer
- Bipartite matching cost: classification + L1 (box coordinates) + GIoU

## Training Configuration

- Optimizer: AdamW with differential learning rates (backbone: `1e-5`, transformer: `1e-4`)
- LR schedule: StepLR (`step_size=20`, `gamma=0.1`)
- Gradient clipping: `max_norm=0.1`
- Mixed-precision training (AMP)
- Compute: Google Colab, Tesla T4 GPU (16GB)

## Reproducing / Running

1. Open `assignment2_DETR.ipynb` in Colab (T4 GPU runtime recommended)
2. Mount Google Drive for checkpoint persistence across sessions
3. Adjust key constants as needed: `CONF_THRESHOLD`, `IOU_THRESHOLD`, `NUM_QUERIES`, `D_MODEL`, `BACKBONE_MODEL_TYPE`
4. Run training, then inference/evaluation cells to generate the Kaggle submission CSV

**Note on evaluation:** the default confidence threshold can be too high and yield zero predictions (mAP = 0.0). Start from a low threshold (e.g. `0.01`) when evaluating.

## Known Limitations / Practical Notes

- Full-resolution inputs (800×1333, as in the original paper) cause OOM on a T4 due to O(S²) attention memory scaling — inputs were reduced to 320×512 to fit
- A full Colab runtime restart is required after an OOM error to reliably free GPU memory
- Checkpoints are saved as nested dicts (`epoch`, `model_state_dict`, `optimizer_state_dict`, `loss`) — load via `checkpoint['model_state_dict']`, not the checkpoint object directly
