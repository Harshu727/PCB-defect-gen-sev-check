# Topology-Guided Synthetic PCB Defect Generation with Severity-Aware Post-Fabrication Inspection

A rule-based, GPU-free pipeline that generates physically realistic synthetic PCB defects — and scores how dangerous each one actually is.

Most synthetic defect generators scatter scratches and missing-copper patches anywhere on the board, including bare substrate where such defects could never occur in real fabrication. This project fixes that by detecting copper trace regions first, then constraining every injected defect to land only where it's physically plausible. On top of that, it adds a severity scoring layer (inspired by the IPC-A-610 industry standard) so each defect gets a risk rating and an actionable recommendation — not just a yes/no label.

## How it works — 4 stages

**1. Topology-Aware Defect Generation**
Detects copper trace/pad regions via brightness thresholding + contour analysis, then injects one of three defect types — scratch, missing copper, short circuit — exclusively onto those regions. Each defect comes with a matching binary mask, ready for supervised ML training without any manual labeling.

**2. Heatmap Visualization**
Aggregates defect masks across all generated samples into frequency heatmaps, showing where defects cluster on the board and visually confirming the topology-aware placement is working as intended.

**3. Defect Detection & Severity Scoring**
Compares the clean and defected images via absolute difference analysis, then computes a two-factor severity score:
- **Defect area ratio** — 60% weight
- **Copper region overlap** — 40% weight

Each defect is classified as **Minor**, **Moderate**, or **Critical**.

**4. Risk Awareness & Recommendation**
Applies defect-type risk multipliers on top of the severity score and outputs a post-fabrication recommendation: **Proceed to Test**, **Inspect First**, **Rework**, or **Discard**.

## Severity scoring model

```
Severity Score = Area Score (0–60) + Copper Overlap Score (0–40)

Area Score    = min(defect_area / board_area, 0.05) / 0.05 × 60
Overlap Score = (defect_pixels_on_copper / total_defect_pixels) × 40
```

| Score range | Class |
|---|---|
| 0 – 24 | Minor |
| 25 – 54 | Moderate |
| 55 – 100 | Critical |

## Risk multipliers (Stage 4)

| Defect type | Multiplier | Why |
|---|---|---|
| Scratch | 1.0× | Surface damage; impact depends on whether it penetrates the copper layer |
| Missing Copper | 1.5× | Direct material loss — risks an open circuit |
| Short Circuit | 2.0× | Conductive bridge — electrical fault risk regardless of size |

## Project structure

```
INPUT/                          # Source clean PCB image(s)

OUTPUT/
  00_trace_map.jpg              # Copper trace detection overlay
  sample_XX_defected.jpg        # Defected PCB image
  sample_XX_mask.png            # Binary ground-truth mask

OUTPUT2/
  heatmap_panel_[type].jpg      # Per-defect-type heatmap (3-panel)
  heatmap_panel_combined.jpg    # Combined heatmap across all types

OUTPUT3/
  sample_XX_panel.jpg           # 4-panel detection + severity card
  severity_report.csv           # Structured severity scores

OUTPUT3/AWARENESS/
  sample_XX_gauge.jpg           # Risk gauge (LOW/ELEVATED/HIGH)
  sample_XX_awareness_card.jpg  # Recommendation card
  board_risk_summary.jpg        # Board-level risk dashboard

Generator.py        # Stage 1 — topology-aware defect generation
Generator2.py        # Stage 2 — heatmap visualization
severity.py          # Stage 3 — defect detection & severity scoring
pcb_awareness.py     # Stage 4 — risk classification & recommendations


## Requirements

```
Python 3.10
opencv-python (cv2) 4.8.0
numpy 1.24.0
matplotlib 3.7.1
```

No GPU or deep learning framework needed at any stage.

```bash
pip install opencv-python numpy matplotlib
```

## Usage

```bash
python Generator.py        # generates defects + masks into OUTPUT/
python Generator2.py       # generates heatmaps into OUTPUT2/
python severity.py         # detects defects + scores severity into OUTPUT3/
python pcb_awareness.py    # generates risk gauges + recommendations
```

## Results

Tested on a 718×402 PCB image (blue substrate, cyan-green copper traces). Copper detection identified 113 contour regions covering ~27.5% of the board area. Six defect samples (two per type) were generated with **zero topology violations** — every defect landed on a copper region.

| Sample | Type | Area % | Score | Class | Weighted Risk | Recommendation |
|---|---|---|---|---|---|---|
| 01 | Scratch | 0.094% | 5.4 | Minor | 5.4 — Low | Proceed to Test |
| 02 | Missing Copper | 0.285% | 23.0 | Minor | 34.5 — Elevated | Inspect First |
| 03 | Short Circuit | 0.286% | 19.1 | Minor | 38.2 — Elevated | Inspect First |
| 04 | Scratch | 0.055% | 7.0 | Minor | 7.0 — Low | Proceed to Test |
| 05 | Missing Copper | 0.154% | 15.0 | Minor | 22.5 — Elevated | Inspect First |
| 06 | Short Circuit | 0.305% | 8.4 | Minor | 16.8 — Elevated | Inspect First |

## Future work

- Extend to all 6 DeepPCB defect categories (missing hole, mouse bite, spur, spurious copper)
- Train a YOLOv8/RT-DETR model on the generated dataset for reference-free detection
- GAN-based realism enhancement, conditioned on the topology-aware placement
- Expand to multiple board types (SMD, multilayer) for broader generalization

## Authors
- Harshithaa B K
  <br>
- Arvin Lourdu V
  


