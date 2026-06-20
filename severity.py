"""
PCB Defect Detector + Severity Scorer
--------------------------------------
Reads all *_defected.jpg files from OUTPUT folder,
detects defects by comparing with original clean image,
computes severity score and class, saves annotated results + CSV table.

Output saved to: D:\PCB DEFECT GENERATOR\OUTPUT3

Usage: python pcb_detect_severity.py
"""

import cv2
import numpy as np
import os
import glob
import csv

# ─── PATHS ───────────────────────────────────────────────────────────────────────

INPUT_IMAGE    = r"D:\PCB DEFECT GENERATOR\INPUT\PCBIMG.png"
DEFECTS_FOLDER = r"D:\PCB DEFECT GENERATOR\OUTPUT"
OUTPUT_PATH    = r"D:\PCB DEFECT GENERATOR\OUTPUT3"

# ─── Severity Scoring ────────────────────────────────────────────────────────────

def compute_severity(diff_mask, original):
    """
    Severity score 0-100:
      60% — defect area as % of total image
      40% — defect overlap with bright (copper) regions
    """
    h, w        = diff_mask.shape
    total_px    = h * w
    defect_px   = np.count_nonzero(diff_mask)
    area_ratio  = defect_px / total_px
    area_pct    = round(area_ratio * 100, 3)

    # Scale: 5%+ defect area = max area score
    area_score  = min(area_ratio / 0.05, 1.0) * 60

    # Copper overlap: defect on bright region = more critical
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, copper = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    overlap   = cv2.bitwise_and(diff_mask, copper)
    overlap_r = np.count_nonzero(overlap) / max(defect_px, 1)
    copper_score = overlap_r * 40

    score = round(area_score + copper_score, 1)
    score = min(score, 100.0)

    if score < 25:
        label = "MINOR"
        color = (0, 180, 0)       # green
    elif score < 55:
        label = "MODERATE"
        color = (0, 165, 255)     # orange
    else:
        label = "CRITICAL"
        color = (0, 0, 220)       # red

    return score, label, color, area_pct


# ─── Defect Detection ────────────────────────────────────────────────────────────

def detect_defect_region(original, defected):
    """
    Find defect region by absolute difference between original and defected image.
    Returns binary mask of defect area.
    """
    diff = cv2.absdiff(original, defected)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold difference — pixels that changed significantly
    _, mask = cv2.threshold(gray_diff, 18, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    return mask


def get_defect_contours(mask):
    """Get individual defect region contours from mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter noise
    return [c for c in contours if cv2.contourArea(c) > 15]


# ─── Annotated Output ────────────────────────────────────────────────────────────

def annotate_image(defected, contours, score, label, color, area_pct, defect_name):
    """Draw bounding boxes, contours, and severity label on defected image."""
    result = defected.copy()

    for cnt in contours:
        # Draw contour outline
        cv2.drawContours(result, [cnt], -1, color, 2)
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)
        # Small label near bounding box
        cv2.putText(result, label, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Bottom info bar
    bar_h = 40
    ih, iw = result.shape[:2]
    bar = np.ones((bar_h, iw, 3), dtype=np.uint8) * 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, f"Defect: {defect_name.replace('_',' ').title()}",
                (8, 16), font, 0.45, (200, 200, 200), 1)
    cv2.putText(bar, f"Severity: {score}/100  |  Area: {area_pct}%  |  Class: {label}",
                (8, 32), font, 0.45, color, 1)

    return np.vstack([result, bar])


def make_detection_panel(original, defected_ann, mask, score, label, color, defect_name, area_pct):
    """4-panel: Original | Detected+Annotated | Diff Mask | Severity Card"""
    h, w = original.shape[:2]
    tw   = 280
    th   = int(h * tw / w)
    font = cv2.FONT_HERSHEY_SIMPLEX

    orig_r = cv2.resize(original,     (tw, th))
    ann_r  = cv2.resize(defected_ann[:h], (tw, th))   # crop bar for resize
    msk_r  = cv2.cvtColor(cv2.resize(mask, (tw, th)), cv2.COLOR_GRAY2BGR)

    # Severity card
    card = np.ones((th, tw, 3), dtype=np.uint8) * 18
    sev_color = {"MINOR": (0,180,0), "MODERATE": (0,165,255), "CRITICAL": (0,0,220)}[label]

    # Large score
    cv2.putText(card, f"{score}",        (tw//2-35, th//2-20), font, 2.2,  sev_color, 3)
    cv2.putText(card, "/ 100",           (tw//2+20, th//2+10), font, 0.55, (180,180,180), 1)
    cv2.putText(card, label,             (tw//2-30, th//2+38), font, 0.75, sev_color, 2)
    cv2.putText(card, f"Area: {area_pct}%", (10, th-40),       font, 0.42, (160,160,160), 1)
    cv2.putText(card, defect_name.replace('_',' ').title(), (10, th-22), font, 0.42, (160,160,160), 1)
    cv2.putText(card, "SEVERITY SCORE",  (10, 22),             font, 0.42, (100,100,100), 1)
    # Border
    cv2.rectangle(card, (2,2), (tw-2, th-2), sev_color, 2)

    lh    = 36
    panel = np.ones((th + lh, tw * 4 + 30, 3), dtype=np.uint8) * 15

    panel[lh:, 0:tw]              = orig_r
    panel[lh:, tw+10:tw*2+10]     = ann_r
    panel[lh:, tw*2+20:tw*3+20]   = msk_r
    panel[lh:, tw*3+30:tw*4+30]   = card

    cv2.putText(panel, "Original PCB",       (8,      24), font, 0.50, (180,180,180), 1)
    cv2.putText(panel, "Defect Detected",    (tw+14,  24), font, 0.50, (100,200,255), 1)
    cv2.putText(panel, "Difference Mask",    (tw*2+24,24), font, 0.50, (100,255,150), 1)
    cv2.putText(panel, "Severity Score",     (tw*3+34,24), font, 0.50, (255,200,100), 1)

    return panel


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    original = cv2.imread(INPUT_IMAGE)
    if original is None:
        print(f"[ERROR] Cannot load: {INPUT_IMAGE}")
        return
    h, w = original.shape[:2]
    print(f"[INFO] Original loaded: {w}x{h}")

    defect_files = sorted(glob.glob(os.path.join(DEFECTS_FOLDER, "*_defected.jpg")))
    if not defect_files:
        print(f"[ERROR] No *_defected.jpg files in: {DEFECTS_FOLDER}")
        return
    print(f"[INFO] Found {len(defect_files)} defected images\n")

    # CSV log
    csv_rows = [["Sample", "Defect Type", "Severity Score", "Defect Area %", "Class", "Defect Regions Found"]]

    for df in defect_files:
        fname    = os.path.basename(df)
        # Extract defect type from filename e.g. sample_01_scratch_defected.jpg
        parts    = fname.replace("_defected.jpg", "").split("_")
        sample   = f"{parts[0]}_{parts[1]}"
        dtype    = "_".join(parts[2:]) if len(parts) > 2 else "unknown"

        defected = cv2.imread(df)
        if defected is None:
            print(f"  [SKIP] Cannot read: {fname}")
            continue

        # Resize defected to match original if needed
        if defected.shape[:2] != (h, w):
            defected = cv2.resize(defected, (w, h))

        # Detect defect region
        diff_mask = detect_defect_region(original, defected)
        contours  = get_defect_contours(diff_mask)

        # Score
        score, label, color, area_pct = compute_severity(diff_mask, original)

        # Annotated image
        ann = annotate_image(defected, contours, score, label, color, area_pct, dtype)

        # Detection panel
        panel = make_detection_panel(original, ann, diff_mask, score, label, color, dtype, area_pct)

        # Save
        base = fname.replace("_defected.jpg", "")
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{base}_detected.jpg"),    ann)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{base}_panel.jpg"),       panel)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{base}_diffmask.png"),    diff_mask)

        csv_rows.append([sample, dtype, score, area_pct, label, len(contours)])

        print(f"  [{sample}] {dtype:20s}  score={score:5.1f}  area={area_pct}%  {label}  regions={len(contours)}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_PATH, "severity_report.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\n[DONE] Outputs saved to: {OUTPUT_PATH}")
    print(f"[DONE] Severity table:   severity_report.csv")
    print(f"\nFor your report:")
    print(f"  sample_XX_*_panel.jpg   -> Results section (4-panel detection view)")
    print(f"  severity_report.csv     -> paste as Table in Results section")

    # Print table to console
    print(f"\n{'='*75}")
    print(f"{'Sample':<12} {'Defect Type':<22} {'Score':>6} {'Area%':>7} {'Class':>10} {'Regions':>8}")
    print(f"{'='*75}")
    for row in csv_rows[1:]:
        print(f"{row[0]:<12} {row[1]:<22} {row[2]:>6} {row[3]:>7} {row[4]:>10} {row[5]:>8}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()