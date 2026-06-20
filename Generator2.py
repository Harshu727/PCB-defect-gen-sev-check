"""
PCB Defect Heatmap Generator
-----------------------------
Reads all *_mask.png files from OUTPUT folder,
generates a heatmap overlay on the original PCB image.

Usage: python pcb_heatmap.py
"""

import cv2
import numpy as np
import os
import glob

# ─── PATHS ───────────────────────────────────────────────────────────────────────

INPUT_IMAGE  = r"D:\PCB DEFECT GENERATOR\INPUT\PCBIMG.png"
MASKS_FOLDER = r"D:\PCB DEFECT GENERATOR\OUTPUT"
OUTPUT_PATH  = r"D:\PCB DEFECT GENERATOR\OUTPUT2"

# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load original PCB image
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        print(f"[ERROR] Cannot load image: {INPUT_IMAGE}")
        return
    h, w = image.shape[:2]
    print(f"[INFO] Loaded PCB image: {w}x{h}")

    # Load all mask files from OUTPUT folder
    mask_files = sorted(glob.glob(os.path.join(MASKS_FOLDER, "*_mask.png")))
    if not mask_files:
        print(f"[ERROR] No *_mask.png files found in: {MASKS_FOLDER}")
        return
    print(f"[INFO] Found {len(mask_files)} mask files")

    # Accumulate all masks into a single heatmap
    heatmap_acc = np.zeros((h, w), dtype=np.float32)

    for mf in mask_files:
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [SKIP] Cannot read: {mf}")
            continue
        # Resize mask to match image if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
        mask_f = mask.astype(np.float32) / 255.0
        heatmap_acc += mask_f
        print(f"  [OK] {os.path.basename(mf)}")

    # ── Per-defect individual heatmaps ──────────────────────────────────────────
    defect_types = ["scratch", "missing_copper", "short_circuit"]

    for dtype in defect_types:
        type_masks = [mf for mf in mask_files if dtype in os.path.basename(mf)]
        if not type_masks:
            continue

        acc = np.zeros((h, w), dtype=np.float32)
        for mf in type_masks:
            mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            acc += mask.astype(np.float32) / 255.0

        heatmap_img = generate_heatmap_overlay(image, acc, dtype)
        out_name = f"heatmap_{dtype}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_PATH, out_name), heatmap_img)
        print(f"  [SAVED] {out_name}")

    # ── Combined heatmap (all defects) ──────────────────────────────────────────
    combined = generate_heatmap_overlay(image, heatmap_acc, "All Defects Combined")
    cv2.imwrite(os.path.join(OUTPUT_PATH, "heatmap_combined.jpg"), combined)
    print(f"  [SAVED] heatmap_combined.jpg")

    # ── Side-by-side panel: Original | Combined Heatmap ─────────────────────────
    panel = make_side_panel(image, heatmap_acc)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "heatmap_panel.jpg"), panel)
    print(f"  [SAVED] heatmap_panel.jpg")

    print(f"\n[DONE] All heatmaps saved to: {OUTPUT_PATH}")


def generate_heatmap_overlay(image, acc, label=""):
    """
    Normalize accumulation map → apply JET colormap → overlay on original image.
    """
    h, w = image.shape[:2]

    # Normalize to 0-255
    if acc.max() > 0:
        norm = (acc / acc.max() * 255).astype(np.uint8)
    else:
        norm = acc.astype(np.uint8)

    # Apply JET colormap (blue=low, green=mid, red=high frequency)
    heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # Blend heatmap with original (alpha blend)
    alpha = 0.55
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    # Add colorbar on right side
    bar_w = 30
    bar_h = h
    colorbar = np.zeros((bar_h, bar_w), dtype=np.uint8)
    for row in range(bar_h):
        colorbar[row, :] = int(255 * (1 - row / bar_h))
    colorbar_color = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET)

    # Add HIGH/LOW labels on colorbar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colorbar_color, "HI", (2, 18),       font, 0.4, (255,255,255), 1)
    cv2.putText(colorbar_color, "LO", (2, bar_h-8),  font, 0.4, (255,255,255), 1)

    # Combine overlay + colorbar
    result = np.hstack([overlay, colorbar_color])

    # Add title bar at top
    title_h = 36
    title_bar = np.ones((title_h, result.shape[1], 3), dtype=np.uint8) * 30
    title = f"Defect Frequency Heatmap — {label.replace('_', ' ').title()}"
    cv2.putText(title_bar, title, (8, 24), font, 0.52, (220, 220, 220), 1)

    final = np.vstack([title_bar, result])
    return final


def make_side_panel(image, acc):
    """
    3-panel: Original | Heatmap Only | Overlay
    """
    h, w = image.shape[:2]
    tw   = 300
    th   = int(h * tw / w)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Panel 1: original
    orig_r = cv2.resize(image, (tw, th))

    # Panel 2: pure heatmap (no overlay)
    if acc.max() > 0:
        norm = (acc / acc.max() * 255).astype(np.uint8)
    else:
        norm = acc.astype(np.uint8)
    norm_r = cv2.resize(norm, (tw, th))
    heatmap_only = cv2.applyColorMap(norm_r, cv2.COLORMAP_JET)

    # Panel 3: blended overlay
    heatmap_full = cv2.applyColorMap(cv2.resize(norm, (w, h)), cv2.COLORMAP_JET)
    overlay_full = cv2.addWeighted(image, 0.45, heatmap_full, 0.55, 0)
    overlay_r    = cv2.resize(overlay_full, (tw, th))

    # Build panel
    lh    = 44
    panel = np.ones((th + lh, tw * 3 + 20, 3), dtype=np.uint8) * 25

    panel[lh:, 0:tw]             = orig_r
    panel[lh:, tw+10:tw*2+10]    = heatmap_only
    panel[lh:, tw*2+20:tw*3+20]  = overlay_r

    cv2.putText(panel, "Original PCB",     (8,      26), font, 0.52, (200,200,200), 1)
    cv2.putText(panel, "Heatmap (Raw)",    (tw+14,  26), font, 0.52, (100,200,255), 1)
    cv2.putText(panel, "Heatmap Overlay",  (tw*2+24,26), font, 0.52, (100,255,150), 1)

    return panel


if __name__ == "__main__":
    main()