"""
PCB Realistic Defect Generator — Final Version
------------------------------------------------
Novel feature: Topology-Aware Defect Placement
  - Detects copper trace regions automatically
  - Injects defects ONLY on detected trace areas (not random substrate)
  - Colors sampled from actual PCB pixels for realism
  - Gaussian alpha blending for soft natural edges

Defect Types:
  1. Scratch        — jagged dark line with feathered edges + metal sheen
  2. Missing Copper — irregular dark patch with rough edges + oxidation ring
  3. Short Circuit  — copper-colored blob bridge with texture + oxidation ring

Usage:
    Just run:  python pcb_defect_final.py
"""

import cv2
import numpy as np
import os
import random

# ─── PATHS ───────────────────────────────────────────────────────────────────────

INPUT_PATH  = r"D:\PCB DEFECT GENERATOR\INPUT\PCBIMG.png"
OUTPUT_PATH = r"D:\PCB DEFECT GENERATOR\OUTPUT"
NUM_SAMPLES = 6        # how many defect images to generate
RANDOM_SEED = 42       # change this for different results each run

# ─── Copper Trace Detection ──────────────────────────────────────────────────────

def detect_traces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    trace_contours = [c for c in contours if 40 < cv2.contourArea(c) < 5000]
    return trace_contours


def pick_point(contours, w, h):
    """Pick a random point guaranteed to lie inside a trace contour."""
    chosen = random.choice(contours)
    x, y, cw, ch = cv2.boundingRect(chosen)
    for _ in range(60):
        px = min(random.randint(x, x + cw), w - 1)
        py = min(random.randint(y, y + ch), h - 1)
        if cv2.pointPolygonTest(chosen, (float(px), float(py)), False) >= 0:
            return px, py
    return min(x + cw // 2, w - 1), min(y + ch // 2, h - 1)


# ─── Defect 1: Scratch ───────────────────────────────────────────────────────────

def make_scratch(img, trace_contours):
    h, w   = img.shape[:2]
    result = img.copy()
    mask   = np.zeros((h, w), dtype=np.uint8)

    cx, cy  = pick_point(trace_contours, w, h)
    angle   = random.uniform(0, np.pi)
    length  = random.randint(55, 110)
    n_seg   = random.randint(10, 18)
    seg_len = length / n_seg

    local_b = int(img[cy, cx][0])
    local_g = int(img[cy, cx][1])
    local_r = int(img[cy, cx][2])
    shine   = (min(255, local_b + 40), min(255, local_g + 40), min(255, local_r + 40))

    x, y = float(cx), float(cy)
    for _ in range(n_seg):
        jit = random.uniform(-0.35, 0.35)
        a   = angle + jit
        nx  = np.clip(x + seg_len * np.cos(a), 0, w - 1)
        ny  = np.clip(y + seg_len * np.sin(a), 0, h - 1)
        ix, iy, inx, iny = int(x), int(y), int(nx), int(ny)

        cv2.line(result, (ix, iy), (inx, iny), (10, 10, 10), 2)   # dark core
        cv2.line(mask,   (ix, iy), (inx, iny), 255, 3)
        cv2.line(result, (ix + 1, iy), (inx + 1, iny), shine, 1)  # metal sheen
        if random.random() > 0.6:
            cv2.line(result, (ix, iy + 1), (inx, iny + 1), (15, 15, 15), 1)
        x, y = nx, ny

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return result, mask


# ─── Defect 2: Missing Copper ────────────────────────────────────────────────────

def make_missing_copper(img, trace_contours):
    h, w   = img.shape[:2]
    result = img.copy()
    mask   = np.zeros((h, w), dtype=np.uint8)

    cx, cy = pick_point(trace_contours, w, h)

    dark_y = min(cy + 50, h - 1)
    dark_x = min(cx + 50, w - 1)
    region    = img[max(0, dark_y - 5):dark_y + 5, max(0, dark_x - 5):dark_x + 5]
    sub_color = tuple(max(0, int(c) - 20) for c in region.mean(axis=(0, 1)))

    rx  = random.randint(9, 20)
    ry  = random.randint(6, 15)
    ang = random.randint(0, 180)

    layer = result.copy()

    cv2.ellipse(layer, (cx, cy), (rx, ry), ang, 0, 360, sub_color, -1)
    cv2.ellipse(mask,  (cx, cy), (rx, ry), ang, 0, 360, 255, -1)

    for _ in range(12):
        oa = random.uniform(0, 2 * np.pi)
        er = random.uniform(0.65, 1.2)
        ex = int(np.clip(cx + rx * er * np.cos(oa), 0, w - 1))
        ey = int(np.clip(cy + ry * er * np.sin(oa), 0, h - 1))
        rs = random.randint(2, 7)
        blob_color = tuple(max(0, c + random.randint(-15, 10)) for c in sub_color)
        cv2.circle(layer, (ex, ey), rs, blob_color, -1)
        cv2.circle(mask,  (ex, ey), rs, 200, -1)

    alpha = cv2.GaussianBlur(mask, (5, 5), 1).astype(float) / 255.0
    for c in range(3):
        result[:, :, c] = (layer[:, :, c] * alpha + result[:, :, c] * (1 - alpha)).astype(np.uint8)

    halo = tuple(max(0, c - 35) for c in sub_color)
    cv2.ellipse(result, (cx, cy), (rx + 3, ry + 3), ang, 0, 360, halo, 1)

    return result, mask


# ─── Defect 3: Short Circuit ─────────────────────────────────────────────────────

def make_short_circuit(img, trace_contours):
    h, w   = img.shape[:2]
    result = img.copy()
    mask   = np.zeros((h, w), dtype=np.uint8)

    p1x, p1y = pick_point(trace_contours, w, h)
    p2x, p2y = pick_point(trace_contours, w, h)

    dist = np.hypot(p2x - p1x, p2y - p1y)
    if dist > 75:
        p2x = int(p1x + 68 * (p2x - p1x) / dist)
        p2y = int(p1y + 68 * (p2y - p1y) / dist)
    p2x = int(np.clip(p2x, 0, w - 1))
    p2y = int(np.clip(p2y, 0, h - 1))

    s_region = img[max(0, p1y - 4):p1y + 4, max(0, p1x - 4):p1x + 4]
    copper   = tuple(int(c) for c in s_region.mean(axis=(0, 1)))
    bright   = tuple(min(255, int(c) + random.randint(28, 50)) for c in copper)
    dim      = tuple(max(0,   int(c) - 25) for c in copper)

    layer = result.copy()
    thick = random.randint(4, 8)

    cv2.line(layer, (p1x, p1y), (p2x, p2y), bright, thick)
    cv2.line(mask,  (p1x, p1y), (p2x, p2y), 255, thick + 2)

    blob_r = random.randint(7, 13)
    cv2.circle(layer, (p1x, p1y), blob_r, bright, -1)
    cv2.circle(mask,  (p1x, p1y), blob_r, 255, -1)

    for _ in range(6):
        tx = int(np.clip(p1x + random.randint(-blob_r + 2, blob_r - 2), 0, w - 1))
        ty = int(np.clip(p1y + random.randint(-blob_r + 2, blob_r - 2), 0, h - 1))
        tv = tuple(min(255, c + random.randint(-20, 20)) for c in bright)
        cv2.circle(layer, (tx, ty), random.randint(1, 3), tv, -1)

    cv2.circle(layer, (p1x, p1y), blob_r + 2, dim, 1)
    drip_r = random.randint(3, 6)
    cv2.circle(layer, (p2x, p2y), drip_r, bright, -1)
    cv2.circle(mask,  (p2x, p2y), drip_r, 255, -1)

    alpha = cv2.GaussianBlur(mask, (5, 5), 1).astype(float) / 255.0
    for c in range(3):
        result[:, :, c] = (layer[:, :, c] * alpha + result[:, :, c] * (1 - alpha)).astype(np.uint8)
    cv2.circle(result, (p1x, p1y), blob_r + 2, dim, 1)

    return result, mask


# ─── Comparison Panel ────────────────────────────────────────────────────────────

def make_panel(original, defected, mask, defect_name):
    h, w = original.shape[:2]
    tw   = 320
    th   = int(h * tw / w)

    orig_r = cv2.resize(original, (tw, th))
    def_r  = cv2.resize(defected,  (tw, th))
    msk_r  = cv2.cvtColor(cv2.resize(mask, (tw, th)), cv2.COLOR_GRAY2BGR)

    lh    = 36
    panel = np.ones((th + lh, tw * 3 + 24, 3), dtype=np.uint8) * 235

    panel[lh:, 0:tw]                = orig_r
    panel[lh:, tw + 12:tw*2 + 12]   = def_r
    panel[lh:, tw*2 + 24:tw*3 + 24] = msk_r

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "Original PCB",  (8, 22), font, 0.55, (50, 50, 50),  1)
    cv2.putText(panel, f"Defected - {defect_name.replace('_', ' ').title()}", (tw + 14, 22), font, 0.52, (0, 70, 170), 1)
    cv2.putText(panel, "Defect Mask",   (tw*2 + 28, 22), font, 0.55, (0, 120, 0), 1)

    return panel


# ─── Main ────────────────────────────────────────────────────────────────────────

DEFECT_FNS = {
    "scratch":        make_scratch,
    "missing_copper": make_missing_copper,
    "short_circuit":  make_short_circuit,
}

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image = cv2.imread(INPUT_PATH)
    if image is None:
        print(f"[ERROR] Cannot load image: {INPUT_PATH}")
        return

    h, w = image.shape[:2]
    print(f"[INFO] Image loaded: {INPUT_PATH}  ({w}x{h})")

    trace_contours = detect_traces(image)
    print(f"[INFO] Detected {len(trace_contours)} copper trace regions")

    if len(trace_contours) < 3:
        print("[ERROR] Too few trace regions detected.")
        return

    # Save trace map — include this in your report
    trace_vis = image.copy()
    cv2.drawContours(trace_vis, trace_contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "00_trace_map.jpg"), trace_vis)
    print("[INFO] Trace map saved -> 00_trace_map.jpg\n")

    cycle = list(DEFECT_FNS.keys())

    for i in range(NUM_SAMPLES):
        dtype    = cycle[i % len(cycle)]
        defected, dmask = DEFECT_FNS[dtype](image.copy(), trace_contours)

        base = f"sample_{i+1:02d}_{dtype}"
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{base}_defected.jpg"), defected)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{base}_mask.png"),     dmask)

        panel = make_panel(image, defected, dmask, dtype)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{base}_comparison.jpg"), panel)

        print(f"  [{i+1}/{NUM_SAMPLES}] {dtype} -> {base}_comparison.jpg")

    print(f"\n[DONE] All outputs saved to: {OUTPUT_PATH}")
    print(f"\nFor your report:")
    print(f"  00_trace_map.jpg           -> Novel feature proof")
    print(f"  sample_XX_*_comparison.jpg -> Results section")


if __name__ == "__main__":
    main()