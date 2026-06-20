"""
PCB Defect Awareness Module
-----------------------------
Reads severity_report.csv from OUTPUT3,
generates visual awareness signals and a risk report.

- Risk Flag per defect (even if score is low but on copper)
- Risk Gauge visualization
- Board-level Risk Summary
- Actionable Recommendation per defect

Output saved to: D:\PCB DEFECT GENERATOR\OUTPUT3\AWARENESS

Usage: python pcb_awareness.py
"""

import cv2
import numpy as np
import os
import csv

# ─── PATHS ───────────────────────────────────────────────────────────────────────

CSV_INPUT   = r"D:\PCB DEFECT GENERATOR\OUTPUT3\severity_report.csv"
PANELS_DIR  = r"D:\PCB DEFECT GENERATOR\OUTPUT3"
OUTPUT_PATH = r"D:\PCB DEFECT GENERATOR\OUTPUT3\AWARENESS2"

# ─── Risk Rules (independent of severity score) ──────────────────────────────────

# Even a LOW score can trigger a HIGH risk flag if defect type is inherently dangerous
DEFECT_RISK_WEIGHT = {
    "scratch":        1.0,   # moderate inherent risk
    "missing_copper": 1.5,   # high — breaks circuit path
    "short_circuit":  2.0,   # critical — always dangerous regardless of size
}

RISK_THRESHOLDS = {
    "LOW":      (0,   15),
    "ELEVATED": (15,  35),
    "HIGH":     (35,  60),
    "CRITICAL": (60, 200),
}

RECOMMENDATIONS = {
    "scratch": {
        "LOW":      "Board can proceed to functional testing. Flag defect location for periodic re-inspection.",
        "ELEVATED": "Inspect scratch region manually before powering on. Verify copper continuity along affected trace.",
        "HIGH":     "Board must not be powered on. Send for rework. If rework not feasible, discard the board.",
        "CRITICAL": "Board is non-functional. Discard or return to manufacturer. Do not deploy under any condition.",
    },
    "missing_copper": {
        "LOW":      "ATTENTION: Copper loss detected on trace. Even small gaps risk open circuit. Inspect before testing.",
        "ELEVATED": "ALERT: Copper loss on active trace. Risk of open circuit. Mandatory inspection before powering on.",
        "HIGH":     "Board must not be powered on. Copper loss may cause open circuit fault. Send for rework or discard.",
        "CRITICAL": "Board is non-functional due to severe copper loss. Discard or return to manufacturer immediately.",
    },
    "short_circuit": {
        "LOW":      "WARNING: Short circuit bridge detected. Even small bridges cause leakage or signal issues. Inspect before testing.",
        "ELEVATED": "HIGH ALERT: Short circuit bridge on copper zone. Risk of component damage. Do not power on without inspection.",
        "HIGH":     "Board must not be powered on. Active short circuit risk. Send for rework or discard immediately.",
        "CRITICAL": "Board is non-functional. Confirmed short circuit. Discard or return to manufacturer. Do not deploy.",
    },
}


def get_risk_level(score, defect_type):
    """
    Compute risk level using weighted score.
    Short circuits get 2x weight — a small short is still dangerous.
    """
    weight       = DEFECT_RISK_WEIGHT.get(defect_type, 1.0)
    weighted     = score * weight
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= weighted < hi:
            return level, round(weighted, 1)
    return "CRITICAL", round(weighted, 1)


LEVEL_COLORS = {
    "LOW":      (0,   180,   0),    # green
    "ELEVATED": (0,   200, 255),    # yellow
    "HIGH":     (0,   140, 255),    # orange
    "CRITICAL": (0,     0, 220),    # red
}

LEVEL_BGR = {
    "LOW":      (20,  80,  20),
    "ELEVATED": (10,  70,  80),
    "HIGH":     (10,  50,  90),
    "CRITICAL": (20,  10,  80),
}


# ─── Gauge Visual ────────────────────────────────────────────────────────────────

def draw_gauge(score, weighted, risk_level, defect_type, sample):
    """Draw a semicircular risk gauge with needle."""
    W, H   = 400, 260
    canvas = np.full((H, W, 3), 18, dtype=np.uint8) 
    cx, cy = W // 2, H - 60
    R      = 130

    # Draw arc segments (LOW=green, ELEVATED=yellow, HIGH=orange, CRITICAL=red)
    segments = [
        (180, 225, (0, 180, 0)),      # LOW
        (225, 270, (0, 200, 180)),    # ELEVATED
        (270, 315, (0, 140, 255)),    # HIGH
        (315, 360, (0, 0, 220)),      # CRITICAL
    ]
    for start, end, color in segments:
        cv2.ellipse(canvas, (cx, cy), (R, R), 0, start, end, color, 18)

    # Needle: map weighted score (0-120) to angle (180-360 degrees)
    needle_angle = 180 + min(weighted / 120.0, 1.0) * 180
    rad          = np.radians(needle_angle)
    nx           = int(cx + (R - 25) * np.cos(rad))
    ny           = int(cy + (R - 25) * np.sin(rad))
    cv2.line(canvas, (cx, cy), (nx, ny), (255, 255, 255), 3)
    cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "LOW",      (18,  H-55), font, 0.38, (0,180,0),   1)
    cv2.putText(canvas, "ELEV.",    (75,  H-90), font, 0.38, (0,200,180), 1)
    cv2.putText(canvas, "HIGH",     (W-100,H-90),font, 0.38, (0,140,255), 1)
    cv2.putText(canvas, "CRIT.",    (W-60, H-55),font, 0.38, (0,0,220),   1)

    # Center info
    color = LEVEL_COLORS[risk_level]
    cv2.putText(canvas, f"{risk_level}",     (cx-35, cy-18), font, 0.75, color, 2)
    cv2.putText(canvas, f"Weighted: {weighted}", (cx-55, cy+12), font, 0.38, (180,180,180), 1)

    # Title
    cv2.putText(canvas, f"Risk Gauge — {sample}", (10, 22), font, 0.42, (220,220,220), 1)
    cv2.putText(canvas, defect_type.replace("_"," ").title(), (10, 42), font, 0.42, color, 1)

    return canvas


# ─── Awareness Card ──────────────────────────────────────────────────────────────

def draw_awareness_card(row, risk_level, weighted):
    """Full awareness card with risk signal, recommendation, and stats."""
    sample, dtype, score, area_pct, cls, regions = row
    score    = float(score)
    area_pct = float(area_pct)
    regions  = int(regions)

    W, H  = 900, 320
    bg    = LEVEL_BGR[risk_level]
    card = np.full((H, W, 3), bg, dtype=np.uint8)

    color = LEVEL_COLORS[risk_level]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Left side: risk signal box
    cv2.rectangle(card, (0, 0), (220, H), tuple(max(0,c-10) for c in bg), -1)
    cv2.putText(card, "RISK LEVEL",   (15, 40),  font, 0.50, (180,180,180), 1)
    cv2.putText(card, risk_level,     (15, 95),  font, 1.10, color, 2)
    cv2.putText(card, f"Score:    {score}/100",    (15, 135), font, 0.42, (200,200,200), 1)
    cv2.putText(card, f"Weighted: {weighted}/100",  (15, 155), font, 0.42, (200,200,200), 1)
    cv2.putText(card, f"Area:     {area_pct}%",    (15, 175), font, 0.42, (200,200,200), 1)
    cv2.putText(card, f"Regions:  {regions}",      (15, 195), font, 0.42, (200,200,200), 1)
    cv2.putText(card, f"Type: {dtype.replace('_',' ').title()}", (15, 220), font, 0.40, color, 1)

    # Divider
    cv2.line(card, (225, 10), (225, H-10), (80,80,80), 1)

    # Right side: recommendation
    cv2.putText(card, f"AWARENESS SIGNAL — {sample.upper()}", (240, 35), font, 0.52, (220,220,220), 1)
    cv2.line(card, (240, 45), (W-20, 45), (80,80,80), 1)

    rec = RECOMMENDATIONS.get(dtype, {}).get(risk_level, "Inspect board manually.")

    # Word wrap recommendation text
    words     = rec.split()
    lines     = []
    line      = ""
    max_chars = 58
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line += (" " if line else "") + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = 80
    for l in lines:
        cv2.putText(card, l, (240, y), font, 0.46, color, 1)
        y += 28

    # Bottom bar
    cv2.rectangle(card, (0, H-38), (W, H), (10,10,10), -1)
    cv2.putText(card, "PCB Defect Awareness Module  |  Topology-Aware Detection System",
                (15, H-14), font, 0.38, (120,120,120), 1)

    # Border
    cv2.rectangle(card, (1,1), (W-1,H-1), color, 2)

    return card


# ─── Board Summary ───────────────────────────────────────────────────────────────

def draw_board_summary(rows):
    """Overall board risk summary panel."""
    W, H  = 900, 380
    canvas = np.full((H, W, 3), 15, dtype=np.uint8)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # Compute overall board risk
    weighted_scores = []
    for row in rows:
        sample, dtype, score, area_pct, cls, regions = row
        _, w = get_risk_level(float(score), dtype)
        weighted_scores.append(w)

    max_w    = max(weighted_scores)
    avg_w    = round(sum(weighted_scores) / len(weighted_scores), 1)
    board_level, _ = get_risk_level(max_w, "short_circuit")  # use max for board level
    color    = LEVEL_COLORS[board_level]

    # Title
    cv2.putText(canvas, "BOARD-LEVEL RISK SUMMARY", (20, 35), font, 0.70, (220,220,220), 2)
    cv2.line(canvas, (20, 48), (W-20, 48), (80,80,80), 1)

    # Overall risk
    cv2.putText(canvas, "Overall Board Risk:",  (20, 85),  font, 0.55, (180,180,180), 1)
    cv2.putText(canvas, board_level,            (280, 85), font, 0.85, color, 2)
    cv2.putText(canvas, f"Max Weighted Score: {max_w}",  (20, 115), font, 0.48, (160,160,160), 1)
    cv2.putText(canvas, f"Avg Weighted Score: {avg_w}",  (20, 138), font, 0.48, (160,160,160), 1)
    cv2.putText(canvas, f"Total Defects Found: {len(rows)}", (20, 161), font, 0.48, (160,160,160), 1)

    cv2.line(canvas, (20, 178), (W-20, 178), (60,60,60), 1)

    # Per defect summary table
    headers = ["Sample", "Type", "Score", "Weighted", "Risk Level", "Action"]
    col_x   = [20, 130, 310, 390, 480, 590]
    cv2.putText(canvas, "DEFECT BREAKDOWN", (20, 205), font, 0.48, (150,150,150), 1)

    y = 230
    for i, hdr in enumerate(headers):
        cv2.putText(canvas, hdr, (col_x[i], y), font, 0.40, (130,130,130), 1)
    cv2.line(canvas, (20, y+8), (W-20, y+8), (50,50,50), 1)

    y = 255
    for row in rows:
        sample, dtype, score, area_pct, cls, regions = row
        rl, ws = get_risk_level(float(score), dtype)
        rc     = LEVEL_COLORS[rl]
        action = "PROCEED TO TEST" if rl == "LOW" else ("INSPECT FIRST" if rl == "ELEVATED" else ("REWORK / DISCARD" if rl == "HIGH" else "DISCARD"))

        cv2.putText(canvas, sample,                      (col_x[0], y), font, 0.38, (200,200,200), 1)
        cv2.putText(canvas, dtype.replace("_"," "),      (col_x[1], y), font, 0.38, (200,200,200), 1)
        cv2.putText(canvas, str(score),                  (col_x[2], y), font, 0.38, (200,200,200), 1)
        cv2.putText(canvas, str(ws),                     (col_x[3], y), font, 0.38, rc,            1)
        cv2.putText(canvas, rl,                          (col_x[4], y), font, 0.38, rc,            1)
        cv2.putText(canvas, action,                      (col_x[5], y), font, 0.38, rc,            1)
        y += 22

    # Bottom recommendation
    cv2.line(canvas, (20, H-50), (W-20, H-50), (60,60,60), 1)
    board_rec = {
    "LOW":      "Board passes screening. Proceed to functional testing with defect locations flagged.",
    "ELEVATED": "Board requires manual inspection of defect regions before powering on.",
    "HIGH":     "Board must not be powered on. Send for rework or discard if rework not feasible.",
    "CRITICAL": "Board is non-functional. Discard or return to manufacturer. Do not deploy.",
}[board_level]
    cv2.putText(canvas, f"Board Recommendation: {board_rec}", (20, H-22), font, 0.40, color, 1)
    cv2.rectangle(canvas, (1,1), (W-1,H-1), color, 2)

    return canvas


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load CSV
    rows = []
    with open(CSV_INPUT, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                rows.append(row)

    print(f"[INFO] Loaded {len(rows)} defect records from CSV\n")

    for row in rows:
        sample, dtype, score, area_pct, cls, regions = row
        risk_level, weighted = get_risk_level(float(score), dtype)

        print(f"  [{sample}] {dtype:20s}  score={score}  weighted={weighted}  → {risk_level}")

        # Gauge
        gauge = draw_gauge(float(score), weighted, risk_level, dtype, sample)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{sample}_gauge.jpg"), gauge)

        # Awareness card
        card = draw_awareness_card(row, risk_level, weighted)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f"{sample}_awareness.jpg"), card)

    # Board summary
    summary = draw_board_summary(rows)
    cv2.imwrite(os.path.join(OUTPUT_PATH, "board_risk_summary.jpg"), summary)
    print(f"\n  [SAVED] board_risk_summary.jpg")

    print(f"\n[DONE] All awareness outputs saved to: {OUTPUT_PATH}")
    print(f"\nFor your report:")
    print(f"  board_risk_summary.jpg     -> Overall board risk table")
    print(f"  sample_XX_awareness.jpg    -> Per-defect awareness card")
    print(f"  sample_XX_gauge.jpg        -> Risk gauge visual")


if __name__ == "__main__":
    main()