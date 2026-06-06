# =============================================================================
#  setup_ref.py — Capture reference images + select ROI + auto-update inspect.py
#  Steps:
#    1. Position the object, press SPACE to capture reference image
#    2. Drag mouse to select ROI, press ENTER to confirm
#    3. ROI_L / ROI_R in inspect.py are updated automatically
# =============================================================================
import cv2
import re
import os

MAIN_FILE = "main.py"       # ← main inspection script
DB_DIR    = "database"
CAMS      = {"L": 0, "R": 1}  # ← change index if cameras differ

os.makedirs(DB_DIR, exist_ok=True)
model = input("Model name (e.g. 00001): ").strip()
if not model:
    print("No model name entered"); exit(1)

roi_results = {}

for side, cam_idx in CAMS.items():
    print(f"\n{'='*50}")
    print(f"  Camera {side}  (index {cam_idx})")
    print(f"{'='*50}")

    # ── Take ref image ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Camera index {cam_idx} not found — skipping side {side}")
        continue

    print("Position object then press SPACE to capture | ESC to skip")
    snapshot = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        disp = frame.copy()
        cv2.putText(disp, f"CAM {side}  SPACE=capture  ESC=skip",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,220,220), 2)
        cv2.imshow(f"Capture REF {side}", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            snapshot = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if snapshot is None:
        print(f"Skipped side {side}")
        continue

    # save ref image
    ref_path = f"{DB_DIR}/{model}_{side}.jpg"
    cv2.imwrite(ref_path, snapshot)
    print(f"Reference image saved -> {ref_path}")

    # ── Select ROI on the image ────────────────────────────────────────────────
    ih, iw = snapshot.shape[:2]
    print(f"Image {iw}x{ih} — drag to select ROI, press ENTER to confirm")

    roi = cv2.selectROI(f"Select ROI {side} — ENTER=confirm  C=reset",
                        snapshot, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print(f"No ROI selected for side {side}")
        continue

    rx1, ry1 = x/iw,     y/ih
    rx2, ry2 = (x+w)/iw, (y+h)/ih
    roi_results[side] = (rx1, ry1, rx2, ry2)

    # preview
    preview = snapshot.copy()
    cv2.rectangle(preview, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.putText(preview, f"ROI {side} OK", (x+4, y+24),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
    cv2.imshow(f"Preview {side} — ESC to close", preview)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    print(f"ROI_{side} = ({rx1:.3f}, {ry1:.3f}, {rx2:.3f}, {ry2:.3f})")

# ── Update inspect.py ────────────────────────────────────────────────────
if not roi_results:
    print("\nno ROI — not update")
    exit(0)

if not os.path.exists(MAIN_FILE):
    print(f"\n{MAIN_FILE} not found — skipping update")
    exit(0)

with open(MAIN_FILE, "r", encoding="utf-8") as f:
    src = f.read()

for side, (rx1, ry1, rx2, ry2) in roi_results.items():
    pattern     = rf"ROI_{side}\s*=\s*\([^)]+\)"
    replacement = f"ROI_{side} = ({rx1:.3f}, {ry1:.3f}, {rx2:.3f}, {ry2:.3f})  # auto-updated by setup_ref.py"
    if re.search(pattern, src):
        src = re.sub(pattern, replacement, src)
        print(f"Updated ROI_{side} in {MAIN_FILE}")
    else:
        print(f"ROI_{side} not found in {MAIN_FILE} — skipping")

with open(MAIN_FILE, "w", encoding="utf-8") as f:
    f.write(src)

print(f"\nDone! You can now run {MAIN_FILE}")