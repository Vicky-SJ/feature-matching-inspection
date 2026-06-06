# =============================================================================
#  Car Sealer Inspection System  —  ORB Feature Matching Edition
#  Compares live camera frames against reference images using ORB keypoint matching
#  PASS = STABLE_FRAMES consecutive frames matching above MATCH_THRESHOLD
#  Each side (L/R) is evaluated independently
# =============================================================================

import cv2
import numpy as np
import json
import time
import socket
import threading
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CAM_LEFT      = 0
CAM_RIGHT     = 1
LOG_FILE      = "inspection_log.json"

# ── Matching ─────────────────────────────────────────────────────────────────
MATCH_THRESHOLD = 8       # lower = easier to match; raise for production
STABLE_FRAMES   = 10      # consecutive matching frames required for PASS
SKIP_FRAME      = 2       # run matching every N frames to reduce CPU load
LOWE_RATIO      = 0.75    # Lowe's ratio test threshold

# ── ROI (ratio 0.0-1.0) — set using setup_ref.py ────────────────────────────
ROI_L = (0.333, 0.463, 0.573, 0.869)  # auto-updated by setup_ref.py  # ← left camera ROI
ROI_R = (0.391, 0.373, 0.620, 0.604)  # auto-updated by setup_ref.py  # ← right camera ROI

# ── Network ─────────────────────────────────────────────────────────────────
MODEL_SERVER_HOST = "0.0.0.0"
MODEL_SERVER_PORT = 9001
PLC_HOST          = "xxx.xxx.xxx.xxx"
PLC_PORT          = 502
PLC_COIL_TRIGGER  = 0     # coil 0 = start inspection
PLC_COIL_STOP     = 1     # coil 1 = stop inspection

# ── Window ───────────────────────────────────────────────────────────────────
WIN_W   = 1280
WIN_H   = 720
VIDEO_H = int(WIN_H * 0.70)
PADDING = 10
REF_W   = int(WIN_W * 0.32)

# ── Colors ───────────────────────────────────────────────────────────────────
C_BG     = (18,  18,  18)
C_PANEL  = (30,  30,  30)
C_BORDER = (55,  55,  55)
C_WHITE  = (255, 255, 255)
C_GRAY   = (130, 130, 130)
C_GREEN  = (0,   220,  80)
C_RED    = (60,   60, 230)
C_YELLOW = (0,   220, 220)
C_ORANGE = (0,   165, 255)

FONT_MONO = cv2.FONT_HERSHEY_PLAIN


# ─────────────────────────────────────────────────────────────────────────────
#  ORB MATCHER
# ─────────────────────────────────────────────────────────────────────────────
def crop_roi(img, roi):
    """Crop image to ROI zone before sending to ORB"""
    h, w = img.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    return img[int(ry1*h):int(ry2*h), int(rx1*w):int(rx2*w)]


def build_orb():
    orb = cv2.ORB_create(nfeatures=1000)  # เพิ่มจาก 50
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    return orb, bf


def extract_features(orb, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def count_good_matches(bf, des1, des2):
    """Return number of good matches using Lowe's ratio test"""
    if des1 is None or des2 is None:
        return 0
    if len(des1) < 2 or len(des2) < 2:
        return 0
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    return len(good)


def draw_match_overlay(frame, good_count, status):
    """Draw status bar overlay on frame"""
    out = frame.copy()
    color = C_GREEN if status == "PASS" else C_RED if status == "NG" else C_ORANGE
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, f"{status}   match={good_count}", (10, 28),
                FONT_MONO, 1.6, color, 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  INSPECTION STATE
# ─────────────────────────────────────────────────────────────────────────────
class InspectionState:
    def __init__(self, side):
        self.side          = side
        self.status        = "WAITING"   # WAITING / INSPECTING / PASS / NG
        self.stable_count  = 0         
        self.best_match    = 0          
        self.manual        = False
        self.session_log   = []          

    def reset(self):
        self.status       = "INSPECTING"
        self.stable_count = 0
        self.best_match   = 0
        self.manual       = False

    def update(self, good_count):
        if self.status not in ("INSPECTING",):
            return self.status

        self.best_match = max(self.best_match, good_count)

        if good_count >= MATCH_THRESHOLD:
            self.stable_count += 1
        else:
            self.stable_count = 0   # reset

        if self.stable_count >= STABLE_FRAMES:
            self.status = "PASS"

        return self.status

    def stop(self):
        """PLC signal stop — if not PASS → NG"""
        if self.status == "INSPECTING":
            self.status = "NG"
        self.session_log.append({
            "time":       time.strftime("%H:%M:%S"),
            "status":     self.status,
            "best_match": self.best_match,
            "manual":     self.manual,
        })


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def put(canvas, text, x, y, scale=1.0, color=C_WHITE, thickness=1):
    cv2.putText(canvas, text, (x, y), FONT_MONO, scale, color, thickness, cv2.LINE_AA)


def make_window(side, live_frame, ref_img, state,
                current_model, prev_model, next_model, good_count):
    canvas = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)

    # ── topbar ────────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (WIN_W, 64), C_PANEL, -1)
    put(canvas, current_model.upper(), 14, 46, scale=2.2, color=C_WHITE, thickness=2)
    dt = time.strftime("%Y-%m-%d   %H:%M:%S")
    tw = cv2.getTextSize(dt, FONT_MONO, 1.5, 1)[0][0]
    put(canvas, dt, WIN_W - tw - 14, 46, scale=1.5, color=C_GRAY)

    # ── ref panel ─────────────────────────────────────────────────────────────
    rx1, ry1 = PADDING, 64 + PADDING
    rx2, ry2 = rx1 + REF_W, ry1 + VIDEO_H - PADDING
    cv2.rectangle(canvas, (rx1-1, ry1-1), (rx2+1, ry2+1), C_BORDER, 1)
    if ref_img is not None:
        ref_r = cv2.resize(ref_img, (REF_W, ry2-ry1), interpolation=cv2.INTER_LINEAR)
        canvas[ry1:ry2, rx1:rx2] = ref_r
    else:
        put(canvas, f"NO REF {side}", rx1+8, ry1+40, scale=1.2, color=C_GRAY)

    lbl = f"REF {side}"
    lw  = cv2.getTextSize(lbl, FONT_MONO, 1.3, 1)[0][0]
    cv2.rectangle(canvas, (rx1, ry1), (rx1+lw+16, ry1+28), C_BG, -1)
    put(canvas, lbl, rx1+8, ry1+22, scale=1.3, color=C_YELLOW)

    # ── live panel ────────────────────────────────────────────────────────────
    lx1 = rx2 + PADDING
    lx2 = WIN_W - PADDING
    ly1, ly2 = ry1, ry2
    cv2.rectangle(canvas, (lx1-1, ly1-1), (lx2+1, ly2+1), C_BORDER, 1)
    if live_frame is not None:
        lf = cv2.resize(live_frame, (lx2-lx1, ly2-ly1), interpolation=cv2.INTER_LINEAR)
        canvas[ly1:ly2, lx1:lx2] = lf

    lbl2 = f"CAM {side}"
    lw2  = cv2.getTextSize(lbl2, FONT_MONO, 1.3, 1)[0][0]
    cv2.rectangle(canvas, (lx1, ly1), (lx1+lw2+16, ly1+28), C_BG, -1)
    put(canvas, lbl2, lx1+8, ly1+22, scale=1.3, color=C_YELLOW)

    # ── bottom bar ────────────────────────────────────────────────────────────
    by      = VIDEO_H + 64 + PADDING
    bar_mid = by + (WIN_H - by) // 2
    cv2.rectangle(canvas, (0, by-4), (WIN_W, WIN_H), C_PANEL, -1)

    # status ใหญ่
    st_color = (C_GREEN if state.status == "PASS"
                else C_RED    if state.status == "NG"
                else C_ORANGE if state.status == "INSPECTING"
                else C_GRAY)
    put(canvas, state.status, 14, bar_mid+18, scale=3.5, color=st_color, thickness=3)

    # stable bar (progress)
    bar_x1, bar_y1 = 14, bar_mid + 30
    bar_x2 = 14 + 200
    bar_y2 = bar_mid + 42
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), C_BORDER, -1)
    fill = int(200 * min(state.stable_count / STABLE_FRAMES, 1.0))
    if fill > 0:
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1+fill, bar_y2), st_color, -1)
    put(canvas, f"{state.stable_count}/{STABLE_FRAMES} stable  match={good_count}",
        14, bar_mid + 56, scale=1.0, color=C_GRAY)

    # divider
    sx = 240
    cv2.line(canvas, (sx, by+4), (sx, WIN_H-6), C_BORDER, 1)
    sx += 14

    # prev / next model
    put(canvas, "PREVIOUS MODEL", sx, bar_mid-4, scale=1.1, color=C_GRAY)
    put(canvas, (prev_model or "-").upper(), sx, bar_mid+26,
        scale=2.0, color=C_GRAY, thickness=2)
    sx += 260
    cv2.line(canvas, (sx, by+4), (sx, WIN_H-6), C_BORDER, 1)
    sx += 14
    put(canvas, "NEXT MODEL", sx, bar_mid-4, scale=1.1, color=C_GRAY)
    put(canvas, (next_model or "-").upper(), sx, bar_mid+26,
        scale=2.0, color=C_WHITE, thickness=2)

    # manual pass button
    btn_w = 200
    bx1   = WIN_W - btn_w - 14
    bx2   = WIN_W - 14
    bby1  = by + 6
    bby2  = WIN_H - 6
    btn_mid = (bby1 + bby2) // 2
    cv2.rectangle(canvas, (bx1, bby1), (bx2, bby2), (30, 80, 30), -1)
    cv2.rectangle(canvas, (bx1, bby1), (bx2, bby2), C_GREEN, 2)
    put(canvas, "MANUAL PASS", bx1+14, btn_mid+10, scale=1.6, color=C_GREEN, thickness=2)

    return canvas, (bx1, bby1, bx2, bby2)


# ─────────────────────────────────────────────────────────────────────────────
#  MOUSE CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class ClickState:
    def __init__(self):
        self.btn_bbox = None

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.btn_bbox:
            bx1, bby1, bx2, bby2 = self.btn_bbox
            if bx1 <= x <= bx2 and bby1 <= y <= bby2:
                state = param["state"]
                if state.status in ("NG", "INSPECTING"):
                    state.status = "PASS"
                    state.manual = True
                    print(f"[MANUAL PASS] side={param['side']}")


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK THREADS
# ─────────────────────────────────────────────────────────────────────────────
def model_server_thread(model_queue, lock):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
    srv.listen(5)
    print(f"[MODEL SERVER] listening on port {MODEL_SERVER_PORT}")
    while True:
        try:
            conn, addr = srv.accept()
            data = conn.recv(64).decode().strip()
            conn.close()
            if data.isdigit() and len(data) == 5:
                with lock:
                    model_queue.append(data)
                print(f"[MODEL SERVER] model queued: {data}")
        except Exception as e:
            print(f"[MODEL SERVER] error: {e}")


def plc_monitor_thread(trigger_flag, stop_flag, model_queue, model_lock):
    """
    PLC signal by OPC-UA
    Install: pip install opcua

    Node for PLC:
      ns=2;s=ModelNumber   → string 
      ns=2;s=TriggerStart  → bool  True = start
      ns=2;s=TriggerStop   → bool  True = stop
    """
    try:
        from opcua import Client as OpcClient
    except ImportError:
        print("[PLC] opcua not installed — run: pip install opcua")
        return

    OPC_URL = f"opc.tcp://{PLC_HOST}:4840"   # port OPC-UA default
    client  = OpcClient(OPC_URL)

    while True:
        try:
            client.connect()
            print(f"[PLC] OPC-UA connected: {OPC_URL}")

            node_model   = client.get_node("ns=2;s=ModelNumber")
            node_start   = client.get_node("ns=2;s=TriggerStart")
            node_stop    = client.get_node("ns=2;s=TriggerStop")

            while True:
                model_val = str(node_model.get_value()).strip()
                start_val = bool(node_start.get_value())
                stop_val  = bool(node_stop.get_value())

                # รับโมเดลใหม่
                if model_val and model_val.isdigit() and len(model_val) == 5:
                    with model_lock:
                        if not model_queue or model_queue[-1] != model_val:
                            model_queue.append(model_val)
                            print(f"[PLC] new model: {model_val}")

                if start_val:
                    trigger_flag.set()
                    node_start.set_value(False)

                if stop_val:
                    stop_flag.set()
                    node_stop.set_value(False)

                time.sleep(0.2)

        except Exception as e:
            print(f"[PLC] OPC-UA error: {e} — retrying in 3s")
            try: client.disconnect()
            except: pass
            time.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model_queue  = deque()
    model_lock   = threading.Lock()
    trigger_flag = threading.Event()
    stop_flag    = threading.Event()

    # ── manual input (comment out when using OPC-UA) ────────────────────────────
    current = input("Model: ").strip()
    if not current:
        print("No model entered"); exit(1)
    model_history = deque(maxlen=10)
    model_history.append(current)

    # ── network threads (uncomment when ready) ───────────────────────────────────
    # threading.Thread(target=model_server_thread,
    #                  args=(model_queue, model_lock), daemon=True).start()
    # threading.Thread(target=plc_monitor_thread,
    #                  args=(trigger_flag, stop_flag, model_queue, model_lock),
    #                  daemon=True).start()

    def load_ref(model):
        l = cv2.imread(f"database/{model}_L.jpg")
        r = cv2.imread(f"database/{model}_R.jpg")
        print(f"Ref L: {'OK' if l is not None else 'NOT FOUND'}")
        print(f"Ref R: {'OK' if r is not None else 'NOT FOUND'}")
        return l, r

    imgl, imgr = load_ref(current)

    # ── ORB ───────────────────────────────────────────────────────────────────
    orb, bf = build_orb()
    kp_ref_L, des_ref_L = extract_features(orb, crop_roi(imgl, ROI_L)) if imgl is not None else ([], None)
    kp_ref_R, des_ref_R = extract_features(orb, crop_roi(imgr, ROI_R)) if imgr is not None else ([], None)
    print(f"[ORB] ref keypoints  L:{len(kp_ref_L)}  R:{len(kp_ref_R)}")
    if len(kp_ref_L) < 10: print("  WARNING: REF L has very few keypoints — try expanding ROI or recapturing ref image")
    if len(kp_ref_R) < 10: print("  WARNING: REF R has very few keypoints — try expanding ROI or recapturing ref image")

    # ── video / camera source ───────────────────────────────────────────────────
    # capL = cv2.VideoCapture("test_L.mp4")   # video file for testing
    # capR = cv2.VideoCapture("test_R.mp4")
    capL = cv2.VideoCapture(CAM_LEFT)         # live webcam
    capR = cv2.VideoCapture(CAM_RIGHT)
    if not capL.isOpened(): print("Cannot open left camera");  exit(1)
    if not capR.isOpened(): print("Cannot open right camera"); exit(1)

    # ── state ─────────────────────────────────────────────────────────────────
    stateL = InspectionState("L")
    stateR = InspectionState("R")
    stateL.status = "INSPECTING"   # auto-inspect mode (change to WAITING when using PLC)
    stateR.status = "INSPECTING"

    # ── windows ───────────────────────────────────────────────────────────────
    WIN_L = "Inspection LEFT"
    WIN_R = "Inspection RIGHT"
    for name, x in [(WIN_L, 0), (WIN_R, WIN_W)]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, WIN_W, WIN_H)
        cv2.moveWindow(name, x, 0)

    clickL = ClickState()
    clickR = ClickState()
    cv2.setMouseCallback(WIN_L, clickL.on_mouse, {"state": stateL, "side": "L"})
    cv2.setMouseCallback(WIN_R, clickR.on_mouse, {"state": stateR, "side": "R"})

    frame_idx  = 0
    good_cnt_L = 0
    good_cnt_R = 0
    print("\nRunning — Press ESC to stop and show results\n")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        # ── PLC trigger / stop ────────────────────────────────────────────────
        if trigger_flag.is_set():
            trigger_flag.clear()
            stateL.reset(); stateR.reset()
            print("[TRIGGER] inspection started")

        if stop_flag.is_set():
            stop_flag.clear()
            stateL.stop(); stateR.stop()
            print(f"[STOP] L={stateL.status} R={stateR.status}")

        # ── model switch ───────────────────────────────
        # Get new model from PLC (if has) → switch immediately
        with model_lock:
            if model_queue and model_queue[0] != current:
                current = model_queue.popleft()
                model_history.append(current)
                imgl, imgr = load_ref(current)
                kp_ref_L, des_ref_L = extract_features(orb, crop_roi(imgl, ROI_L)) if imgl is not None else ([], None)
                kp_ref_R, des_ref_R = extract_features(orb, crop_roi(imgr, ROI_R)) if imgr is not None else ([], None)
                print(f"[MODEL SWITCH] -> {current}")
            next_in_queue = model_queue[0] if model_queue else None

        # ── ORB matching ทุก SKIP_FRAME ───────────────────────────────────────
        # reuse keypoints — cal one per frame
        _, des_L = extract_features(orb, crop_roi(frameL, ROI_L))
        _, des_R = extract_features(orb, crop_roi(frameR, ROI_R))

        if frame_idx % SKIP_FRAME == 0:
            if stateL.status == "INSPECTING" and des_ref_L is not None:
                good_cnt_L = count_good_matches(bf, des_ref_L, des_L)
                stateL.update(good_cnt_L)

            if stateR.status == "INSPECTING" and des_ref_R is not None:
                good_cnt_R = count_good_matches(bf, des_ref_R, des_R)
                stateR.update(good_cnt_R)

        frame_idx += 1

        # ── overlay (reuse kp from above) ─────────────────────────────────────
        # draw ROI box on live frame
        def draw_roi_box(frame, roi):
            h, w = frame.shape[:2]
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame,
                          (int(rx1*w), int(ry1*h)),
                          (int(rx2*w), int(ry2*h)),
                          (0, 220, 220), 2)
        draw_roi_box(frameL, ROI_L)
        draw_roi_box(frameR, ROI_R)
        frameL_disp = draw_match_overlay(frameL, good_cnt_L, stateL.status)
        frameR_disp = draw_match_overlay(frameR, good_cnt_R, stateR.status)

        prev_model = model_history[-2] if len(model_history) >= 2 else None

        canvasL, bboxL = make_window("LEFT",  frameL_disp, imgl, stateL,
                                     current, prev_model, next_in_queue, good_cnt_L)
        canvasR, bboxR = make_window("RIGHT", frameR_disp, imgr, stateR,
                                     current, prev_model, next_in_queue, good_cnt_R)
        clickL.btn_bbox = bboxL
        clickR.btn_bbox = bboxR

        cv2.imshow(WIN_L, canvasL)
        cv2.imshow(WIN_R, canvasR)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            # ESC = stop inspection
            stateL.stop()
            stateR.stop()
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    # ── save log ────────────────────────────────────────────────────────────────
    log = {
        "model": current,
        "L": {"status": stateL.status, "best_match": stateL.best_match,
              "manual": stateL.manual, "sessions": stateL.session_log},
        "R": {"status": stateR.status, "best_match": stateR.best_match,
              "manual": stateR.manual, "sessions": stateR.session_log},
    }
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"\nLog saved -> {LOG_FILE}")
    except Exception as e:
        print(f"Log error: {e}")

    print("\n========== SUMMARY ==========")
    print(f"  Model : {current}")
    print(f"  L     : {stateL.status}  best_match={stateL.best_match}")
    print(f"  R     : {stateR.status}  best_match={stateR.best_match}")
    print("==============================")


if __name__ == "__main__":
    main()