import cv2
import numpy as np
from collections import deque

VIDEO_PATH = "video_2.mp4"
HOMO_PATH  = "table_homography2.npz"

# ===== HSV =====
HSV_LOWER = (148, 50, 50)   
HSV_UPPER = (172,255,255)


# ===== Tracker params =====
GATE_DIST = 200.0
MAX_MISS  = 30
TRACK_LEN = 150

# ===== Detección asimétrica TOP/BOTTOM =====
MIN_AREA_TOP = 8
MIN_AREA_BOT = 20
MIN_R_TOP = 1
MIN_R_BOT = 2
MAX_R_TOP = 80
MAX_R_BOT = 80
CIRC_MIN_TOP = 0.06
CIRC_MIN_BOT = 0.10


data = np.load(HOMO_PATH)
M, W, H = data["M"], int(data["W"]), int(data["H"])

def init_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def find_candidates(table_bgr):
    hsv = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        (x, y), r = cv2.minEnclosingCircle(c)

        # TOP más permisivo
        if y < H/2:
            min_area = MIN_AREA_TOP
            min_r = MIN_R_TOP
            max_r = MAX_R_TOP
        else:
            min_area = MIN_AREA_BOT
            min_r = MIN_R_BOT
            max_r = MAX_R_BOT

        if area < min_area:
            continue
        if r < min_r or r > max_r:
            continue

        peri = cv2.arcLength(c, True)
        if peri > 1e-6:
            circ = 4*np.pi*area/(peri*peri)
            circ_thr = CIRC_MIN_TOP if y < H/2 else CIRC_MIN_BOT
            if circ < circ_thr:
                continue


        cands.append((float(x), float(y), float(r), float(area)))

    return cands, mask

def choose_measurement(cands, px, py, have_init):
    if not cands:
        return None

    if not have_init:
        x, y, r, a = max(cands, key=lambda t: t[3])
        return (x, y, r)

    best = None
    best_d = 1e18
    for (x, y, r, a) in cands:
        d = (x - px)*(x - px) + (y - py)*(y - py)
        if d < best_d:
            best_d = d
            best = (x, y, r)

    if best is None:
        return None
    if np.sqrt(best_d) > GATE_DIST:
        return None
    return best

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No puedo abrir el vídeo.")

kf = init_kalman()
have_init = False
miss = 0

track = deque(maxlen=TRACK_LEN)  # (x,y,valid)

tick_prev = cv2.getTickCount()
fps_s = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    table = cv2.warpPerspective(frame, M, (W, H))

    # predicción
    pred = kf.predict()
    px = float(pred[0, 0])
    py = float(pred[1, 0])

    cands, mask = find_candidates(table)
    meas = choose_measurement(cands, px, py, have_init)

    valid = False
    x, y, r = px, py, 0.0

    if meas is not None:
        mx, my, mr = meas
        if not have_init:
            kf.statePost = np.array([[mx],[my],[0],[0]], np.float32)
            have_init = True
        else:
            z = np.array([[np.float32(mx)], [np.float32(my)]])
            kf.correct(z)

        x, y, r = mx, my, mr
        valid = True
        miss = 0
    else:
        miss += 1

    if miss >= MAX_MISS:
        have_init = False
        miss = 0
        track.clear()

    track.append((x, y, valid))

    # estela (solo detecciones válidas consecutivas)
    for i in range(1, len(track)):
        x1, y1, v1 = track[i-1]
        x2, y2, v2 = track[i]
        if v1 and v2:
            cv2.line(table, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)

    # punto actual
    if valid:
        cv2.circle(table, (int(x), int(y)), int(max(r, 2)), (0,255,0), 2)
        cv2.circle(table, (int(x), int(y)), 2, (0,255,0), -1)
    else:
        cv2.circle(table, (int(px), int(py)), 3, (0,0,255), -1)

    # FPS
    tick_now = cv2.getTickCount()
    dt = (tick_now - tick_prev) / cv2.getTickFrequency()
    tick_prev = tick_now
    fps = 1.0 / max(dt, 1e-6)
    fps_s = 0.9*fps_s + 0.1*fps if fps_s > 0 else fps

    cv2.putText(table, f"FPS:{fps_s:.1f} miss:{miss} cands:{len(cands)}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("track_ball", table)
    # cv2.imshow("mask", mask)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
