import cv2
import numpy as np
from collections import deque
import os

VIDEO_PATH = "data/videos/video_1.mp4"
HOMO_PATH  = "data/calibration/table_homography.npz"
OUT_VIDEO_PATH = "results/03_score_bounces_output.mp4"

os.makedirs("results", exist_ok=True)

HSV_LOWER = (148, 80, 80)
HSV_UPPER = (172, 255, 255)

GATE_DIST = 200.0
MAX_MISS  = 30
TRACK_LEN = 120

MIN_AREA_TOP = 8
MIN_AREA_BOT = 20
MIN_R_TOP = 1
MIN_R_BOT = 2
MAX_R_TOP = 80
MAX_R_BOT = 80
CIRC_MIN = 0.10

MARGIN_X   = 8
MARGIN_TOP = 0
MARGIN_BOT = 8

MID_BAND = 25
LOST_RESET = 200
COOLDOWN_FRAMES = 14

SPEED_MIN = 3.5

ACC_MIN_TOP = 5.0
ACC_MIN_BOT = 10.0
ANGLE_MIN_TOP = 15.0
ANGLE_MIN_BOT = 30.0

REV_COS_MAX = -0.25

BASE_DELAY_MS = 1
SLOW_DELAY_MS = 80
SLOW_MO_FRAMES = 30
slow_counter = 0

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

def side_from_pos(x, y, last_side=None):
    if y < H/2 - MID_BAND:
        return "TOP"
    if y > H/2 + MID_BAND:
        return "BOT"
    return last_side if last_side is not None else ("TOP" if y < H/2 else "BOT")

def angle_between(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

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
            if circ < CIRC_MIN:
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

def draw_winner_screen(frame_bgr, winner_text):
    out = frame_bgr.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]), (0,0,0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    msg1 = f"WINNER: {winner_text}"
    msg2 = "MODE: FIRST POINT WINS"
    cv2.putText(out, msg1, (40, out.shape[0]//2 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, msg2, (40, out.shape[0]//2 + 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)
    return out

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No puedo abrir el vídeo.")

fps_out = cap.get(cv2.CAP_PROP_FPS)
if fps_out is None or fps_out <= 1:
    fps_out = 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps_out, (W, H))

kf = init_kalman()
have_init = False
miss = 0

track = deque(maxlen=TRACK_LEN)
p_hist = deque(maxlen=10)

cooldown = 0
bounce_total = 0

winner = None

last_bounce_side = None
bounce_streak = 0
rally_active = False

lost_counter = 0
last_table_frame = None

tick_prev = cv2.getTickCount()
fps_s = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    table = cv2.warpPerspective(frame, M, (W, H))
    last_table_frame = table.copy()

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
        lost_counter = 0
    else:
        miss += 1
        lost_counter += 1

    if miss >= MAX_MISS:
        have_init = False
        miss = 0
        track.clear()
        p_hist.clear()

    if lost_counter >= LOST_RESET:
        rally_active = False
        last_bounce_side = None
        bounce_streak = 0
        cooldown = 0
        lost_counter = 0

    track.append((x, y, valid))

    pseudo_valid = (not valid) and have_init and (miss <= 3)
    use_x = x if valid else px
    use_y = y if valid else py

    in_table = (valid or pseudo_valid) and (MARGIN_X <= use_x <= W-MARGIN_X) and (MARGIN_TOP <= use_y <= H-MARGIN_BOT)
    p_hist.append((use_x, use_y, in_table))

    cv2.line(table, (0, H//2), (W, H//2), (0, 255, 255), 2)

    for i in range(1, len(track)):
        x1, y1, v1 = track[i-1]
        x2, y2, v2 = track[i]
        if v1 and v2:
            cv2.line(table, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)

    if valid:
        cv2.circle(table, (int(x), int(y)), int(max(r, 2)), (0,255,0), 2)
        cv2.circle(table, (int(x), int(y)), 2, (0,255,0), -1)
    else:
        cv2.circle(table, (int(px), int(py)), 3, (0,0,255), -1)

    bounce = False
    bounce_side = None
    spd = 0.0
    acc = 0.0
    ang = 0.0
    cosBC = 1.0

    if cooldown > 0:
        cooldown -= 1

    if len(p_hist) >= 4 and cooldown == 0:
        (x0,y0,v0) = p_hist[-4]
        (x1,y1,v1) = p_hist[-3]
        (x2,y2,v2) = p_hist[-2]
        (x3,y3,v3) = p_hist[-1]

        if v0 and v1 and v2 and v3:
            vA = np.array([x1-x0, y1-y0], dtype=np.float32)
            vB = np.array([x2-x1, y2-y1], dtype=np.float32)
            vC = np.array([x3-x2, y3-y2], dtype=np.float32)

            a1 = vB - vA
            a2 = vC - vB

            acc = float(max(np.linalg.norm(a1), np.linalg.norm(a2)))
            spd = float(max(np.linalg.norm(vB), np.linalg.norm(vC)))
            ang = angle_between(vB, vC)

            nB = float(np.linalg.norm(vB))
            nC = float(np.linalg.norm(vC))
            if nB > 1e-6 and nC > 1e-6:
                cosBC = float(np.dot(vB, vC) / (nB * nC))

            side_tmp = side_from_pos(x2, y2, last_bounce_side)
            acc_thr = ACC_MIN_TOP if side_tmp == "TOP" else ACC_MIN_BOT
            ang_thr = ANGLE_MIN_TOP if side_tmp == "TOP" else ANGLE_MIN_BOT

            if (spd > SPEED_MIN) and (cosBC > REV_COS_MAX) and ((acc > acc_thr) or (ang > ang_thr)):
                bounce = True
                bounce_side = side_tmp
                cooldown = COOLDOWN_FRAMES

    if bounce:
        bounce_total += 1
        rally_active = True
        slow_counter = max(slow_counter, SLOW_MO_FRAMES)

        if bounce_side == last_bounce_side:
            bounce_streak += 1
        else:
            last_bounce_side = bounce_side
            bounce_streak = 1

        cv2.putText(table, f"BOUNCE {bounce_side}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        if winner is None and rally_active and bounce_streak >= 2:
            winner = "BOT" if bounce_side == "TOP" else "TOP"
            rally_active = False
            last_bounce_side = None
            bounce_streak = 0

    hud_line1 = "BOT  vs  TOP"
    hud_line2 = "WIN CONDITION: FIRST POINT WINS"

    cv2.putText(table, hud_line1, (15, H-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(table, hud_line2, (15, H-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(table,
                f"bounces:{bounce_total} cd:{cooldown} cands:{len(cands)} miss:{miss} spd:{spd:.1f} acc:{acc:.1f} ang:{ang:.1f} cos:{cosBC:.2f}",
                (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if slow_counter > 0:
        cv2.putText(table, "SLOW MOTION", (W-230, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    tick_now = cv2.getTickCount()
    dt = (tick_now - tick_prev) / cv2.getTickFrequency()
    tick_prev = tick_now
    fps = 1.0 / max(dt, 1e-6)
    fps_s = 0.9*fps_s + 0.1*fps if fps_s > 0 else fps
    cv2.putText(table, f"FPS:{fps_s:.1f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    out_writer.write(table)
    cv2.imshow("score_bounces", table)

    delay = SLOW_DELAY_MS if slow_counter > 0 else BASE_DELAY_MS
    k = cv2.waitKey(delay) & 0xFF
    if slow_counter > 0:
        slow_counter -= 1

    if k == 27:
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()

if winner is None:
    winner_text = "NO_DECISION"
else:
    winner_text = winner

print(f"WINNER: {winner_text}")

if last_table_frame is not None:
    final_img = draw_winner_screen(last_table_frame, winner_text)
    cv2.imshow("WINNER", final_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
