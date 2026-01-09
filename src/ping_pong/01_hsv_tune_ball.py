import cv2
import numpy as np
import os
import time

VIDEO_PATH = "data/videos/video_1.mp4"
HOMO_PATH  = "data/calibration/table_homography.npz"
OUT_VIDEO_PATH = "results/01_hsv_tune_ball_output.mp4"

os.makedirs("results", exist_ok=True)

data = np.load(HOMO_PATH)
M, W, H = data["M"], int(data["W"]), int(data["H"])

paused = False
last_table = None
t_prev = time.time()

def mouse_callback(event, x, y, flags, param):
    global last_table
    if event == cv2.EVENT_LBUTTONDOWN and last_table is not None:
        hsv = cv2.cvtColor(last_table, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        print(f"HSV click ({x},{y}) -> H:{h} S:{s} V:{v}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No puedo abrir el vídeo.")

fps_out = cap.get(cv2.CAP_PROP_FPS)
if fps_out is None or fps_out <= 1:
    fps_out = 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps_out, (W, H))

cv2.namedWindow("table")
cv2.setMouseCallback("table", mouse_callback)

while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            break

        table = cv2.warpPerspective(frame, M, (W, H))
        last_table = table.copy()

    if last_table is None:
        continue

    t_now = time.time()
    fps_inst = 1.0 / max(1e-6, (t_now - t_prev))
    t_prev = t_now

    vis = last_table.copy()
    fps_txt = f"FPS:{fps_inst:.1f}"
    cv2.putText(vis, fps_txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, fps_txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)

    out_writer.write(vis)
    cv2.imshow("table", vis)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('p'):
        paused = not paused
        print("PAUSA:", paused)

cap.release()
out_writer.release()
cv2.destroyAllWindows()
