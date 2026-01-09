import cv2
import numpy as np

VIDEO_PATH = "video_1.mp4"
HOMO_PATH  = "table_homography.npz"

data = np.load(HOMO_PATH)
M, W, H = data["M"], int(data["W"]), int(data["H"])

paused = False
last_table = None

def mouse_callback(event, x, y, flags, param):
    global last_table
    if event == cv2.EVENT_LBUTTONDOWN and last_table is not None:
        hsv = cv2.cvtColor(last_table, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        print(f"HSV click ({x},{y}) -> H:{h} S:{s} V:{v}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No puedo abrir el vídeo.")

cv2.namedWindow("table")
cv2.setMouseCallback("table", mouse_callback)

while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            break

        table = cv2.warpPerspective(frame, M, (W, H))
        last_table = table.copy()

    # si está pausado, sigue mostrando el último frame
    if last_table is None:
        continue

    cv2.imshow("table", last_table)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:  
        break
    elif k == ord('p'):
        paused = not paused
        print("PAUSA:", paused)

cap.release()
cv2.destroyAllWindows()
