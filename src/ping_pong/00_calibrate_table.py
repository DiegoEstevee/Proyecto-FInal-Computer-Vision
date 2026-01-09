import cv2
import numpy as np

VIDEO_PATH = "video_1.mp4"          
HOMO_PATH  = "table_homography.npz" 
H = 450
W = int(H * 2.74 / 1.525)
print("Usando W,H =", W, H)
            

pts_src = []

def mouse_callback(event, x, y, flags, param):
    global pts_src
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_src) < 4:
        pts_src.append([x, y])
        print(f"Punto {len(pts_src)}: ({x}, {y})")

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("No puedo leer el vídeo. Revisa VIDEO_PATH.")

clone = frame.copy()
win = "SCRIPT 0: Click 4 esquinas de la MESA y ENTER (ESC salir)"
cv2.namedWindow(win)
cv2.setMouseCallback(win, mouse_callback)

while True:
    vis = clone.copy()

    for i, p in enumerate(pts_src):
        cv2.circle(vis, tuple(p), 7, (0, 255, 0), -1)
        cv2.putText(vis, str(i+1), (p[0]+10, p[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(vis, "Click 4 esquinas. ENTER para guardar. ESC salir.",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow(win, vis)
    k = cv2.waitKey(20) & 0xFF

    if k == 27:  
        cv2.destroyAllWindows()
        raise SystemExit

    if k == 13 and len(pts_src) == 4:  
        break

cv2.destroyAllWindows()

src = order_points(pts_src)
dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src, dst)
np.savez(HOMO_PATH, M=M, W=W, H=H)

print(" Homografía guardada en:", HOMO_PATH)
print("M =\n", M)
