import cv2
import numpy as np

from letter_classifier import load_templates, detect_letter

CALIB_PATH = "calibration_data.npz"

def load_calibration(path: str):
    data = np.load(path, allow_pickle=True)
    K = data["intrinsics"]
    dist = data["dist_coeffs"]
    return K, dist

def main():
    templates = load_templates()

    VIDEO_PATH = "video_contraseña_final.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se ha podido abrir la cámara / vídeo.")
        return

    # Leer un frame para fijar tamaño
    ret, frame0 = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    h, w = frame0.shape[:2]

    # Cargar calibración y calcular new camera matrix 
    K, dist = load_calibration(CALIB_PATH)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0, newImgSize=(w, h))

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    thr = 0.45

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar undistort antes de detectar papel/letra
        frame_und = cv2.undistort(frame, K, dist, None, newK)

        label, score, dbg = detect_letter(frame_und, templates, thr=thr)

        
        out = frame_und.copy()
        txt = f"Letra: {label if label else '-'}  score: {score:.3f}  thr:{thr:.2f}"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("Letter classification (undistorted)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("]"):
            thr = min(0.95, thr + 0.02)
        if key == ord("["):
            thr = max(0.10, thr - 0.02)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
