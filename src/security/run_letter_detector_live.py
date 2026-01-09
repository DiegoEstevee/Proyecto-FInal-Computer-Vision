import cv2
import numpy as np

from letter_classifier import load_templates, detect_letter

CALIB_PATH = "src/calibration/calibration_data.npz"
VIDEO_PATH = "data/videos/password.mp4"
OUT_VIDEO_PATH = "results/letter_detector_output.mp4"

def load_calibration(path: str):
    data = np.load(path, allow_pickle=True)
    K = data["intrinsics"]
    dist = data["dist_coeffs"]
    return K, dist

def main():
    templates = load_templates()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se ha podido abrir la cámara / vídeo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    ret, frame0 = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        cap.release()
        return

    h, w = frame0.shape[:2]

    K, dist = load_calibration(CALIB_PATH)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0, newImgSize=(w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        OUT_VIDEO_PATH,
        fourcc,
        fps,
        (w, h)
    )

    thr = 0.45

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_und = cv2.undistort(frame, K, dist, None, newK)

        label, score, dbg = detect_letter(frame_und, templates, thr=thr)

        out = frame_und.copy()
        txt = f"Letra: {label if label else '-'}  score: {score:.3f}  thr:{thr:.2f}"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)

        out_writer.write(out)
        cv2.imshow("Letter classification (undistorted)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("]"):
            thr = min(0.95, thr + 0.02)
        if key == ord("["):
            thr = max(0.10, thr - 0.02)

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
