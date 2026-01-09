import cv2
import numpy as np
from collections import deque, Counter

from letter_classifier import load_templates, detect_letter

VIDEO_PATH = "data/videos/password.mp4"


PASSWORD = ["A", "B", "C"]

# --- Calibración  ---
CALIB_PATH = "src/calibration/calibration_data.npz"
USE_CALIB = True  

# --- Parámetros del nuevo decodificador robusto ---
WIN = 21                 # ventana de frames para voto 
DOM_RATIO = 0.55         # mínimo para considerar dominante 
LOCK_FRAMES = 10         # cuántos frames bloqueamos tras aceptar una letra
MISMATCH_TOL = 2         # mismatches permitidos antes de reset 

def draw_overlay(frame, current_label, score: float, thr: float,
                 progress: int, status: str, seq_str: str):
    out = frame

    detected_txt = f"Actual: {current_label if current_label else '(ninguno)'}  score:{score:.3f} thr:{thr:.2f}"
    cv2.putText(out, detected_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, detected_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    prog_txt = f"Progreso: {progress}/{len(PASSWORD)}"
    cv2.putText(out, prog_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, prog_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    seq_txt = f"Secuencia: {seq_str}"
    cv2.putText(out, seq_txt, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, seq_txt, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    st_txt = f"Estado: {status}"
    cv2.putText(out, st_txt, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, st_txt, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    help_txt = "'r': reset | 'q': salir"
    cv2.putText(out, help_txt, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, help_txt, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    return out

def seq_progress_str(progress: int):
    return " ".join([PASSWORD[i] if i < progress else "_" for i in range(len(PASSWORD))])

def dominant_label(win: deque):
    
    vals = [x for x in win if x is not None]
    if len(vals) == 0:
        return None, 0.0
    c = Counter(vals)
    lab, cnt = c.most_common(1)[0]
    ratio = cnt / max(1, len(vals))
    return lab, ratio

def main():
    templates = load_templates()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se pudo abrir el video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0
    delay_ms = max(1, int(1000.0 / fps))

    # --- calibración ---
    K = dist = newK = None
    if USE_CALIB:
        data = np.load(CALIB_PATH, allow_pickle=True)
        K = data["intrinsics"]
        dist = data["dist_coeffs"]

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            ret0, frame0 = cap.read()
            if not ret0:
                print("No se pudo leer primer frame para tamaño.")
                cap.release()
                return
            h, w = frame0.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1, newImgSize=(w, h))

    thr = 0.45
    print("Password:", " - ".join(PASSWORD))
    

    # --- estado del nuevo decoder ---
    window = deque(maxlen=WIN)
    progress = 0
    status = "WAIT_DOMINANT"
    last_accepted = None
    lock = 0
    mismatch_strikes = 0
    unlocked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if USE_CALIB:
            frame_u = cv2.undistort(frame, K, dist, None, newK)
        else:
            frame_u = frame

        label, score, dbg = detect_letter(frame_u, templates, thr=thr)

        # guardamos label en ventana
        window.append(label)

        # locking tras aceptar 
        if lock > 0:
            lock -= 1
            status = f"LOCK({lock})"
        else:
            dom, ratio = dominant_label(window)

            if dom is None or ratio < DOM_RATIO:
                status = f"WAIT_DOMINANT ({ratio:.2f})"
            else:
                expected = PASSWORD[progress] if progress < len(PASSWORD) else None

                # no aceptar la misma letra otra vez sin cambiar
                if dom == last_accepted:
                    status = f"HOLD_SAME ({dom} {ratio:.2f})"
                else:
                    
                    if expected is None:
                        unlocked = True
                        status = "DONE"
                    else:
                        
                        if dom == expected:
                            progress += 1
                            last_accepted = dom
                            lock = LOCK_FRAMES
                            mismatch_strikes = 0
                            status = f"ACCEPT {dom} ({ratio:.2f})"

                            if progress >= len(PASSWORD):
                                unlocked = True
                                status = "UNLOCKED"
                        else:
                            # dominante distinta de la esperada
                            mismatch_strikes += 1
                            status = f"MISMATCH dom={dom} exp={expected} ({ratio:.2f}) strike {mismatch_strikes}/{MISMATCH_TOL}"

                            # solo resetea si se repite varias veces 
                            if mismatch_strikes > MISMATCH_TOL:
                                progress = 0
                                last_accepted = None
                                lock = LOCK_FRAMES
                                mismatch_strikes = 0
                                status = "RESET (stable mismatch)"

        vis = frame_u.copy()
        vis = draw_overlay(
            vis,
            current_label=label,
            score=score,
            thr=thr,
            progress=progress,
            status=status,
            seq_str=seq_progress_str(progress),
        )

        if unlocked:
            cv2.putText(vis, "CONTRASENA CORRECTA", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, "CONTRASENA CORRECTA", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Password letters (Video)", vis)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            progress = 0
            last_accepted = None
            lock = 0
            mismatch_strikes = 0
            unlocked = False
            window.clear()
            status = "RESET"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

