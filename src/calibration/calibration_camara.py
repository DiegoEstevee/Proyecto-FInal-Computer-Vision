import cv2
import numpy as np
import glob
import os
from typing import List, Tuple

# =========================
# CONFIG
# =========================

IMAGES_GLOB = "data/calibration/*.jpeg"
OUTPUT_NPZ = "src/calibration/calibration_data.npz"

RESIZE_TO = (1080, 810)         
CHESSBOARD_SHAPE = (7, 9)       
SQUARE_SIZE = 0.030             


# =========================
# Helpers
# =========================
def load_images(filenames: List[str], resize_to=None) -> List[np.ndarray]:
    imgs = []
    for f in filenames:
        img = cv2.imread(f)
        if img is None:
            print("Error al cargar:", f)
            continue
        if resize_to is not None:
            img = cv2.resize(img, resize_to)  
        imgs.append(img)
    return imgs

def get_chessboard_points(chessboard_shape: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = chessboard_shape  
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  
    objp[:, :2] = grid * square_size
    return objp

# =========================
# Main calibration
# =========================
def main():
    files = sorted(glob.glob(IMAGES_GLOB))
    if len(files) == 0:
        raise RuntimeError(f"No se encontraron imágenes con el patrón: {IMAGES_GLOB}")

    imgs = load_images(files, resize_to=RESIZE_TO)
    if len(imgs) == 0:
        raise RuntimeError("No se pudo cargar ninguna imagen.")

    
    h, w = imgs[0].shape[:2]
    img_size = (w, h)

    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    objp_one = get_chessboard_points(CHESSBOARD_SHAPE, SQUARE_SIZE).astype(np.float32)

    objpoints = []  
    imgpoints = []  

    valid_count = 0
    for idx, img in enumerate(imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE)
        if not ret:
            print(f" No detectado tablero en: {files[idx]}")
            continue

        corners_ref = cv2.cornerSubPix(gray, corners, (7, 9), (-1, -1), criteria)
        if corners_ref is None:
            corners_ref = corners

        objpoints.append(objp_one.copy())
        imgpoints.append(corners_ref.astype(np.float32))
        valid_count += 1

    if valid_count < 3:
        raise RuntimeError(f"Demasiadas pocas detecciones válidas: {valid_count}. Necesitas varias vistas del tablero.")

    # Calibrate
    camera_matrix_init = None
    dist_coeffs_init = None

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_coeffs_init
    )

    # Extrinsics (R|t)
    extrinsics = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        Rt = np.hstack((R, tvec.reshape(3, 1)))  # 3x4
        extrinsics.append(Rt)
    extrinsics = np.array(extrinsics, dtype=np.float32)

    # Save
    np.savez(
        OUTPUT_NPZ,
        intrinsics=K,
        dist_coeffs=dist,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        extrinsics=extrinsics,
        rms=rms,
        img_size=np.array(img_size, dtype=np.int32),
        chessboard_shape=np.array(CHESSBOARD_SHAPE, dtype=np.int32),
        square_size=np.array([SQUARE_SIZE], dtype=np.float32)
    )

    print("\n Calibración completada")
    print("Imágenes válidas:", valid_count, "/", len(imgs))
    print("RMS:", rms)
    print("K (intrínsecos):\n", K)
    print("dist:\n", dist.ravel())
    print(f" Guardado en: {OUTPUT_NPZ}")

if __name__ == "__main__":
    main()

