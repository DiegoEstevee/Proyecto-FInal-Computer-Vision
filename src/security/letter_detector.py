import cv2
import numpy as np
from typing import Optional, Dict, Tuple

TEMPLATE_SIZE = 200
WARP_W, WARP_H = 1000, 707  

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_paper_quad(frame_bgr: np.ndarray, min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
    H, W = frame_bgr.shape[:2]

    # Segmentación de blanco en HSV 
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (179, 70, 255)) 

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area_ratio * (H * W):
        return None

    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return order_points(approx)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_points(box)

def warp_paper(frame_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))


def extract_letter_norm_from_warp(warped_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    dbg: Dict[str, np.ndarray] = {}

    H, W = warped_bgr.shape[:2]

    # ROI fija central (evita contorno mayor del frame completo)
    y1, y2 = int(0.12 * H), int(0.95 * H)
    x1, x2 = int(0.20 * W), int(0.80 * W)
    roi = warped_bgr[y1:y2, x1:x2].copy()
    dbg["roi"] = roi

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarización 
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    dbg["th"] = th

    norm, dbg_norm = normalize_to_square(th, TEMPLATE_SIZE)
    dbg.update(dbg_norm)

    return norm, dbg

def detect_paper_and_norm(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    dbg: Dict[str, np.ndarray] = {}

    quad = find_paper_quad(frame_bgr)
    if quad is None:
        return None, dbg

    warped = warp_paper(frame_bgr, quad)
    dbg["warped"] = warped

    norm, dbg2 = extract_letter_norm_from_warp(warped)
    dbg.update(dbg2)
    return norm, dbg

def draw_quad(frame_bgr: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()
    if quad is None:
        cv2.putText(out, "NO PAPER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return out
    q = quad.astype(int).reshape(-1, 2)
    cv2.polylines(out, [q], True, (0, 255, 255), 3)
    return out



TEMPLATE_SIZE = 200

def remove_border_white_components(bin_img: np.ndarray) -> np.ndarray:
    """Elimina componentes blancas conectadas al borde (bin_img 0/255)."""
    h, w = bin_img.shape[:2]
    out = bin_img.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # floodfill desde bordes donde haya blanco
    for x in range(w):
        if out[0, x] == 255:
            cv2.floodFill(out, mask, (x, 0), 0)
        if out[h - 1, x] == 255:
            cv2.floodFill(out, mask, (x, h - 1), 0)
    for y in range(h):
        if out[y, 0] == 255:
            cv2.floodFill(out, mask, (0, y), 0)
        if out[y, w - 1] == 255:
            cv2.floodFill(out, mask, (w - 1, y), 0)

    return out

def normalize_to_square(bin_img: np.ndarray, out_size: int = TEMPLATE_SIZE):
    """
    Selecciona el componente más probable de ser letra (CC stats),
    recorta y hace resize con padding. Devuelve (norm, dbg).
    """
    dbg: Dict[str, np.ndarray] = {}

    
    b = (bin_img > 127).astype(np.uint8) * 255

    # Quita ruido conectada al borde
    cleaned = remove_border_white_components(b)
    dbg["cleaned_for_norm"] = cleaned

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if n_labels <= 1:
        return None, dbg

    h, w = cleaned.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    img_area = h * w

    best_i = -1
    best_score = -1e18

    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        cxi, cyi = centroids[i]

        # filtros de área (ajusta si la letra es pequeña/grande)
        if area < 1200:
            continue
        if area > 0.55 * img_area:
            continue

        # evita franjas
        ar = bw / float(bh + 1e-6)
        if ar > 5.0 or ar < 0.20:
            continue

        # evita componentes pegadas al borde
        m = 3
        if x <= m or y <= m or (x + bw) >= (w - m) or (y + bh) >= (h - m):
            continue

        # score: preferir grande y centrado
        dist = (cxi - cx0) ** 2 + (cyi - cy0) ** 2
        score = float(area) - 0.02 * dist

        if score > best_score:
            best_score = score
            best_i = i

    if best_i == -1:
        return None, dbg

    x, y, bw, bh, area = stats[best_i]

    # máscara del componente
    comp = (labels == best_i).astype(np.uint8) * 255
    dbg["component_chosen"] = comp

    # recorte con un pequeño padding
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    crop = comp[y1:y2, x1:x2]
    dbg["crop"] = crop

    # resize manteniendo aspecto
    ch, cw = crop.shape[:2]
    scale = (out_size * 0.85) / max(ch, cw)
    nh, nw = int(ch * scale), int(cw * scale)

    # evita que se convierta en línea por redondeo
    nh = max(10, nh)
    nw = max(10, nw)

    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)

    out = np.zeros((out_size, out_size), dtype=np.uint8)
    y0 = (out_size - nh) // 2
    x0 = (out_size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    dbg["norm"] = out

    return out, dbg
