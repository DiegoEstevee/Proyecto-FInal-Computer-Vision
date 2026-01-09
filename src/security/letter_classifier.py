import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from letter_detector import detect_paper_and_norm

TEMPLATE_SIZE = 200

def load_templates():
    tmpls = {}
    for lab in ["A", "B", "C"]:
        path = f"data/templates/tmpl_{lab}.png"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        tmpls[lab] = img
    return tmpls

def match_letter(norm: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
    best_lab = None
    best_score = -1.0
    for lab, tmpl in templates.items():
        res = cv2.matchTemplate(norm, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        score = float(max_val)
        if score > best_score:
            best_score = score
            best_lab = lab
    return best_lab, best_score

def detect_letter(frame_bgr: np.ndarray, templates: Dict[str, np.ndarray], thr: float = 0.45):
    norm, dbg = detect_paper_and_norm(frame_bgr)
    if norm is None:
        return None, 0.0, dbg

    lab, score = match_letter(norm, templates)
    if score < thr:
        return None, score, dbg
    return lab, score, dbg

