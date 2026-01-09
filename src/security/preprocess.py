import os
import cv2
import numpy as np
import argparse

TEMPLATE_SIZE = 200

def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img

def binarize_letter(gray: np.ndarray) -> np.ndarray:
    
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    return th

def crop_to_main_component(bin_img: np.ndarray) -> np.ndarray:
    
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bin_img
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # padding alrededor
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(bin_img.shape[1], x + w + pad)
    y2 = min(bin_img.shape[0], y + h + pad)
    return bin_img[y1:y2, x1:x2]

def resize_with_padding(bin_img: np.ndarray, out_size: int = TEMPLATE_SIZE, fill=0) -> np.ndarray:
    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    
    scale = (out_size * 0.85) / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    nh, nw = max(10, nh), max(10, nw)

    resized = cv2.resize(bin_img, (nw, nh), interpolation=cv2.INTER_NEAREST)

    out = np.full((out_size, out_size), fill, dtype=np.uint8)
    y0 = (out_size - nh) // 2
    x0 = (out_size - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out

def make_template_from_file(src_path: str) -> np.ndarray:
    gray = load_gray(src_path)
    if gray is None:
        raise FileNotFoundError(f"No pude leer: {src_path}")

    th = binarize_letter(gray)
    crop = crop_to_main_component(th)
    tmpl = resize_with_padding(crop, TEMPLATE_SIZE, fill=0)

    
    tmpl = (tmpl > 127).astype(np.uint8) * 255
    return tmpl

def find_source_for_label(src_dir: str, label: str):
    # intenta varias extensiones comunes
    exts = [".png", ".jpg", ".jpeg"]
    for e in exts:
        p = os.path.join(src_dir, f"{label}{e}")
        if os.path.exists(p):
            return p
        p2 = os.path.join(src_dir, f"{label.lower()}{e}")
        if os.path.exists(p2):
            return p2
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, default=".", help="Carpeta donde están A/B/C fuente")
    ap.add_argument("--out_dir", type=str, default=".", help="Carpeta de salida")
    ap.add_argument("--labels", type=str, default="ABCD", help="Etiquetas a generar")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for lab in list(args.labels):
        src = find_source_for_label(args.src_dir, lab)
        if src is None:
            print(f"No encuentro imagen fuente para '{lab}' en {args.src_dir}")
            continue

        tmpl = make_template_from_file(src)
        out_path = os.path.join(args.out_dir, f"tmpl_{lab}.png")
        cv2.imwrite(out_path, tmpl)
        print(f"Guardado template: {out_path}  (desde {src})")

    
    # cv2.imshow("tmpl", tmpl); cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
