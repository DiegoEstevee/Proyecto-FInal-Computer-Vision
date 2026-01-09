# Proyecto Final – Computer Vision

Proyecto de Visión por Ordenador en tiempo real con OpenCV que integra dos módulos independientes:

- Sistema de seguridad mediante contraseña visual.
- Sistema de análisis de ping-pong con detección y seguimiento de la pelota, detección de botes y cálculo automático de puntos.

Todos los resultados se guardan en la carpeta results.

---

## Estructura del repositorio

- data  
  - videos  
  - templates  
  - calibration  
- src  
  - calibration  
  - security  
  - ping_pong  
- results  
- requirements.txt  
- README.md  

---

## Requisitos

Instalar las dependencias (preferiblemente en un entorno virtual):

pip install -r requirements.txt

Todos los scripts deben ejecutarse desde la raíz del repositorio.

---

## PARTE 1 — Contraseña visual

### Archivos necesarios
- Vídeo: data/videos/password.mp4  
- Templates: data/templates/tmpl_A.png, tmpl_B.png, tmpl_C.png  

---

### 1. Calibración de la cámara

Ejecutar:  
python src/calibration/calibration_camara.py

Genera:  
src/calibration/calibration_data.npz

---

### 2. Detector de letras (opcional)

Ejecutar:  
python src/security/run_letter_detector_live.py

Genera:  
results/letter_detector_output.mp4

---

### 3. Sistema de contraseña completo

Ejecutar:  
python src/security/run_letter_password_video.py

Genera:  
results/password_output.mp4

Controles:  
- q → salir  
- r → resetear la contraseña  

---

## PARTE 2 — Análisis de Ping-Pong

Esta parte es independiente del sistema de contraseña.

### Archivos necesarios
- data/videos/video_1.mp4  
- data/videos/video_2.mp4  

---

### 1. Calibración de la mesa

Ejecutar:  
python src/ping_pong/00_calibrate_table.py

Seleccionar con el ratón las 4 esquinas de la mesa y pulsar ENTER.

Genera:  
data/calibration/table_homography.npz

---

### 2. Ajuste HSV de la pelota

Ejecutar:  
python src/ping_pong/01_hsv_tune_ball.py

Genera:  
results/01_hsv_tune_ball_output.mp4

---

### 3. Seguimiento de la pelota

Ejecutar:  
python src/ping_pong/02_track_ball.py

Genera:  
results/02_track_ball_output.mp4

---

### 4. Detección de botes y puntuación

Ejecutar:  
python src/ping_pong/03_score_bounces.py

Genera:  
results/03_score_bounces_output.mp4

---

Proyecto realizado como Proyecto Final de Computer Vision utilizando OpenCV.
