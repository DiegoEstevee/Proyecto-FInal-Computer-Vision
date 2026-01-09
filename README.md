# Proyecto Final – Computer Vision

Proyecto de Visión por Ordenador en tiempo real con OpenCV que integra dos módulos independientes:

- Sistema de seguridad mediante contraseña visual.
- Sistema de análisis de ping-pong con detección y seguimiento de la pelota, detección de botes y cálculo automático de puntos.

Todos los resultados se guardan en la carpeta results.


## Antes de empezar
- Cabe a destacar que no hemos hecho un video explicativo ya que al no hacer un sistema con camara en tiempo real, no tenia sentido hacer el video por lo que a continuación viene la explicación.
  - Primera parte:
    - El video de demostración se encuentra en la carpeta de data/videos en la que se puede ver un video en el que se alternan diferentes letras, al aplicar nuestro sistema de seguridad, este detectará el patron y se activará el       sistema.
  - Segunda parte:
    - El video de demostración también se encuentra en la carpeta de data/videos en la que se puede ver una partida de ping_pong en la que gana el jugador del fondo, aplicando nuestro sistema veremos que este acierta y da como ganador al jugador del fondo.

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
- report  
  - Informe Trabajo Computer Vision 
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

---

### 2. Preprocesamiento

Ejecutar:  
python src/calibration/preprocess.py

Genera:  
data\templates\tmpl_A.png
data\templates\tmpl_B.png
data\templates\tmpl_C.png

---

### 3. Detector de letras (opcional)

Ejecutar:  
python src/security/run_letter_detector_live.py

Genera:  
results/letter_detector_output.mp4

---

### 4. Sistema de contraseña completo

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

Dentro de la carpeta report se encuentra el informe de este proyetco en que se detalla una introducción, la metodología y los resultados.
