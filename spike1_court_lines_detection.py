#!/usr/bin/env python3
"""
Spike 1 - Detección de Bordes de Cancha por Líneas

Nuevo enfoque:
1. Detectar las líneas de la cancha (bordes del rectángulo)
2. Encontrar las 4 esquinas de la cancha
3. Crear un polígono con la zona de juego
4. Filtrar detecciones que estén dentro del polígono

Ventajas:
- No depende del color del piso
- Funciona aunque jugadores tapen partes del piso
- Detecta el rectángulo real de juego
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_court_corners(frame):
    """
    Detecta las 4 esquinas de la cancha usando:
    1. Detección de líneas verdes/amarillas/blancas (líneas de cancha)
    2. Intersección de líneas para encontrar esquinas
    
    Retorna las 4 esquinas en orden: TL, TR, BR, BL
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detectar líneas blancas/amarillas/verdes claras
    # Las líneas de cancha suelen ser más claras que el piso
    
    # Blanco
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Amarillo
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combinar
    lines_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Aplicar Canny para detección de bordes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Usar la máscara de líneas para realzar bordes
    masked_gray = cv2.bitwise_and(gray, gray, mask=lines_mask)
    
    # Canny
    edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
    
    # Detectar líneas con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=100, maxLineGap=30)
    
    if lines is None or len(lines) < 4:
        # Fallback: usar todo el frame como cancha
        h, w = frame.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # Separar líneas horizontales y verticales
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        
        if angle < np.pi/6:  # Horizontal
            horizontal_lines.append((x1, y1, x2, y2))
        elif angle > np.pi/3:  # Vertical
            vertical_lines.append((x1, y1, x2, y2))
    
    # Encontrar las 4 líneas más extremas
    h, w = frame.shape[:2]
    
    # Líneas horizontales: top y bottom
    top_y = h
    bottom_y = 0
    
    for x1, y1, x2, y2 in horizontal_lines:
        avg_y = (y1 + y2) / 2
        if avg_y < h/2 and avg_y < top_y:
            top_y = avg_y
        elif avg_y > h/2 and avg_y > bottom_y:
            bottom_y = avg_y
    
    # Líneas verticales: left y right
    left_x = w
    right_x = 0
    
    for x1, y1, x2, y2 in vertical_lines:
        avg_x = (x1 + x2) / 2
        if avg_x < w/2 and avg_x < left_x:
            left_x = avg_x
        elif avg_x > w/2 and avg_x > right_x:
            right_x = avg_x
    
    # Si no encontramos suficientes líneas, usar valores por defecto
    if top_y == h:
        top_y = int(h * 0.1)
    if bottom_y == 0:
        bottom_y = int(h * 0.9)
    if left_x == w:
        left_x = int(w * 0.15)
    if right_x == 0:
        right_x = int(w * 0.85)
    
    # Construir las 4 esquinas
    corners = np.array([
        [left_x, top_y],      # Top-left
        [right_x, top_y],     # Top-right
        [right_x, bottom_y],  # Bottom-right
        [left_x, bottom_y]    # Bottom-left
    ], dtype=np.float32)
    
    return corners


def point_in_polygon(point, polygon):
    """
    Verifica si un punto está dentro de un polígono.
    """
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def bbox_in_polygon(bbox, polygon, threshold=0.3):
    """
    Verifica si un porcentaje del bounding box está dentro del polígono.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Verificar el centro del bbox
    center_inside = point_in_polygon((cx, cy), polygon)
    
    # Verificar las esquinas del bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    inside = sum(1 for c in corners if point_in_polygon(c, polygon))
    
    # Está en cancha si el centro está dentro o al menos 2 esquinas
    return center_inside or inside >= 2, inside / 4


def analyze_with_lines(video_path: str, max_frames: int = 100):
    """
    Analiza el video usando detección de líneas de cancha.
    """
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    # Crear directorio para resultados
    os.makedirs("runs/lines_analysis", exist_ok=True)
    
    all_detections = []
    court_counts = []
    
    # Detectar esquinas en el primer frame y usarlas para todos
    ret, first_frame = cap.read()
    if ret:
        corners = detect_court_corners(first_frame)
        print(f"\n📐 Esquinas detectadas:")
        print(f"   TL: ({corners[0][0]:.0f}, {corners[0][1]:.0f})")
        print(f"   TR: ({corners[1][0]:.0f}, {corners[1][1]:.0f})")
        print(f"   BR: ({corners[2][0]:.0f}, {corners[2][1]:.0f})")
        print(f"   BL: ({corners[3][0]:.0f}, {corners[3][1]:.0f})")
        
        # Crear polígono
        polygon = corners.reshape((-1, 1, 2)).astype(np.int32)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"\n🔍 Analizando {max_frames} frames...")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar personas con YOLO
        results = model(frame, classes=[0], verbose=False)
        
        frame_detections = []
        court_count = 0
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:  # persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Verificar si está en la cancha
                    in_court, corners_inside = bbox_in_polygon((x1, y1, x2, y2), polygon)
                    
                    detection = {
                        'frame': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'confidence': round(conf, 3),
                        'in_court': in_court,
                        'corners_inside': corners_inside
                    }
                    
                    frame_detections.append(detection)
                    
                    if in_court:
                        court_count += 1
        
        all_detections.extend(frame_detections)
        court_counts.append(court_count)
        
        # Guardar frames de muestra
        if frame_idx % 25 == 0:
            sample_frame = frame.copy()
            
            # Dibujar polígono de cancha
            cv2.polylines(sample_frame, [polygon], True, (255, 0, 0), 3)
            
            # Dibujar detecciones
            for det in frame_detections:
                x1, y1, x2, y2 = det['bbox']
                color = (0, 255, 0) if det['in_court'] else (0, 0, 255)
                cv2.rectangle(sample_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(sample_frame, (det['center'][0], det['center'][1]), 5, color, -1)
            
            cv2.imwrite(f"runs/lines_analysis/frame_{frame_idx:04d}.jpg", sample_frame)
            
            print(f"   Frame {frame_idx}: {court_count} en cancha de {len(frame_detections)}")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS CON DETECCIÓN DE LÍNEAS")
    print(f"{'='*60}")
    
    total_detections = len(all_detections)
    court_detections = sum(1 for d in all_detections if d['in_court'])
    
    print(f"\n📍 Clasificación:")
    print(f"   Total detecciones: {total_detections}")
    print(f"   Detecciones EN CANCHA: {court_detections} ({100*court_detections/total_detections:.1f}%)")
    print(f"   Detecciones FUERA: {total_detections - court_detections}")
    
    avg_court = np.mean(court_counts)
    print(f"\n📈 Promedio por frame:")
    print(f"   Personas en cancha: {avg_court:.2f}/frame")
    
    frames_with_4 = sum(1 for c in court_counts if c == 4)
    frames_with_3_5 = sum(1 for c in court_counts if 3 <= c <= 5)
    
    print(f"\n🎯 Resultados:")
    print(f"   Frames con 4 en cancha: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 en cancha: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    
    # Veredicto
    print(f"\n{'='*60}")
    if frames_with_4 / max_frames >= 0.85:
        print("✅ FILTRADO POR LÍNEAS EXITOSO")
    elif frames_with_3_5 / max_frames >= 0.85:
        print("⚠️  FILTRADO PARCIAL")
    else:
        print("❌ Requiere ajuste de detección de líneas")
    print(f"{'='*60}")
    
    print(f"\n📁 Archivos guardados en runs/lines_analysis/")
    
    return {
        'total_detections': total_detections,
        'court_detections': court_detections,
        'avg_court_per_frame': round(avg_court, 2),
        'frames_with_4': frames_with_4,
        'frames_with_3_5': frames_with_3_5,
        'corners': corners.tolist()
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Detección de Bordes de Cancha\n")
    results = analyze_with_lines(video_path, max_frames=100)