#!/usr/bin/env python3
"""
Spike 3 - Detección de Pelota con YOLO (Sports Ball)

Usa YOLO con la clase 37 (sports ball) del dataset COCO para detectar
la pelota de pádel. Compara con detección por color HSV.

Ejecución:
    python spike3_ball_yolo.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# Configuración
VIDEO_PATH = "test-videos/ProPadel2.mp4"
CORNERS_PATH = "runs/court_propadel2/court_corners.json"
OUTPUT_DIR = "runs/spike3_ball_yolo"
MAX_FRAMES = 300  # ~10 segundos a 30fps

# Clase 37 en COCO = sports ball
SPORTS_BALL_CLASS = 37

# Parámetros de la pelota
BALL_RADIUS_RANGE_PX = (3, 20)  # Rango esperado en píxeles


@dataclass
class BallDetection:
    """Detección de pelota en un frame."""
    frame_idx: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    source: str  # 'yolo' o 'hsv'


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    pt = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula distancia euclidiana entre dos puntos."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def detect_ball_yolo(model: YOLO, frame: np.ndarray, polygon: np.ndarray, conf_threshold: float = 0.3) -> List[BallDetection]:
    """
    Detecta la pelota usando YOLO con clase sports ball.
    
    Args:
        model: Modelo YOLO cargado
        frame: Frame del video
        polygon: Polígono de la cancha
        conf_threshold: Umbral de confianza mínimo
    
    Returns:
        Lista de detecciones de pelota
    """
    detections = []
    
    # Ejecutar YOLO solo para la clase sports ball
    results = model(frame, classes=[SPORTS_BALL_CLASS], conf=conf_threshold, verbose=False)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Centro del bounding box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Ancho y alto
                w = x2 - x1
                h = y2 - y1
                
                # Verificar que esté dentro de la cancha o cerca
                in_court = point_in_polygon((cx, cy), polygon)
                
                # Verificar tamaño razonable para una pelota
                radius_estimate = min(w, h) / 2
                size_ok = BALL_RADIUS_RANGE_PX[0] <= radius_estimate <= BALL_RADIUS_RANGE_PX[1]
                
                # Agregar detección (incluso si está fuera, para análisis)
                detections.append(BallDetection(
                    frame_idx=0,  # Se asignará después
                    x=cx,
                    y=cy,
                    w=w,
                    h=h,
                    conf=conf,
                    source='yolo'
                ))
    
    return detections


def detect_ball_hsv(frame: np.ndarray, polygon: np.ndarray) -> List[BallDetection]:
    """
    Detecta la pelota por color HSV (método anterior).
    
    Args:
        frame: Frame del video
        polygon: Polígono de la cancha
    
    Returns:
        Lista de detecciones de pelota
    """
    detections = []
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rango de color para pelota amarilla/verde
    lower = np.array([20, 100, 100])
    upper = np.array([50, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        if BALL_RADIUS_RANGE_PX[0] <= radius <= BALL_RADIUS_RANGE_PX[1]:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.4:
                    detections.append(BallDetection(
                        frame_idx=0,
                        x=x,
                        y=y,
                        w=radius * 2,
                        h=radius * 2,
                        conf=circularity,  # Usamos circularidad como "confianza"
                        source='hsv'
                    ))
    
    return detections


def run_spike3_yolo():
    """Ejecuta la prueba de detección de pelota con YOLO."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Cargar esquinas de la cancha
    print(f"📐 Cargando configuración de cancha...")
    with open(CORNERS_PATH, 'r') as f:
        corners = json.load(f)
    
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    print(f"   TL: {corners['TL']}")
    print(f"   TR: {corners['TR']}")
    print(f"   BR: {corners['BR']}")
    print(f"   BL: {corners['BL']}")
    
    # Cargar modelo YOLO
    print(f"\n🔄 Cargando modelo YOLO...")
    model = YOLO('yolov8m.pt')
    print(f"   Modelo cargado: yolov8m.pt")
    print(f"   Clase objetivo: {SPORTS_BALL_CLASS} (sports ball)")
    
    # Abrir video
    print(f"\n📹 Abriendo video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames a analizar: {MAX_FRAMES}")
    
    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{OUTPUT_DIR}/ball_detection_yolo.mp4", fourcc, fps, (width, height))
    
    # Métricas
    metrics = {
        'yolo_detections': 0,
        'hsv_detections': 0,
        'yolo_in_court': 0,
        'hsv_in_court': 0,
        'frames_with_yolo': 0,
        'frames_with_hsv': 0,
        'yolo_positions': [],
        'hsv_positions': []
    }
    
    frame_count = 0
    
    print(f"\n🔍 Analizando {MAX_FRAMES} frames...")
    print(f"{'='*60}")
    
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar cancha
        cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
        
        # Detectar con YOLO
        yolo_detections = detect_ball_yolo(model, frame, polygon, conf_threshold=0.25)
        
        # Detectar con HSV
        hsv_detections = detect_ball_hsv(frame, polygon)
        
        # Actualizar métricas
        if yolo_detections:
            metrics['frames_with_yolo'] += 1
            for det in yolo_detections:
                metrics['yolo_detections'] += 1
                metrics['yolo_positions'].append((frame_count, det.x, det.y, det.conf))
                if point_in_polygon((det.x, det.y), polygon):
                    metrics['yolo_in_court'] += 1
        
        if hsv_detections:
            metrics['frames_with_hsv'] += 1
            for det in hsv_detections:
                metrics['hsv_detections'] += 1
                metrics['hsv_positions'].append((frame_count, det.x, det.y, det.conf))
                if point_in_polygon((det.x, det.y), polygon):
                    metrics['hsv_in_court'] += 1
        
        # Dibujar detecciones YOLO (verde)
        for det in yolo_detections:
            x, y, w, h = det.x, det.y, det.w, det.h
            color = (0, 255, 0) if point_in_polygon((x, y), polygon) else (0, 128, 0)
            cv2.rectangle(frame, 
                         (int(x - w/2), int(y - h/2)),
                         (int(x + w/2), int(y + h/2)),
                         color, 2)
            cv2.putText(frame, f"YOLO {det.conf:.2f}", 
                       (int(x - w/2), int(y - h/2) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Dibujar detecciones HSV (naranja)
        for det in hsv_detections:
            x, y, r = det.x, det.y, det.w / 2
            color = (0, 165, 255) if point_in_polygon((x, y), polygon) else (0, 100, 200)
            cv2.circle(frame, (int(x), int(y)), int(r), color, 2)
            cv2.putText(frame, f"HSV {det.conf:.2f}", 
                       (int(x), int(y) - int(r) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Panel de información
        y_offset = 30
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"YOLO: {len(yolo_detections)} detections", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        y_offset += 20
        cv2.putText(frame, f"HSV: {len(hsv_detections)} detections", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        out.write(frame)
        
        if frame_count % 50 == 0:
            print(f"   Frame {frame_count}: YOLO={len(yolo_detections)}, HSV={len(hsv_detections)}")
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Calcular estadísticas finales
    print(f"\n{'='*60}")
    print("📊 RESULTADOS DE DETECCIÓN DE PELOTA")
    print(f"{'='*60}")
    
    print(f"\n🎾 YOLO (sports ball class):")
    print(f"   Total detecciones: {metrics['yolo_detections']}")
    print(f"   Detecciones en cancha: {metrics['yolo_in_court']}")
    print(f"   Frames con detección: {metrics['frames_with_yolo']}/{frame_count} ({100*metrics['frames_with_yolo']/frame_count:.1f}%)")
    
    print(f"\n🎨 HSV (color amarillo/verde):")
    print(f"   Total detecciones: {metrics['hsv_detections']}")
    print(f"   Detecciones en cancha: {metrics['hsv_in_court']}")
    print(f"   Frames con detección: {metrics['frames_with_hsv']}/{frame_count} ({100*metrics['frames_with_hsv']/frame_count:.1f}%)")
    
    # Evaluar cuál método es mejor
    yolo_rate = metrics['frames_with_yolo'] / frame_count * 100
    hsv_rate = metrics['frames_with_hsv'] / frame_count * 100
    
    print(f"\n📈 Conclusión:")
    if yolo_rate > hsv_rate:
        print(f"   ✅ YOLO tiene mejor tasa de detección ({yolo_rate:.1f}% vs {hsv_rate:.1f}%)")
    elif hsv_rate > yolo_rate:
        print(f"   ✅ HSV tiene mejor tasa de detección ({hsv_rate:.1f}% vs {yolo_rate:.1f}%)")
    else:
        print(f"   ⚖️ Ambos métodos tienen tasa similar ({yolo_rate:.1f}%)")
    
    # Guardar métricas
    output_metrics = {
        'video_path': VIDEO_PATH,
        'frames_analyzed': frame_count,
        'fps': fps,
        'yolo': {
            'total_detections': metrics['yolo_detections'],
            'in_court': metrics['yolo_in_court'],
            'frames_with_detection': metrics['frames_with_yolo'],
            'detection_rate': yolo_rate,
            'positions': [(int(f), float(x), float(y), float(c)) for f, x, y, c in metrics['yolo_positions'][:100]]
        },
        'hsv': {
            'total_detections': metrics['hsv_detections'],
            'in_court': metrics['hsv_in_court'],
            'frames_with_detection': metrics['frames_with_hsv'],
            'detection_rate': hsv_rate,
            'positions': [(int(f), float(x), float(y), float(c)) for f, x, y, c in metrics['hsv_positions'][:100]]
        }
    }
    
    with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    print(f"\n✅ Video guardado en: {OUTPUT_DIR}/ball_detection_yolo.mp4")
    print(f"📁 Métricas guardadas en: {OUTPUT_DIR}/metrics.json")
    
    return output_metrics


if __name__ == "__main__":
    print("🎾 Spike 3 - Detección de Pelota con YOLO")
    print("="*60)
    print("\nComparando YOLO (sports ball) vs detección por color HSV\n")
    
    results = run_spike3_yolo()