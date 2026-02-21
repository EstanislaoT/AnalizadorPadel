#!/usr/bin/env python3
"""
Spike 1 - Detección de Cancha con Perspectiva

Mejoras:
1. Detectar las 4 esquinas como polígono arbitrario (trapezoide/perspectiva)
2. Agregar filtro de confianza >= 0.5
3. Tomar las 4 detecciones con mayor confianza dentro del polígono

La cancha de pádel vista desde un ángulo será un trapezoide, no un rectángulo.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_court_polygon(frame):
    """
    Detecta las 4 esquinas de la cancha considerando la perspectiva.
    
    Retorna un polígono de 4 vértices (puede ser trapezoide).
    """
    h, w = frame.shape[:2]
    
    # Convertir a HSV para detectar líneas de cancha
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detectar líneas claras (blancas/amarillas)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    lines_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Detectar bordes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Combinar con máscara de líneas
    combined = cv2.bitwise_and(edges, edges, mask=lines_mask)
    
    # Detectar líneas con Hough
    lines = cv2.HoughLinesP(combined, 1, np.pi/180, threshold=50,
                            minLineLength=80, maxLineGap=20)
    
    if lines is None or len(lines) < 4:
        # Fallback: usar todo el frame centrado
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.10)
        return np.array([
            [margin_x, margin_y],           # TL
            [w - margin_x, margin_y],       # TR
            [w - margin_x, h - margin_y],   # BR
            [margin_x, h - margin_y]        # BL
        ], dtype=np.float32)
    
    # Recolectar puntos de todas las líneas
    all_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_points.append((x1, y1))
        all_points.append((x2, y2))
    
    points = np.array(all_points, dtype=np.float32)
    
    # Usar convex hull para encontrar el polígono exterior
    hull = cv2.convexHull(points)
    
    # Simplificar a 4 vértices usando approxPolyDP
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) >= 4:
        # Tomar los 4 vértices más extremos
        # Encontrar: top-left, top-right, bottom-right, bottom-left
        center = np.mean(approx, axis=0)[0]
        
        top_left = None
        top_right = None
        bottom_right = None
        bottom_left = None
        
        for point in approx:
            x, y = point[0]
            if x < center[0] and y < center[1]:
                if top_left is None or y < top_left[1]:
                    top_left = (x, y)
            elif x >= center[0] and y < center[1]:
                if top_right is None or y < top_right[1]:
                    top_right = (x, y)
            elif x >= center[0] and y >= center[1]:
                if bottom_right is None or y > bottom_right[1]:
                    bottom_right = (x, y)
            else:
                if bottom_left is None or y > bottom_left[1]:
                    bottom_left = (x, y)
        
        # Si no encontramos todos los vértices, usar valores por defecto
        if None in [top_left, top_right, bottom_right, bottom_left]:
            # Fallback a heurística
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.10)
            return np.array([
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y]
            ], dtype=np.float32)
        
        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        return corners
    
    # Fallback
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.10)
    return np.array([
        [margin_x, margin_y],
        [w - margin_x, margin_y],
        [w - margin_x, h - margin_y],
        [margin_x, h - margin_y]
    ], dtype=np.float32)


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0


def analyze_with_perspective(video_path: str, max_frames: int = 100, conf_threshold: float = 0.5):
    """
    Analiza el video con:
    1. Detección de polígono de cancha (perspectiva)
    2. Filtro de confianza
    3. Máximo 4 detecciones por frame
    """
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    os.makedirs("runs/perspective_analysis", exist_ok=True)
    
    # Detectar polígono en el primer frame
    ret, first_frame = cap.read()
    if ret:
        corners = detect_court_polygon(first_frame)
        print(f"\n📐 Polígono de cancha detectado:")
        print(f"   TL: ({corners[0][0]:.0f}, {corners[0][1]:.0f})")
        print(f"   TR: ({corners[1][0]:.0f}, {corners[1][1]:.0f})")
        print(f"   BR: ({corners[2][0]:.0f}, {corners[2][1]:.0f})")
        print(f"   BL: ({corners[3][0]:.0f}, {corners[3][1]:.0f})")
        
        polygon = corners.reshape((-1, 1, 2)).astype(np.int32)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    all_detections = []
    court_counts = []
    final_counts = []  # Después de filtrar
    
    print(f"\n🔍 Analizando {max_frames} frames...")
    print(f"   Umbral de confianza: {conf_threshold}")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=[0], verbose=False)
        
        frame_detections = []
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    in_court = point_in_polygon((cx, cy), polygon)
                    
                    frame_detections.append({
                        'frame': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'confidence': conf,
                        'in_court': in_court
                    })
        
        # Filtrar por confianza y posición
        filtered = [d for d in frame_detections if d['confidence'] >= conf_threshold and d['in_court']]
        
        # Si hay más de 4, tomar los 4 con mayor confianza
        if len(filtered) > 4:
            filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
        
        all_detections.extend(frame_detections)
        court_counts.append(sum(1 for d in frame_detections if d['in_court']))
        final_counts.append(len(filtered))
        
        # Guardar frames de muestra
        if frame_idx % 25 == 0:
            sample_frame = frame.copy()
            
            # Dibujar polígono
            cv2.polylines(sample_frame, [polygon], True, (255, 0, 0), 3)
            
            # Dibujar todas las detecciones
            for det in frame_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Color según estado
                if det['in_court'] and conf >= conf_threshold:
                    color = (0, 255, 0)  # Verde: válido
                elif det['in_court']:
                    color = (0, 255, 255)  # Amarillo: en cancha pero baja confianza
                else:
                    color = (0, 0, 255)  # Rojo: fuera de cancha
                
                cv2.rectangle(sample_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{conf:.2f}"
                cv2.putText(sample_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Añadir texto con estadísticas
            cv2.putText(sample_frame, f"In court: {court_counts[-1]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(sample_frame, f"Filtered: {len(filtered)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite(f"runs/perspective_analysis/frame_{frame_idx:04d}.jpg", sample_frame)
            
            print(f"   Frame {frame_idx}: {court_counts[-1]} en cancha → {len(filtered)} filtrados")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS CON PERSPECTIVA + CONFIANZA")
    print(f"{'='*60}")
    
    total = len(all_detections)
    in_court = sum(1 for d in all_detections if d['in_court'])
    
    print(f"\n📍 Clasificación:")
    print(f"   Total detecciones: {total}")
    print(f"   En cancha: {in_court} ({100*in_court/total:.1f}%)")
    
    print(f"\n📈 Promedios por frame:")
    print(f"   En cancha (sin filtro): {np.mean(court_counts):.2f}")
    print(f"   Después de filtrar: {np.mean(final_counts):.2f}")
    
    frames_with_4 = sum(1 for c in final_counts if c == 4)
    frames_with_3_5 = sum(1 for c in final_counts if 3 <= c <= 5)
    
    print(f"\n🎯 Resultados finales:")
    print(f"   Frames con 4 jugadores: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 jugadores: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    
    print(f"\n{'='*60}")
    if frames_with_4 / max_frames >= 0.85:
        print("✅ FILTRADO EXITOSO")
    elif frames_with_3_5 / max_frames >= 0.85:
        print("⚠️  FILTRADO PARCIAL")
    else:
        print("❌ Requiere ajuste")
    print(f"{'='*60}")
    
    return {
        'avg_court': round(np.mean(court_counts), 2),
        'avg_filtered': round(np.mean(final_counts), 2),
        'frames_with_4': frames_with_4,
        'frames_with_3_5': frames_with_3_5,
        'corners': corners.tolist()
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Detección con Perspectiva\n")
    results = analyze_with_perspective(video_path, max_frames=100, conf_threshold=0.5)