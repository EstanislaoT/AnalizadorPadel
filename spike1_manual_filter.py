#!/usr/bin/env python3
"""
Spike 1 - Filtrado de Jugadores con Selección Manual de Cancha

Usa la selección manual de la cancha para filtrar solo los jugadores
que están dentro del área de juego.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

def compute_iou(box1, box2):
    """Calcula IoU entre dos bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersección
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Unión
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def nms(detections, iou_threshold=0.5):
    """Non-Maximum Suppression para eliminar detecciones duplicadas."""
    if len(detections) == 0:
        return []
    
    # Ordenar por confianza
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # Eliminar detecciones con alto IoU con la mejor
        remaining = []
        for det in sorted_dets:
            iou = compute_iou(best['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)
        
        sorted_dets = remaining
    
    return keep

def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def analyze_with_manual_court(video_path: str, corners_path: str, max_frames: int = 100, conf_threshold: float = 0.5):
    """
    Analiza el video usando la selección manual de cancha.
    """
    # Cargar esquinas de la cancha
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    print(f"📐 Cancha cargada:")
    print(f"   TL: {corners['TL']}")
    print(f"   TR: {corners['TR']}")
    print(f"   BR: {corners['BR']}")
    print(f"   BL: {corners['BL']}")
    
    # Crear polígono
    points = np.array([
        corners['TL'],
        corners['TR'],
        corners['BR'],
        corners['BL']
    ], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    print(f"\n🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    os.makedirs("runs/manual_filter", exist_ok=True)
    
    all_detections = []
    final_counts = []
    
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
                    # Usar la PARTE INFERIOR del bbox (pies) en lugar del centro
                    # porque los jugadores tienen los pies en el piso de la cancha
                    cy_bottom = y2  # Coordenada Y inferior (pies)
                    
                    # Verificar si los pies están en la cancha
                    in_court = point_in_polygon((cx, cy_bottom), polygon)
                    
                    frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy_bottom],
                        'confidence': conf,
                        'in_court': in_court
                    })
        
        # Filtrar: en cancha + confianza >= threshold
        filtered = [d for d in frame_detections if d['in_court'] and d['confidence'] >= conf_threshold]
        
        # Aplicar NMS para eliminar duplicados
        filtered = nms(filtered, iou_threshold=0.3)
        
        if len(filtered) > 4:
            filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
        
        final_counts.append(len(filtered))
        all_detections.extend(frame_detections)
        
        # Guardar frames de muestra
        if frame_idx % 25 == 0:
            sample = frame.copy()
            
            # Dibujar polígono
            cv2.polylines(sample, [polygon], True, (255, 0, 0), 3)
            
            # Dibujar detecciones
            for d in frame_detections:
                x1, y1, x2, y2 = d['bbox']
                conf = d['confidence']
                
                if d['in_court'] and conf >= conf_threshold:
                    color = (0, 255, 0)  # Verde: válido
                elif d['in_court']:
                    color = (0, 255, 255)  # Amarillo: en cancha, baja conf
                else:
                    color = (0, 0, 255)  # Rojo: fuera
                
                cv2.rectangle(sample, (x1, y1), (x2, y2), color, 2)
                cv2.putText(sample, f"{conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.putText(sample, f"Filtered: {len(filtered)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite(f"runs/manual_filter/frame_{frame_idx:04d}.jpg", sample)
            print(f"   Frame {frame_idx}: {len(filtered)} jugadores")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS CON SELECCIÓN MANUAL")
    print(f"{'='*60}")
    
    total = len(all_detections)
    in_court = sum(1 for d in all_detections if d['in_court'])
    
    print(f"\n📍 Clasificación:")
    print(f"   Total detecciones: {total}")
    print(f"   En cancha: {in_court} ({100*in_court/total:.1f}%)")
    print(f"   Fuera de cancha: {total - in_court}")
    
    print(f"\n📈 Promedio por frame:")
    print(f"   Jugadores filtrados: {np.mean(final_counts):.2f}/frame")
    
    frames_with_4 = sum(1 for c in final_counts if c == 4)
    frames_with_3_5 = sum(1 for c in final_counts if 3 <= c <= 5)
    
    # Distribución completa
    distribution = {}
    for c in final_counts:
        distribution[c] = distribution.get(c, 0) + 1
    
    print(f"\n🎯 Resultados:")
    print(f"   Frames con 4 jugadores: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 jugadores: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    
    print(f"\n📊 Distribución completa:")
    for count in sorted(distribution.keys()):
        pct = 100 * distribution[count] / max_frames
        print(f"   {count} jugadores: {distribution[count]} frames ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    if frames_with_4 / max_frames >= 0.85:
        print("✅ FILTRADO EXITOSO")
    elif frames_with_3_5 / max_frames >= 0.85:
        print("⚠️  FILTRADO PARCIALMENTE EXITOSO")
    else:
        print("❌ Requiere más trabajo")
    print(f"{'='*60}")
    
    print(f"\n📁 Archivos guardados en runs/manual_filter/")
    
    return {
        'total': total,
        'in_court': in_court,
        'frames_with_4': frames_with_4,
        'frames_with_3_5': frames_with_3_5,
        'avg': round(np.mean(final_counts), 2)
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 (720p, h264).mp4"
    corners_path = "runs/manual_court/court_corners.json"
    
    print("🏃 Spike 1 - Filtrado con Selección Manual\n")
    print("="*60)
    
    results = analyze_with_manual_court(video_path, corners_path, max_frames=100, conf_threshold=0.5)