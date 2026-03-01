#!/usr/bin/env python3
"""
Guarda ejemplos de frames con 3, 4, y 5 jugadores detectados.
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
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def nms(detections, iou_threshold=0.3):
    """Non-Maximum Suppression."""
    if len(detections) == 0:
        return []
    
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        remaining = []
        for det in sorted_dets:
            if compute_iou(best['bbox'], det['bbox']) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
    
    return keep

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def save_examples(video_path: str, corners_path: str, conf_threshold: float = 0.5):
    # Cargar esquinas
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    print(f"🔄 Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Analizando video...")
    cap = cv2.VideoCapture(video_path)
    
    os.makedirs("runs/examples", exist_ok=True)
    
    # Guardar hasta 3 ejemplos de cada tipo
    examples_saved = {2: 0, 3: 0, 4: 0}
    max_examples = 3
    
    frame_idx = 0
    
    max_frames = 500  # Escanear más frames
    
    while cap.isOpened() and frame_idx < max_frames and any(v < max_examples for v in examples_saved.values()):
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
                    cy_bottom = y2
                    
                    in_court = point_in_polygon((cx, cy_bottom), polygon)
                    
                    if in_court and conf >= conf_threshold:
                        frame_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
        
        # Aplicar NMS
        frame_detections = nms(frame_detections, iou_threshold=0.3)
        
        count = len(frame_detections)
        
        # Guardar si es un ejemplo que necesitamos
        if count in [2, 3, 4] and examples_saved[count] < max_examples:
            sample = frame.copy()
            
            # Dibujar polígono
            cv2.polylines(sample, [polygon], True, (255, 0, 0), 3)
            
            # Dibujar detecciones
            for d in frame_detections:
                x1, y1, x2, y2 = d['bbox']
                conf = d['confidence']
                cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(sample, f"{conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(sample, f"Count: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            filename = f"runs/examples/{count}players_frame{frame_idx:04d}.jpg"
            cv2.imwrite(filename, sample)
            examples_saved[count] += 1
            print(f"   Guardado: {count} jugadores - frame {frame_idx}")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n📁 Ejemplos guardados:")
    for count, num in examples_saved.items():
        print(f"   {count} jugadores: {num} ejemplos")

if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 (720p, h264).mp4"
    corners_path = "runs/manual_court/court_corners.json"
    
    print("🏃 Generando ejemplos de detecciones\n")
    save_examples(video_path, corners_path)