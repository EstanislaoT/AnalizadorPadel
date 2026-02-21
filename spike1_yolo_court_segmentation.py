#!/usr/bin/env python3
"""
Spike 1 - Segmentación de Cancha con YOLO

Usar YOLOv8-seg para intentar detectar el piso de la cancha como:
- Área deportiva
- Superficie de juego

Si funciona, podemos usar la máscara segmentada para filtrar jugadores.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_court_with_yolo(frame, model):
    """
    Intenta detectar la cancha usando YOLO segmentación.
    
    YOLO puede detectar objetos como:
    - sports ball (37)
    - otras clases que podrían representar áreas deportivas
    
    También probaremos con las máscaras de segmentación.
    """
    # Ejecutar YOLO con segmentación
    results = model(frame, verbose=False)
    
    court_mask = None
    
    for r in results:
        if r.masks is not None:
            # Hay máscaras de segmentación
            for i, mask in enumerate(r.masks):
                # Obtener la clase
                cls = r.boxes[i].cls[0]
                conf = r.boxes[i].conf[0]
                
                print(f"   Segmento detectado: clase {int(cls)}, confianza {conf:.2f}")
                
                # Convertir máscara a array numpy
                mask_array = mask.data.cpu().numpy()[0]
                mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
                
                # Si es una máscara grande (>10% del frame), podría ser la cancha
                mask_area = np.sum(mask_resized > 0.5)
                frame_area = frame.shape[0] * frame.shape[1]
                
                if mask_area > frame_area * 0.1:
                    print(f"      → Máscara grande: {100*mask_area/frame_area:.1f}% del frame")
                    if court_mask is None:
                        court_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                    else:
                        court_mask = cv2.bitwise_or(court_mask, (mask_resized > 0.5).astype(np.uint8) * 255)
    
    return court_mask


def analyze_with_yolo_segmentation(video_path: str, max_frames: int = 10):
    """
    Analiza el video usando segmentación YOLO para detectar la cancha.
    """
    print(f"🔄 Cargando modelo YOLO v8 nano segmentación...")
    model = YOLO('yolov8n-seg.pt')  # Modelo de segmentación
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    os.makedirs("runs/yolo_seg", exist_ok=True)
    
    print(f"\n🔍 Analizando {max_frames} frames con segmentación YOLO...")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\n   Frame {frame_idx}:")
        
        # Intentar detectar cancha
        court_mask = detect_court_with_yolo(frame, model)
        
        if court_mask is not None:
            # Guardar frame con máscara
            output = frame.copy()
            overlay = output.copy()
            overlay[court_mask > 0] = [0, 255, 0]  # Verde
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
            
            # Detectar personas también
            person_results = model(frame, classes=[0], verbose=False)
            for r in person_results:
                for box in r.boxes:
                    if box.cls[0] == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cv2.imwrite(f"runs/yolo_seg/frame_{frame_idx:04d}.jpg", output)
            print(f"      → Guardado en runs/yolo_seg/frame_{frame_idx:04d}.jpg")
        else:
            # Guardar frame sin máscara pero con detecciones
            output = frame.copy()
            person_results = model(frame, classes=[0], verbose=False)
            for r in person_results:
                for box in r.boxes:
                    if box.cls[0] == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cv2.imwrite(f"runs/yolo_seg/frame_{frame_idx:04d}.jpg", output)
            print(f"      → Sin máscara de cancha detectada")
    
    cap.release()
    
    print(f"\n📁 Archivos guardados en runs/yolo_seg/")


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Segmentación de Cancha con YOLO\n")
    analyze_with_yolo_segmentation(video_path, max_frames=10)