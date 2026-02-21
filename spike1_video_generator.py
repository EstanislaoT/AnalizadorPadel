#!/usr/bin/env python3
"""
Spike 1 - Generador de video con detecciones YOLO

Genera un video corto (10 segundos) con bounding boxes para verificación visual.
Usa stream=True para evitar problemas de memoria.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def generate_detection_video(
    video_path: str, 
    output_path: str = "runs/detect/spike1/detection_output.mp4",
    max_seconds: int = 10
):
    """
    Genera un video con bounding boxes de detección YOLO.
    """
    # Crear directorio de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: No se puede abrir el video")
        return None
    
    # Info del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   FPS: {fps:.2f} | Resolución: {width}x{height}")
    
    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Error: No se puede crear el video de salida")
        cap.release()
        return None
    
    frames_to_process = min(int(fps * max_seconds), total_frames)
    print(f"\n🎯 Procesando {frames_to_process} frames ({max_seconds}s)...")
    
    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Frame {frame_idx} no se pudo leer")
            break
        
        # Ejecutar YOLO
        results = model(frame, classes=[0], verbose=False)
        
        # Dibujar bounding boxes
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:  # persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    # Color verde con grosor 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Etiqueta con confianza
                    label = f"Person {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Escribir frame al video
        out.write(frame)
        
        # Progreso cada 30 frames
        if (frame_idx + 1) % 30 == 0:
            print(f"   Procesados {frame_idx + 1}/{frames_to_process} frames")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Video guardado en: {output_path}")
    print(f"   Tamaño: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return output_path


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Generador de Video con Detecciones\n")
    generate_detection_video(video_path, max_seconds=10)