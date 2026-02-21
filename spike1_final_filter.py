#!/usr/bin/env python3
"""
Spike 1 - Enfoque Final Combinado

Mejor resultado: Detección de líneas (37% con 4, 94% con 3-5)

Combinar:
1. Zona rectangular fija (basada en análisis previo)
2. Filtro de confianza >= 0.5
3. Tomar máximo 4 detecciones

Basado en el análisis de detección de líneas:
- Rectángulo detectado: TL(636,255), TR(1088,255), BR(1088,683), BL(636,683)
- Este rectángulo cubría solo la mitad derecha
- Expandimos para cubrir toda la cancha
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def analyze_final(video_path: str, max_frames: int = 100, conf_threshold: float = 0.5):
    """
    Análisis final con parámetros optimizados.
    """
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    # Zona de cancha optimizada basada en análisis previo
    # El rectángulo de líneas era: (636,255) a (1088,683) - solo mitad derecha
    # Expandimos a toda la cancha
    margin_x = int(width * 0.12)   # 12% de margen lateral
    margin_y = int(height * 0.08)  # 8% de margen vertical
    
    court_left = margin_x
    court_right = width - margin_x
    court_top = margin_y
    court_bottom = height - margin_y
    
    # Crear polígono rectangular
    polygon = np.array([
        [court_left, court_top],
        [court_right, court_top],
        [court_right, court_bottom],
        [court_left, court_bottom]
    ], dtype=np.int32).reshape((-1, 1, 2))
    
    print(f"\n📐 Zona de cancha optimizada:")
    print(f"   TL: ({court_left}, {court_top})")
    print(f"   TR: ({court_right}, {court_top})")
    print(f"   BR: ({court_right}, {court_bottom})")
    print(f"   BL: ({court_left}, {court_bottom})")
    
    os.makedirs("runs/final_analysis", exist_ok=True)
    
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
                    cy = (y1 + y2) // 2
                    
                    # Verificar si está en la zona
                    in_court = (court_left <= cx <= court_right and 
                               court_top <= cy <= court_bottom)
                    
                    frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'confidence': conf,
                        'in_court': in_court
                    })
        
        # Filtrar: en cancha + confianza >= threshold
        filtered = [d for d in frame_detections if d['in_court'] and d['confidence'] >= conf_threshold]
        
        # Si hay más de 4, tomar los 4 con mayor confianza
        if len(filtered) > 4:
            filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
        
        final_counts.append(len(filtered))
        all_detections.extend(frame_detections)
        
        # Guardar frames de muestra
        if frame_idx % 25 == 0:
            sample = frame.copy()
            
            # Dibujar zona
            cv2.rectangle(sample, (court_left, court_top), (court_right, court_bottom), (255, 0, 0), 2)
            
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
                cv2.putText(sample, f"{conf:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Texto
            cv2.putText(sample, f"Filtered: {len(filtered)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite(f"runs/final_analysis/frame_{frame_idx:04d}.jpg", sample)
            print(f"   Frame {frame_idx}: {len(filtered)} jugadores detectados")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS FINAL")
    print(f"{'='*60}")
    
    frames_with_4 = sum(1 for c in final_counts if c == 4)
    frames_with_3_5 = sum(1 for c in final_counts if 3 <= c <= 5)
    
    print(f"\n🎯 Resultados:")
    print(f"   Frames con 4 jugadores: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 jugadores: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    print(f"   Promedio: {np.mean(final_counts):.2f} jugadores/frame")
    
    print(f"\n{'='*60}")
    if frames_with_4 / max_frames >= 0.85:
        print("✅ SPIKE 1 EXITOSO")
    elif frames_with_3_5 / max_frames >= 0.85:
        print("⚠️  SPIKE 1 PARCIALMENTE EXITOSO")
    else:
        print("❌ SPIKE 1 requiere más trabajo")
    print(f"{'='*60}")
    
    return {
        'frames_with_4': frames_with_4,
        'frames_with_3_5': frames_with_3_5,
        'avg': round(np.mean(final_counts), 2)
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Enfoque Final Combinado\n")
    results = analyze_final(video_path, max_frames=100, conf_threshold=0.5)