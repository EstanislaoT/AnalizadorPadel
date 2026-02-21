#!/usr/bin/env python3
"""
Spike 1 - Análisis detallado de detecciones YOLO

Analiza las coordenadas de cada detección para identificar:
- Qué detecciones están dentro de la zona de la cancha
- Qué detecciones están fuera (espectadores, árbitro, etc.)
- Genera imágenes de muestra con las zonas marcadas
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json

def analyze_detections(video_path: str, max_frames: int = 100):
    """
    Analiza detecciones YOLO y clasifica por zona.
    """
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    # Definir zona de cancha (heurística: centro del frame)
    # Cancha de pádel es rectangular, típicamente en el centro
    margin_x = int(width * 0.15)  # 15% de margen lateral
    margin_y = int(height * 0.12)  # 12% de margen vertical
    
    court_left = margin_x
    court_right = width - margin_x
    court_top = margin_y
    court_bottom = height - margin_y
    
    print(f"\n📐 Zona de cancha definida:")
    print(f"   Izquierda: {court_left}px | Derecha: {court_right}px")
    print(f"   Arriba: {court_top}px | Abajo: {court_bottom}px")
    print(f"   Centro de cancha: ({width//2}, {height//2})")
    
    # Crear directorio para frames de análisis
    os.makedirs("runs/analysis", exist_ok=True)
    
    all_detections = []
    frames_inside_court = []
    frames_outside_court = []
    
    print(f"\n🔍 Analizando {max_frames} frames...")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=[0], verbose=False)
        
        frame_detections = []
        inside_count = 0
        outside_count = 0
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:  # persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    # Centro del bounding box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Clasificar por zona
                    is_inside = (court_left <= cx <= court_right and 
                                court_top <= cy <= court_bottom)
                    
                    detection = {
                        'frame': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'confidence': round(conf, 3),
                        'inside_court': is_inside
                    }
                    
                    frame_detections.append(detection)
                    
                    if is_inside:
                        inside_count += 1
                    else:
                        outside_count += 1
        
        all_detections.extend(frame_detections)
        frames_inside_court.append(inside_count)
        frames_outside_court.append(outside_count)
        
        # Guardar frame de muestra cada 25 frames
        if frame_idx % 25 == 0:
            sample_frame = frame.copy()
            
            # Dibujar zona de cancha
            cv2.rectangle(sample_frame, (court_left, court_top), 
                         (court_right, court_bottom), (255, 0, 0), 2)
            
            # Dibujar detecciones
            for det in frame_detections:
                x1, y1, x2, y2 = det['bbox']
                cx, cy = det['center']
                color = (0, 255, 0) if det['inside_court'] else (0, 0, 255)  # verde si dentro, rojo si fuera
                cv2.rectangle(sample_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(sample_frame, (cx, cy), 5, color, -1)
            
            cv2.imwrite(f"runs/analysis/frame_{frame_idx:04d}.jpg", sample_frame)
            
            print(f"   Frame {frame_idx}: {inside_count} dentro, {outside_count} fuera | Total: {len(frame_detections)}")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS DE DETECCIONES")
    print(f"{'='*60}")
    
    total_inside = sum(frames_inside_court)
    total_outside = sum(frames_outside_court)
    total_detections = total_inside + total_outside
    
    print(f"\n📍 Clasificación por zona:")
    print(f"   Detecciones DENTRO de cancha: {total_inside} ({100*total_inside/total_detections:.1f}%)")
    print(f"   Detecciones FUERA de cancha: {total_outside} ({100*total_outside/total_detections:.1f}%)")
    
    # Promedio por frame
    avg_inside = np.mean(frames_inside_court)
    avg_outside = np.mean(frames_outside_court)
    
    print(f"\n📈 Promedio por frame:")
    print(f"   Dentro de cancha: {avg_inside:.1f} personas/frame")
    print(f"   Fuera de cancha: {avg_outside:.1f} personas/frame")
    
    # Frames con exactamente 4 jugadores dentro
    frames_with_4 = sum(1 for c in frames_inside_court if c == 4)
    frames_with_3_5 = sum(1 for c in frames_inside_court if 3 <= c <= 5)
    
    print(f"\n🎯 Resultados del filtrado:")
    print(f"   Frames con 4 personas DENTRO: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 personas DENTRO: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    
    # Análisis de posiciones típicas fuera de cancha
    outside_dets = [d for d in all_detections if not d['inside_court']]
    if outside_dets:
        # Agrupar por región
        top_region = sum(1 for d in outside_dets if d['center'][1] < court_top)
        bottom_region = sum(1 for d in outside_dets if d['center'][1] > court_bottom)
        left_region = sum(1 for d in outside_dets if d['center'][0] < court_left)
        right_region = sum(1 for d in outside_dets if d['center'][0] > court_right)
        
        print(f"\n🗺️ Ubicación de detecciones FUERA de cancha:")
        print(f"   Arriba (árbitro/personal): {top_region}")
        print(f"   Abajo (cámara/producción): {bottom_region}")
        print(f"   Izquierda (espectadores): {left_region}")
        print(f"   Derecha (espectadores): {right_region}")
    
    # Veredicto
    print(f"\n{'='*60}")
    if frames_with_4 / max_frames >= 0.85:
        print("✅ SPIKE 1 EXITOSO con filtrado de zona")
        print("   YOLO detecta correctamente los 4 jugadores cuando se filtra por zona")
    elif frames_with_3_5 / max_frames >= 0.85:
        print("⚠️  SPIKE 1 PARCIAL")
        print("   La mayoría de frames tienen 3-5 detecciones dentro de cancha")
        print("   Posibles causas: oclusiones, jugadores en los bordes")
    else:
        print("❌ SPIKE 1 requiere ajuste de parámetros de zona")
    print(f"{'='*60}")
    
    # Guardar reporte
    report = {
        'total_frames': max_frames,
        'court_zone': {
            'left': court_left,
            'right': court_right,
            'top': court_top,
            'bottom': court_bottom
        },
        'summary': {
            'total_inside_court': total_inside,
            'total_outside_court': total_outside,
            'avg_inside_per_frame': round(avg_inside, 2),
            'avg_outside_per_frame': round(avg_outside, 2)
        },
        'frames_with_4_inside': frames_with_4,
        'frames_with_3_5_inside': frames_with_3_5,
        'outside_breakdown': {
            'top': top_region if outside_dets else 0,
            'bottom': bottom_region if outside_dets else 0,
            'left': left_region if outside_dets else 0,
            'right': right_region if outside_dets else 0
        }
    }
    
    with open('runs/analysis/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Archivos guardados en runs/analysis/")
    print(f"   - Frames de muestra con zonas marcadas")
    print(f"   - report.json con estadísticas completas")
    
    return report


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Análisis Detallado de Detecciones YOLO\n")
    report = analyze_detections(video_path, max_frames=100)