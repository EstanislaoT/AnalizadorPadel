#!/usr/bin/env python3
"""
Spike 1 - Detección de Piso de Cancha

Nuevo enfoque: Detectar el piso de la cancha por color y bordes,
luego filtrar solo las detecciones que intersectan con el piso.

Ventajas sobre filtrado por zona fija:
- Se adapta a cualquier ángulo de cámara
- No requiere configuración manual de zona
- Excluye automáticamente árbitros y espectadores que están fuera del piso
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_court_floor(frame):
    """
    Detecta el piso de la cancha usando:
    1. Detección de color verde/azul típico de canchas de pádel
    2. Detección de bordes
    3. Combinación de ambas
    
    Retorna una máscara binaria donde 255 = piso de cancha
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rango para verde típico de cancha de pádel
    # El verde puede variar, así que usamos rangos amplios
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Rango para azul (algunas canchas son azules)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    
    # Crear máscaras de color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combinar máscaras (verde O azul)
    color_mask = cv2.bitwise_or(green_mask, blue_mask)
    
    # Aplicar operaciones morfológicas para limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar el contorno más grande (probablemente la cancha)
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Ordenar por área y tomar el más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Crear máscara solo con el contorno más grande
        floor_mask = np.zeros_like(color_mask)
        cv2.drawContours(floor_mask, [largest_contour], -1, 255, -1)
        
        return floor_mask
    
    return color_mask


def detect_court_lines(frame):
    """
    Detecta las líneas de la cancha usando Canny y Hough.
    Útil para validar los bordes de la cancha.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detectar líneas con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=20)
    
    return edges, lines


def bbox_intersects_mask(bbox, mask):
    """
    Verifica si un bounding box intersecta con la máscara del piso.
    
    Criterio: Al menos un porcentaje del bbox debe estar dentro del piso.
    """
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    
    if bbox_area == 0:
        return False, 0
    
    # Extraer la región del bbox de la máscara
    bbox_mask = mask[y1:y2, x1:x2]
    
    # Contar píxeles del piso dentro del bbox
    floor_pixels = np.count_nonzero(bbox_mask)
    
    # Calcular porcentaje de intersección
    intersection_ratio = floor_pixels / bbox_area
    
    return intersection_ratio > 0.3, intersection_ratio  # 30% del bbox debe estar en el piso


def analyze_with_floor_detection(video_path: str, max_frames: int = 100):
    """
    Analiza el video usando detección de piso de cancha.
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
    os.makedirs("runs/floor_analysis", exist_ok=True)
    
    all_detections = []
    floor_counts = []
    floor_ratios = []
    
    print(f"\n🔍 Analizando {max_frames} frames con detección de piso...")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar piso de cancha
        floor_mask = detect_court_floor(frame)
        
        # Detectar personas con YOLO
        results = model(frame, classes=[0], verbose=False)
        
        frame_detections = []
        floor_detections = 0
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:  # persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Verificar intersección con piso
                    intersects, ratio = bbox_intersects_mask((x1, y1, x2, y2), floor_mask)
                    
                    detection = {
                        'frame': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'confidence': round(conf, 3),
                        'intersects_floor': intersects,
                        'floor_ratio': round(ratio, 3)
                    }
                    
                    frame_detections.append(detection)
                    
                    if intersects:
                        floor_detections += 1
        
        all_detections.extend(frame_detections)
        floor_counts.append(floor_detections)
        floor_ratios.extend([d['floor_ratio'] for d in frame_detections if d['intersects_floor']])
        
        # Guardar frames de muestra cada 25 frames
        if frame_idx % 25 == 0:
            sample_frame = frame.copy()
            
            # Dibujar máscara de piso semi-transparente
            overlay = sample_frame.copy()
            overlay[floor_mask > 0] = [0, 255, 0]  # Verde para piso
            cv2.addWeighted(overlay, 0.3, sample_frame, 0.7, 0, sample_frame)
            
            # Dibujar detecciones
            for det in frame_detections:
                x1, y1, x2, y2 = det['bbox']
                color = (0, 255, 255) if det['intersects_floor'] else (0, 0, 255)  # Amarillo si intersecta, rojo si no
                cv2.rectangle(sample_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['confidence']:.2f} ({det['floor_ratio']:.1%})"
                cv2.putText(sample_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imwrite(f"runs/floor_analysis/frame_{frame_idx:04d}.jpg", sample_frame)
            
            total = len(frame_detections)
            print(f"   Frame {frame_idx}: {floor_detections}/{total} en piso | " +
                  f"Ratios: {[f'{r:.1%}' for r in floor_ratios[-floor_detections:]]}")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS CON DETECCIÓN DE PISO")
    print(f"{'='*60}")
    
    total_detections = len(all_detections)
    floor_detections = sum(1 for d in all_detections if d['intersects_floor'])
    
    print(f"\n📍 Clasificación:")
    print(f"   Total detecciones: {total_detections}")
    print(f"   Detecciones EN PISO: {floor_detections} ({100*floor_detections/total_detections:.1f}%)")
    print(f"   Detecciones FUERA DE PISO: {total_detections - floor_detections}")
    
    # Promedio por frame
    avg_floor = np.mean(floor_counts)
    print(f"\n📈 Promedio por frame:")
    print(f"   Personas en piso: {avg_floor:.2f}/frame")
    
    # Frames con 4 jugadores
    frames_with_4 = sum(1 for c in floor_counts if c == 4)
    frames_with_3_5 = sum(1 for c in floor_counts if 3 <= c <= 5)
    
    print(f"\n🎯 Resultados:")
    print(f"   Frames con 4 en piso: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 en piso: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    
    # Ratio promedio de intersección
    if floor_ratios:
        avg_ratio = np.mean(floor_ratios)
        print(f"\n📐 Ratio de intersección promedio: {avg_ratio:.1%}")
    
    # Veredicto
    print(f"\n{'='*60}")
    if frames_with_4 / max_frames >= 0.85:
        print("✅ FILTRADO POR PISO EXITOSO")
        print("   La detección de piso funciona correctamente")
    elif frames_with_3_5 / max_frames >= 0.85:
        print("⚠️  FILTRADO PARCIAL")
        print("   Mayoría de frames tienen 3-5 detecciones en piso")
    else:
        print("❌ Requiere ajuste de parámetros de detección de piso")
    print(f"{'='*60}")
    
    print(f"\n📁 Archivos guardados en runs/floor_analysis/")
    
    return {
        'total_detections': total_detections,
        'floor_detections': floor_detections,
        'avg_floor_per_frame': round(avg_floor, 2),
        'frames_with_4': frames_with_4,
        'frames_with_3_5': frames_with_3_5
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Detección de Piso de Cancha\n")
    results = analyze_with_floor_detection(video_path, max_frames=100)