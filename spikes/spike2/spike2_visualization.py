#!/usr/bin/env python3
"""
Spike 2 - Visualización de Trayectorias

Genera un video con las trayectorias de los jugadores dibujadas,
incluyendo métricas en tiempo real y heatmaps.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import colorsys


# Colores para cada jugador (distintos y visibles)
PLAYER_COLORS = [
    (0, 255, 0),    # Verde
    (255, 0, 0),    # Azul
    (0, 0, 255),    # Rojo
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Amarillo
]


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
    """Verifica si un punto está dentro de un polígono."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def pixel_to_meters(pixel_dist: float, court_width_px: float, court_width_m: float = 10.0) -> float:
    """Convierte distancia en pixels a metros (aproximado)."""
    return pixel_dist * (court_width_m / court_width_px)


def draw_trajectory(frame: np.ndarray, 
                    positions: List[Tuple[int, int]], 
                    color: Tuple[int, int, int],
                    max_length: int = 100) -> np.ndarray:
    """Dibuja la trayectoria de un jugador en el frame."""
    if len(positions) < 2:
        return frame
    
    # Solo dibujar los últimos N puntos
    recent_positions = positions[-max_length:]
    
    # Dibujar línea de trayectoria con gradiente de transparencia
    for i in range(1, len(recent_positions)):
        alpha = i / len(recent_positions)  # Más opaco hacia el punto actual
        
        p1 = recent_positions[i-1]
        p2 = recent_positions[i]
        
        # Color con transparencia simulada
        line_color = tuple(int(c * alpha) for c in color)
        
        cv2.line(frame, p1, p2, line_color, 2)
    
    # Dibujar punto actual (más grande)
    if positions:
        current_pos = positions[-1]
        cv2.circle(frame, current_pos, 8, color, -1)
        cv2.circle(frame, current_pos, 10, (255, 255, 255), 2)
    
    return frame


def draw_metrics_panel(frame: np.ndarray, 
                       player_metrics: Dict,
                       track_ids: List[int]) -> np.ndarray:
    """Dibuja un panel con métricas de los jugadores."""
    h, w = frame.shape[:2]
    
    # Panel en la esquina superior derecha
    panel_width = 280
    panel_height = 30 + len(track_ids) * 60
    
    # Crear panel semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width - 10, 10), 
                 (w - 10, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Título
    cv2.putText(frame, "METRICAS JUGADORES", (w - panel_width, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Métricas por jugador
    y_offset = 55
    for i, track_id in enumerate(track_ids):
        if track_id not in player_metrics:
            continue
            
        metrics = player_metrics[track_id]
        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        
        # Indicador de color
        cv2.rectangle(frame, (w - panel_width, y_offset), 
                     (w - panel_width + 15, y_offset + 50), color, -1)
        
        # Datos del jugador
        text_x = w - panel_width + 25
        
        cv2.putText(frame, f"Jugador #{track_id}", (text_x, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        distance = metrics.get('total_distance_m', 0)
        avg_speed = metrics.get('avg_velocity_m_s', 0)
        
        cv2.putText(frame, f"Dist: {distance:.1f}m", (text_x, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Vel: {avg_speed:.1f} m/s", (text_x, y_offset + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y_offset += 60
    
    return frame


def draw_court_polygon(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Dibuja el polígono de la cancha."""
    cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
    return frame


def create_visualization_video(video_path: str,
                               corners_path: str,
                               output_path: str = "runs/spike2/visualization.mp4",
                               max_frames: int = 500,
                               conf_threshold: float = 0.5) -> Dict:
    """
    Crea un video con visualización de trayectorias.
    
    Args:
        video_path: Ruta al video de entrada
        corners_path: Ruta al JSON con las esquinas de la cancha
        output_path: Ruta del video de salida
        max_frames: Máximo de frames a procesar
        conf_threshold: Umbral de confianza para detecciones
    
    Returns:
        Diccionario con estadísticas del procesamiento
    """
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar esquinas de la cancha
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    # Calcular factor de conversión a metros
    court_width_px = np.sqrt((corners['TR'][0] - corners['TL'][0])**2 + 
                             (corners['TR'][1] - corners['TL'][1])**2)
    
    # Cargar modelo
    print(f"🔄 Cargando modelo YOLOv8m...")
    model = YOLO('yolov8m.pt')
    
    # Abrir video
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30.0
    
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Configurar writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Estructuras para tracking
    trajectories = defaultdict(list)  # track_id -> [(x, y), ...]
    player_metrics = defaultdict(lambda: {
        'total_distance_px': 0,
        'total_distance_m': 0,
        'velocities': [],
        'frame_count': 0
    })
    
    frame_count = 0
    
    print(f"\n🎬 Generando video con trayectorias...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar cancha
        frame = draw_court_polygon(frame, polygon)
        
        # Tracking con YOLO
        results = model.track(frame, classes=[0], persist=True, verbose=False, conf=conf_threshold)
        
        frame_detections = []
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy_bottom = y2
                    
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    in_court = point_in_polygon((cx, cy_bottom), polygon)
                    
                    frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': (cx, cy_bottom),
                        'confidence': conf,
                        'in_court': in_court,
                        'track_id': track_id
                    })
        
        # Filtrar detecciones
        filtered = [d for d in frame_detections if d['in_court']]
        filtered = nms(filtered, iou_threshold=0.3)
        
        if len(filtered) > 4:
            filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
        
        # Procesar detecciones filtradas
        active_track_ids = []
        
        for det in filtered:
            track_id = det['track_id']
            pos = det['center']
            
            if track_id < 0:
                continue
            
            active_track_ids.append(track_id)
            
            # Actualizar trayectoria
            trajectories[track_id].append(pos)
            
            # Calcular velocidad instantánea
            if len(trajectories[track_id]) >= 2:
                prev_pos = trajectories[track_id][-2]
                dist_px = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                dist_m = pixel_to_meters(dist_px, court_width_px)
                
                player_metrics[track_id]['total_distance_px'] += dist_px
                player_metrics[track_id]['total_distance_m'] += dist_m
                player_metrics[track_id]['velocities'].append(dist_m * fps)
            
            player_metrics[track_id]['frame_count'] += 1
        
        # Dibujar trayectorias
        track_ids = list(trajectories.keys())
        for i, track_id in enumerate(track_ids):
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            frame = draw_trajectory(frame, trajectories[track_id], color)
            
            # Calcular velocidad promedio para métricas
            velocities = player_metrics[track_id]['velocities']
            if velocities:
                player_metrics[track_id]['avg_velocity_m_s'] = np.mean(velocities)
        
        # Dibujar bounding boxes
        for i, det in enumerate(filtered):
            if det['track_id'] < 0:
                continue
            
            x1, y1, x2, y2 = det['bbox']
            track_idx = track_ids.index(det['track_id']) if det['track_id'] in track_ids else i
            color = PLAYER_COLORS[track_idx % len(PLAYER_COLORS)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta con track_id
            label = f"#{det['track_id']}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar panel de métricas
        frame = draw_metrics_panel(frame, player_metrics, active_track_ids)
        
        # Información del frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Escribir frame
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"   Procesado frame {frame_count}")
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"\n✅ Video guardado en: {output_path}")
    
    # Guardar métricas finales
    final_metrics = {
        str(tid): {
            'total_frames': m['frame_count'],
            'total_distance_m': round(m['total_distance_m'], 2),
            'avg_velocity_m_s': round(np.mean(m['velocities']), 2) if m['velocities'] else 0,
            'max_velocity_m_s': round(max(m['velocities']), 2) if m['velocities'] else 0
        }
        for tid, m in player_metrics.items()
    }
    
    with open(f"{output_dir}/visualization_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    return {
        'frames_processed': frame_count,
        'output_path': output_path,
        'players_detected': len(trajectories),
        'metrics': player_metrics
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 (720p, h264).mp4"
    corners_path = "runs/manual_court/court_corners.json"
    
    print("🎬 Spike 2 - Visualización de Trayectorias\n")
    print("="*60)
    
    results = create_visualization_video(
        video_path=video_path,
        corners_path=corners_path,
        output_path="runs/spike2/visualization.mp4",
        max_frames=500,
        conf_threshold=0.5
    )
    
    print(f"\n{'='*60}")
    print("📊 RESUMEN")
    print(f"{'='*60}")
    print(f"Frames procesados: {results['frames_processed']}")
    print(f"Jugadores detectados: {results['players_detected']}")