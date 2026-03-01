#!/usr/bin/env python3
"""
Spike 2 - Análisis de Movimiento de Jugadores

Extrae trayectorias de cada jugador y calcula métricas de movimiento:
- Velocidad promedio y máxima
- Distancia total recorrida
- Heatmap de posiciones
- Detección de sprints
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class PlayerPosition:
    """Posición de un jugador en un frame."""
    frame_idx: int
    x: float
    y: float
    confidence: float
    timestamp: float  # en segundos
    

@dataclass
class PlayerTrajectory:
    """Trayectoria completa de un jugador."""
    track_id: int
    positions: List[PlayerPosition] = field(default_factory=list)
    
    def add_position(self, pos: PlayerPosition):
        self.positions.append(pos)
    
    @property
    def total_frames(self) -> int:
        return len(self.positions)
    
    def get_positions_array(self) -> np.ndarray:
        """Retorna posiciones como array numpy (N, 2)."""
        return np.array([[p.x, p.y] for p in self.positions])


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


def smooth_trajectory(positions: List[PlayerPosition], window_size: int = 3) -> List[PlayerPosition]:
    """Suaviza una trayectoria usando promedio móvil."""
    if len(positions) < window_size:
        return positions
    
    smoothed = []
    half_window = window_size // 2
    
    for i, pos in enumerate(positions):
        start = max(0, i - half_window)
        end = min(len(positions), i + half_window + 1)
        window = positions[start:end]
        
        avg_x = np.mean([p.x for p in window])
        avg_y = np.mean([p.y for p in window])
        
        smoothed.append(PlayerPosition(
            frame_idx=pos.frame_idx,
            x=avg_x,
            y=avg_y,
            confidence=pos.confidence,
            timestamp=pos.timestamp
        ))
    
    return smoothed


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula distancia euclidiana entre dos puntos."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_velocity(dist: float, time_delta: float) -> float:
    """Calcula velocidad (distancia/tiempo)."""
    if time_delta <= 0:
        return 0.0
    return dist / time_delta


def pixel_to_meters(pixel_dist: float, court_width_px: float, court_width_m: float = 10.0) -> float:
    """Convierte distancia en pixels a metros (aproximado)."""
    return pixel_dist * (court_width_m / court_width_px)


def generate_heatmap(positions: List[Tuple[float, float]], 
                     resolution: Tuple[int, int] = (100, 100),
                     sigma: float = 2.0) -> np.ndarray:
    """Genera un heatmap a partir de posiciones."""
    heatmap = np.zeros(resolution[::-1])  # (height, width)
    
    if len(positions) == 0:
        return heatmap
    
    # Normalizar posiciones al rango del heatmap
    positions = np.array(positions)
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    # Evitar división por cero
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)
    
    for x, y in positions:
        # Normalizar a [0, resolution-1]
        hx = int((x - x_min) / x_range * (resolution[0] - 1))
        hy = int((y - y_min) / y_range * (resolution[1] - 1))
        
        # Aumentar valor en la posición
        if 0 <= hx < resolution[0] and 0 <= hy < resolution[1]:
            heatmap[hy, hx] += 1
    
    # Aplicar suavizado gaussiano
    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (0, 0), sigma)
    
    # Normalizar a [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def analyze_trajectories(video_path: str, 
                         corners_path: str,
                         output_dir: str = "runs/spike2",
                         max_frames: int = 500,
                         conf_threshold: float = 0.5,
                         fps: float = 30.0) -> Dict:
    """
    Analiza trayectorias de jugadores en un video.
    
    Args:
        video_path: Ruta al video de entrada
        corners_path: Ruta al JSON con las esquinas de la cancha
        output_dir: Directorio de salida
        max_frames: Máximo de frames a analizar
        conf_threshold: Umbral de confianza para detecciones
        fps: Frames por segundo del video
    
    Returns:
        Diccionario con métricas y estadísticas
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar esquinas de la cancha
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    print(f"📐 Cancha cargada:")
    print(f"   TL: {corners['TL']}")
    print(f"   TR: {corners['TR']}")
    print(f"   BR: {corners['BR']}")
    print(f"   BL: {corners['BL']}")
    
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    # Calcular ancho de cancha en pixels para conversión a metros
    court_width_px = np.sqrt((corners['TR'][0] - corners['TL'][0])**2 + 
                             (corners['TR'][1] - corners['TL'][1])**2)
    court_height_px = np.sqrt((corners['BL'][0] - corners['TL'][0])**2 + 
                              (corners['BL'][1] - corners['TL'][1])**2)
    
    print(f"\n📏 Dimensiones de cancha en pixels:")
    print(f"   Ancho: {court_width_px:.1f}px ≈ 10m")
    print(f"   Alto: {court_height_px:.1f}px ≈ 20m")
    
    # Cargar modelo
    print(f"\n🔄 Cargando modelo YOLOv8m...")
    model = YOLO('yolov8m.pt')
    
    # Abrir video
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {video_fps}")
    
    # Usar FPS del video si está disponible
    if video_fps > 0:
        fps = video_fps
    
    # Diccionario de trayectorias
    trajectories: Dict[int, PlayerTrajectory] = {}
    
    # Todas las detecciones para estadísticas
    all_detections = []
    frame_count = 0
    
    print(f"\n🔍 Analizando {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Usar tracking de YOLO
        results = model.track(frame, classes=[0], persist=True, verbose=False, conf=conf_threshold)
        
        frame_detections = []
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) / 2  # Usar float para más precisión
                    cy_bottom = y2  # Posición de los pies
                    
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    in_court = point_in_polygon((cx, cy_bottom), polygon)
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy_bottom],
                        'confidence': conf,
                        'in_court': in_court,
                        'track_id': track_id,
                        'frame_idx': frame_count
                    }
                    
                    frame_detections.append(detection)
                    
                    # Si está en cancha y tiene track_id válido, guardar en trayectoria
                    if in_court and track_id >= 0:
                        if track_id not in trajectories:
                            trajectories[track_id] = PlayerTrajectory(track_id=track_id)
                        
                        pos = PlayerPosition(
                            frame_idx=frame_count,
                            x=cx,
                            y=cy_bottom,
                            confidence=conf,
                            timestamp=frame_count / fps
                        )
                        trajectories[track_id].add_position(pos)
        
        # Filtrar detecciones en cancha
        filtered = [d for d in frame_detections if d['in_court']]
        filtered = nms(filtered, iou_threshold=0.3)
        
        if len(filtered) > 4:
            filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
        
        all_detections.extend(frame_detections)
        
        if frame_count % 100 == 0:
            print(f"   Frame {frame_count}: {len(trajectories)} tracks únicos")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS DE TRAYECTORIAS")
    print(f"{'='*60}")
    
    # Filtrar trayectorias con pocas detecciones (ruido)
    min_positions = max_frames * 0.1  # Al menos 10% de los frames
    valid_trajectories = {tid: traj for tid, traj in trajectories.items() 
                         if traj.total_frames >= min_positions}
    
    print(f"\n🎯 Trayectorias válidas (≥{min_positions:.0f} frames): {len(valid_trajectories)}")
    
    # Métricas por jugador
    player_metrics = {}
    
    for track_id, trajectory in valid_trajectories.items():
        positions = trajectory.positions
        
        # Suavizar trayectoria
        smoothed = smooth_trajectory(positions, window_size=5)
        
        # Calcular distancias entre frames consecutivos
        distances_px = []
        velocities_px_s = []
        
        for i in range(1, len(smoothed)):
            p1 = (smoothed[i-1].x, smoothed[i-1].y)
            p2 = (smoothed[i].x, smoothed[i].y)
            
            dist = calculate_distance(p1, p2)
            time_delta = smoothed[i].timestamp - smoothed[i-1].timestamp
            
            distances_px.append(dist)
            velocities_px_s.append(calculate_velocity(dist, time_delta))
        
        # Convertir a metros
        distances_m = [pixel_to_meters(d, court_width_px) for d in distances_px]
        velocities_m_s = [pixel_to_meters(v, court_width_px) for v in velocities_px_s]
        
        # Calcular métricas agregadas
        total_distance_px = sum(distances_px)
        total_distance_m = sum(distances_m)
        
        avg_velocity_m_s = np.mean(velocities_m_s) if velocities_m_s else 0
        max_velocity_m_s = max(velocities_m_s) if velocities_m_s else 0
        
        # Detectar sprints (velocidad > umbral)
        sprint_threshold = 3.0  # m/s (aproximadamente 10.8 km/h)
        sprints = sum(1 for v in velocities_m_s if v > sprint_threshold)
        
        player_metrics[track_id] = {
            'total_frames': trajectory.total_frames,
            'total_distance_px': total_distance_px,
            'total_distance_m': total_distance_m,
            'avg_velocity_m_s': avg_velocity_m_s,
            'max_velocity_m_s': max_velocity_m_s,
            'sprints_count': sprints,
            'positions': [(p.x, p.y) for p in smoothed]
        }
        
        print(f"\n🏃 Jugador #{track_id}:")
        print(f"   Frames detectados: {trajectory.total_frames}")
        print(f"   Distancia total: {total_distance_m:.1f}m ({total_distance_px:.0f}px)")
        print(f"   Velocidad promedio: {avg_velocity_m_s:.2f} m/s ({avg_velocity_m_s * 3.6:.1f} km/h)")
        print(f"   Velocidad máxima: {max_velocity_m_s:.2f} m/s ({max_velocity_m_s * 3.6:.1f} km/h)")
        print(f"   Sprints detectados: {sprints}")
    
    # Generar heatmaps por jugador
    print(f"\n🗺️ Generando heatmaps...")
    
    for track_id, metrics in player_metrics.items():
        positions = metrics['positions']
        if len(positions) > 10:
            heatmap = generate_heatmap(positions, resolution=(100, 100))
            
            # Guardar heatmap como imagen
            heatmap_img = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            cv2.imwrite(f"{output_dir}/heatmap_player_{track_id}.png", heatmap_color)
            
            print(f"   Heatmap generado para jugador #{track_id}")
    
    # Generar heatmap combinado
    all_positions = []
    for metrics in player_metrics.values():
        all_positions.extend(metrics['positions'])
    
    if all_positions:
        combined_heatmap = generate_heatmap(all_positions, resolution=(100, 100))
        heatmap_img = (combined_heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        cv2.imwrite(f"{output_dir}/heatmap_combined.png", heatmap_color)
        print(f"   Heatmap combinado generado")
    
    # Guardar métricas en JSON
    metrics_output = {
        'video_path': video_path,
        'frames_analyzed': frame_count,
        'fps': fps,
        'court_corners': corners,
        'court_dimensions': {
            'width_px': court_width_px,
            'height_px': court_height_px
        },
        'players': {
            str(tid): {
                'total_frames': m['total_frames'],
                'total_distance_m': round(m['total_distance_m'], 2),
                'avg_velocity_m_s': round(m['avg_velocity_m_s'], 2),
                'max_velocity_m_s': round(m['max_velocity_m_s'], 2),
                'sprints_count': m['sprints_count']
            }
            for tid, m in player_metrics.items()
        }
    }
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\n📁 Resultados guardados en {output_dir}/")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN SPIKE 2")
    print(f"{'='*60}")
    print(f"Frames analizados: {frame_count}")
    print(f"Jugadores detectados: {len(valid_trajectories)}")
    
    if player_metrics:
        total_distance = sum(m['total_distance_m'] for m in player_metrics.values())
        avg_speed = np.mean([m['avg_velocity_m_s'] for m in player_metrics.values()])
        max_speed = max(m['max_velocity_m_s'] for m in player_metrics.values())
        
        print(f"\n📊 Métricas agregadas:")
        print(f"   Distancia total (todos): {total_distance:.1f}m")
        print(f"   Velocidad promedio: {avg_speed:.2f} m/s ({avg_speed * 3.6:.1f} km/h)")
        print(f"   Velocidad máxima: {max_speed:.2f} m/s ({max_speed * 3.6:.1f} km/h)")
    
    return {
        'trajectories': valid_trajectories,
        'metrics': player_metrics,
        'frames_analyzed': frame_count
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 (720p, h264).mp4"
    corners_path = "runs/manual_court/court_corners.json"
    
    print("🏃 Spike 2 - Análisis de Movimiento de Jugadores\n")
    print("="*60)
    
    results = analyze_trajectories(
        video_path=video_path,
        corners_path=corners_path,
        output_dir="runs/spike2",
        max_frames=500,
        conf_threshold=0.5
    )