#!/usr/bin/env python3
"""
Spike 2 - Tracking Estable con Re-asignación de IDs

Mantiene IDs consistentes para los 4 jugadores usando:
1. Posiciones conocidas de cada jugador
2. Re-asignación por proximidad cuando YOLO genera nuevos IDs
3. Zonas de cancha (cada jugador tiene su "zona base")

Estrategia:
- Si un nuevo track_id aparece cerca de donde estaba un jugador conocido → reasignar
- Mantener historial de posiciones por jugador (1-4)
- Usar distancia máxima para considerar "mismo jugador"
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# IDs fijos para los 4 jugadores
PLAYER_IDS = [1, 2, 3, 4]

# Distancia máxima (en pixels) para considerar que es el mismo jugador
MAX_REASSIGN_DISTANCE = 100  # Aproximadamente 2.4m (más restrictivo)

# Máxima velocidad válida en m/s (para filtrar outliers)
MAX_VALID_VELOCITY = 10.0  # ~36 km/h, velocidad máxima realista en pádel

# Frames máximos sin detectar antes de considerar "perdido"
MAX_FRAMES_MISSING = 30  # 1 segundo a 30fps

# Colores para cada jugador
PLAYER_COLORS = {
    1: (0, 255, 0),     # Verde - Jugador 1
    2: (255, 0, 0),     # Azul - Jugador 2  
    3: (0, 0, 255),     # Rojo - Jugador 3
    4: (255, 255, 0),   # Cyan - Jugador 4
}


@dataclass
class PlayerState:
    """Estado actual de un jugador."""
    player_id: int
    last_position: Optional[Tuple[float, float]] = None
    last_seen_frame: int = -1
    current_track_id: Optional[int] = None
    positions_history: List[Tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)
    
    def update_position(self, frame_idx: int, x: float, y: float, track_id: int = None):
        self.last_position = (x, y)
        self.last_seen_frame = frame_idx
        if track_id is not None:
            self.current_track_id = track_id
        self.positions_history.append((frame_idx, x, y))
    
    def get_recent_average_position(self, window: int = 30) -> Optional[Tuple[float, float]]:
        """Obtiene posición promedio de los últimos N frames."""
        if not self.positions_history:
            return None
        recent = self.positions_history[-window:]
        avg_x = np.mean([p[1] for p in recent])
        avg_y = np.mean([p[2] for p in recent])
        return (avg_x, avg_y)


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


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula distancia euclidiana entre dos puntos."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def assign_player_ids(
    detections: List[Dict],
    players_state: Dict[int, PlayerState],
    frame_idx: int,
    max_distance: float = MAX_REASSIGN_DISTANCE
) -> Dict[int, Dict]:
    """
    Asigna IDs de jugador (1-4) a las detecciones del frame actual.
    
    Estrategia:
    1. Si hay menos de 4 jugadores con posición conocida, asignar nuevos
    2. Si todos tienen posición, usarHungarian algorithm o greedy matching
    3. Priorizar distancia a última posición conocida
    
    Returns:
        Dict mapeando player_id -> detección
    """
    
    if not detections:
        return {}
    
    # Obtener jugadores activos (con posición conocida)
    active_players = {
        pid: state for pid, state in players_state.items() 
        if state.last_position is not None
    }
    
    # Si no hay jugadores activos, asignar los primeros 4
    if not active_players:
        assignment = {}
        for i, det in enumerate(detections[:4]):
            player_id = PLAYER_IDS[i]
            assignment[player_id] = det
        return assignment
    
    # Algoritmo greedy de asignación por mínima distancia
    assignment = {}
    used_detections = set()
    
    # Crear matriz de distancias
    det_positions = [(d['center'][0], d['center'][1]) for d in detections]
    player_positions = {pid: state.last_position for pid, state in active_players.items()}
    
    # Asignar por distancia mínima iterativamente
    while len(assignment) < min(len(detections), 4):
        best_dist = float('inf')
        best_player = None
        best_det_idx = None
        
        for pid, ppos in player_positions.items():
            if pid in assignment:
                continue
            for i, dpos in enumerate(det_positions):
                if i in used_detections:
                    continue
                dist = distance(ppos, dpos)
                if dist < best_dist:
                    best_dist = dist
                    best_player = pid
                    best_det_idx = i
        
        if best_player is None or best_det_idx is None:
            break
        
        # Solo asignar si está dentro del rango máximo
        if best_dist <= max_distance:
            assignment[best_player] = detections[best_det_idx]
            used_detections.add(best_det_idx)
        else:
            # Buscar un jugador sin asignar para este detection
            for pid in PLAYER_IDS:
                if pid not in assignment and pid not in player_positions:
                    assignment[pid] = detections[best_det_idx]
                    used_detections.add(best_det_idx)
                    break
            break
    
    # Si quedan detecciones sin asignar y hay jugadores libres
    unassigned_players = [pid for pid in PLAYER_IDS if pid not in assignment]
    for i, det in enumerate(detections):
        if i not in used_detections and unassigned_players:
            player_id = unassigned_players.pop(0)
            assignment[player_id] = det
    
    return assignment


def analyze_with_stable_tracking(
    video_path: str,
    corners_path: str,
    output_dir: str = "runs/spike2_stable",
    max_frames: int = 500,
    conf_threshold: float = 0.5
) -> Dict:
    """
    Analiza el video manteniendo IDs estables para los 4 jugadores.
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
    
    # Calcular dimensiones para conversión a metros
    court_width_px = np.sqrt((corners['TR'][0] - corners['TL'][0])**2 + 
                             (corners['TR'][1] - corners['TL'][1])**2)
    
    print(f"\n📏 Ancho de cancha: {court_width_px:.1f}px ≈ 10m")
    
    # Cargar modelo
    print(f"\n🔄 Cargando modelo YOLOv8m...")
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
    
    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{output_dir}/visualization_stable.mp4", fourcc, fps, (width, height))
    
    # Estado de los 4 jugadores
    players_state: Dict[int, PlayerState] = {pid: PlayerState(player_id=pid) for pid in PLAYER_IDS}
    
    # Métricas por jugador
    player_metrics = {pid: {
        'total_frames': 0,
        'total_distance_px': 0,
        'total_distance_m': 0,
        'velocities': [],
        'positions': []
    } for pid in PLAYER_IDS}
    
    frame_count = 0
    
    print(f"\n🔍 Analizando {max_frames} frames con tracking estable...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar cancha
        cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
        
        # Detectar con YOLO (sin tracking, solo detección)
        results = model.predict(frame, classes=[0], verbose=False, conf=conf_threshold)
        
        frame_detections = []
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) / 2
                    cy_bottom = y2  # Posición de pies
                    
                    in_court = point_in_polygon((cx, cy_bottom), polygon)
                    
                    if in_court:
                        frame_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': (cx, cy_bottom),
                            'confidence': conf
                        })
        
        # Aplicar NMS
        frame_detections = nms(frame_detections, iou_threshold=0.3)
        
        # Limitar a 4 detecciones máximo
        if len(frame_detections) > 4:
            frame_detections = sorted(frame_detections, key=lambda x: x['confidence'], reverse=True)[:4]
        
        # Asignar IDs de jugador
        assignment = assign_player_ids(frame_detections, players_state, frame_count)
        
        # Actualizar estado y métricas
        for player_id, det in assignment.items():
            cx, cy = det['center']
            
            # Actualizar estado
            players_state[player_id].update_position(frame_count, cx, cy)
            
            # Actualizar métricas
            player_metrics[player_id]['total_frames'] += 1
            player_metrics[player_id]['positions'].append((cx, cy))
            
            # Calcular distancia
            if len(player_metrics[player_id]['positions']) >= 2:
                prev_pos = player_metrics[player_id]['positions'][-2]
                dist_px = distance(prev_pos, (cx, cy))
                dist_m = dist_px * (10.0 / court_width_px)
                
                player_metrics[player_id]['total_distance_px'] += dist_px
                player_metrics[player_id]['total_distance_m'] += dist_m
                player_metrics[player_id]['velocities'].append(dist_m * fps)
            
            # Dibujar
            color = PLAYER_COLORS[player_id]
            x1, y1, x2, y2 = det['bbox']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)
            cv2.putText(frame, f"P{player_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Dibujar trayectoria reciente
            positions = player_metrics[player_id]['positions'][-50:]
            for i in range(1, len(positions)):
                alpha = i / len(positions)
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, 
                        (int(positions[i-1][0]), int(positions[i-1][1])),
                        (int(positions[i][0]), int(positions[i][1])),
                        line_color, 2)
        
        # Panel de métricas
        y_offset = 30
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = 60
        for pid in PLAYER_IDS:
            metrics = player_metrics[pid]
            color = PLAYER_COLORS[pid]
            dist_m = metrics['total_distance_m']
            vel = np.mean(metrics['velocities']) if metrics['velocities'] else 0
            
            text = f"P{pid}: {dist_m:.1f}m | {vel:.1f}m/s"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"   Frame {frame_count}: {len(assignment)} jugadores asignados")
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Calcular métricas finales
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS CON TRACKING ESTABLE")
    print(f"{'='*60}")
    
    final_metrics = {}
    
    for pid in PLAYER_IDS:
        metrics = player_metrics[pid]
        velocities = metrics['velocities']
        
        final_metrics[pid] = {
            'total_frames': metrics['total_frames'],
            'total_distance_m': round(metrics['total_distance_m'], 2),
            'avg_velocity_m_s': round(np.mean(velocities), 2) if velocities else 0,
            'max_velocity_m_s': round(max(velocities), 2) if velocities else 0,
            'sprints_count': sum(1 for v in velocities if v > 3.0)
        }
        
        print(f"\n🏃 Jugador {pid}:")
        print(f"   Frames detectados: {metrics['total_frames']}/{frame_count} ({100*metrics['total_frames']/frame_count:.1f}%)")
        print(f"   Distancia total: {metrics['total_distance_m']:.1f}m")
        print(f"   Velocidad promedio: {final_metrics[pid]['avg_velocity_m_s']:.2f} m/s")
        print(f"   Velocidad máxima: {final_metrics[pid]['max_velocity_m_s']:.2f} m/s")
    
    # Guardar métricas
    output_metrics = {
        'video_path': video_path,
        'frames_analyzed': frame_count,
        'fps': fps,
        'court_corners': corners,
        'players': final_metrics
    }
    
    with open(f"{output_dir}/metrics_stable.json", 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    # Generar heatmaps por jugador
    print(f"\n🗺️ Generando heatmaps...")
    
    for pid in PLAYER_IDS:
        positions = player_metrics[pid]['positions']
        if len(positions) > 10:
            # Crear heatmap
            heatmap = np.zeros((100, 100))
            
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = max(x_max - x_min, 1)
            y_range = max(y_max - y_min, 1)
            
            for x, y in positions:
                hx = int((x - x_min) / x_range * 99)
                hy = int((y - y_min) / y_range * 99)
                if 0 <= hx < 100 and 0 <= hy < 100:
                    heatmap[hy, hx] += 1
            
            heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (0, 0), 2)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            heatmap_img = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            cv2.imwrite(f"{output_dir}/heatmap_player_{pid}.png", heatmap_color)
            print(f"   Heatmap generado para jugador {pid}")
    
    print(f"\n✅ Video guardado en: {output_dir}/visualization_stable.mp4")
    print(f"📁 Métricas guardadas en: {output_dir}/metrics_stable.json")
    
    return {
        'frames_analyzed': frame_count,
        'players_metrics': final_metrics
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 (720p, h264).mp4"
    corners_path = "runs/manual_court/court_corners.json"
    
    print("🏃 Spike 2 - Tracking Estable con Re-asignación de IDs\n")
    print("="*60)
    
    results = analyze_with_stable_tracking(
        video_path=video_path,
        corners_path=corners_path,
        output_dir="runs/spike2_stable",
        max_frames=1800,  # ~60 segundos
        conf_threshold=0.5
    )