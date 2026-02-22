#!/usr/bin/env python3
"""
Spike 3 - Tracking de Pelota

Detecta y rastrea la pelota de pádel usando:
1. Detección por color HSV (amarillo/verde fluorescente)
2. HoughCircles para detectar forma circular
3. Kalman Filter para suavizar la trayectoria
4. Integración con tracking de jugadores

La pelota de pádel es pequeña (~4cm diámetro) y rápida,
lo que presenta desafíos únicos de detección.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque


# Colores de pelota de pádel típicos (HSV)
# Las pelotas de pádel suelen ser amarillas o verdes fluorescentes
BALL_COLOR_RANGES = {
    'yellow': {
        'lower': np.array([20, 100, 100]),   # Amarillo
        'upper': np.array([40, 255, 255])
    },
    'green': {
        'lower': np.array([35, 100, 100]),   # Verde fluorescente
        'upper': np.array([85, 255, 255])
    },
    'neon_yellow': {
        'lower': np.array([25, 80, 150]),    # Amarillo neón brillante
        'upper': np.array([35, 255, 255])
    }
}

# Parámetros de la pelota
BALL_DIAMETER_CM = 4.0  # Diámetro oficial de pelota de pádel
BALL_RADIUS_RANGE_PX = (3, 15)  # Rango esperado en píxeles

# Parámetros de HoughCircles
HOUGH_PARAM1 = 50   # Umbral para detección de bordes (Canny)
HOUGH_PARAM2 = 20   # Umbral de acumulador (menos = más círculos detectados)
HOUGH_MIN_DIST = 10  # Distancia mínima entre círculos

# Parámetros de Kalman Filter
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1

# Frames máximos sin detectar antes de considerar "perdida"
MAX_FRAMES_MISSING_BALL = 10

# Velocidad máxima válida para pelota de pádel (m/s)
# ~120 km/h es velocidad máxima realista
MAX_BALL_VELOCITY = 30.0

# Velocidad mínima para considerar que la pelota está en movimiento
MIN_BALL_VELOCITY = 0.5


@dataclass
class BallState:
    """Estado de la pelota."""
    position: Optional[Tuple[float, float]] = None
    velocity: Optional[Tuple[float, float]] = None
    last_seen_frame: int = -1
    positions_history: List[Tuple[int, float, float]] = field(default_factory=list)
    kalman: Optional[cv2.KalmanFilter] = None
    
    def __post_init__(self):
        """Inicializa el filtro de Kalman."""
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 estados (x, y, vx, vy), 2 mediciones (x, y)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
    
    def update(self, frame_idx: int, x: float, y: float):
        """Actualiza el estado con una nueva detección."""
        self.position = (x, y)
        self.last_seen_frame = frame_idx
        self.positions_history.append((frame_idx, x, y))
        
        # Actualizar Kalman
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)
        
        # Calcular velocidad
        if len(self.positions_history) >= 2:
            prev_frame, prev_x, prev_y = self.positions_history[-2]
            dt = frame_idx - prev_frame
            if dt > 0:
                self.velocity = ((x - prev_x) / dt, (y - prev_y) / dt)
    
    def predict(self) -> Tuple[float, float]:
        """Predice la siguiente posición usando Kalman."""
        prediction = self.kalman.predict()
        return (prediction[0, 0], prediction[1, 0])
    
    def get_predicted_region(self, search_radius: int = 50) -> Tuple[int, int, int, int]:
        """Obtiene región de búsqueda basada en predicción."""
        pred_x, pred_y = self.predict()
        return (
            int(pred_x - search_radius),
            int(pred_y - search_radius),
            int(pred_x + search_radius),
            int(pred_y + search_radius)
        )


def detect_ball_by_color(frame: np.ndarray, color_range: str = 'yellow') -> List[Tuple[float, float, float]]:
    """
    Detecta la pelota por color HSV.
    
    Returns:
        Lista de (x, y, radius) de posibles pelotas
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if color_range not in BALL_COLOR_RANGES:
        color_range = 'yellow'
    
    lower = BALL_COLOR_RANGES[color_range]['lower']
    upper = BALL_COLOR_RANGES[color_range]['upper']
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operaciones morfológicas para limpiar ruido
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:  # Muy pequeño
            continue
        
        # Obtener círculo mínimo que encierra el contorno
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Verificar que el radio está en rango esperado
        if BALL_RADIUS_RANGE_PX[0] <= radius <= BALL_RADIUS_RANGE_PX[1]:
            # Verificar circularidad
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.5:  # Bastante circular
                    candidates.append((x, y, radius))
    
    return candidates


def detect_ball_by_hough(frame: np.ndarray, search_region: Tuple[int, int, int, int] = None) -> List[Tuple[float, float, float]]:
    """
    Detecta la pelota usando HoughCircles.
    
    Args:
        frame: Frame del video
        search_region: (x1, y1, x2, y2) región de búsqueda, None para toda la imagen
    
    Returns:
        Lista de (x, y, radius) de círculos detectados
    """
    if search_region:
        x1, y1, x2, y2 = search_region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame
        x1, y1 = 0, 0
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detectar círculos
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=BALL_RADIUS_RANGE_PX[0],
        maxRadius=BALL_RADIUS_RANGE_PX[1]
    )
    
    candidates = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            # Ajustar coordenadas si se usó ROI
            candidates.append((x + x1, y + y1, r))
    
    return candidates


def detect_ball_by_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    polygon: np.ndarray,
    ball_state: BallState,
    px_to_m: float,
    fps: float
) -> List[Tuple[float, float, float]]:
    """
    Detecta la pelota por movimiento entre frames.
    
    La pelota en movimiento aparece como un objeto que se desplaza
    entre frames consecutivos.
    
    Returns:
        Lista de (x, y, radius) de candidatos en movimiento
    """
    if prev_frame is None:
        return []
    
    # Convertir a escala de grises
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calcular diferencia absoluta
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Umbral para detectar movimiento
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilatar para conectar regiones
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Encontrar contornos de movimiento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10 or area > 500:  # Muy pequeño o muy grande
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Verificar que esté dentro de la cancha
        if not point_in_polygon((x, y), polygon):
            continue
        
        # Verificar radio
        if BALL_RADIUS_RANGE_PX[0] <= radius <= BALL_RADIUS_RANGE_PX[1]:
            # Verificar que la velocidad sea válida
            if ball_state.position is not None:
                dist_px = distance((x, y), ball_state.position)
                velocity = dist_px * px_to_m * fps
                if velocity > MAX_BALL_VELOCITY:
                    continue
            
            candidates.append((x, y, radius))
    
    return candidates


def detect_ball_combined(
    frame: np.ndarray,
    ball_state: BallState,
    frame_idx: int,
    polygon: np.ndarray,
    prev_frame: np.ndarray = None,
    px_to_m: float = 0.02,
    fps: float = 30.0,
    search_radius: int = 100
) -> Optional[Tuple[float, float, float]]:
    """
    Detecta la pelota combinando color, forma y movimiento.
    
    Prioriza:
    1. Candidatos dentro de la cancha
    2. Candidatos con velocidad válida
    3. Detecciones cerca de la predicción de Kalman
    
    Returns:
        (x, y, radius) de la mejor detección, o None
    """
    all_candidates = []
    
    # 1. Detección por movimiento (más confiable para pelota en juego)
    motion_candidates = detect_ball_by_motion(prev_frame, frame, polygon, ball_state, px_to_m, fps)
    all_candidates.extend(motion_candidates)
    
    # 2. Detección por color (solo dentro de la cancha)
    for color_name in BALL_COLOR_RANGES.keys():
        color_candidates = detect_ball_by_color(frame, color_name)
        for x, y, r in color_candidates:
            if point_in_polygon((x, y), polygon):
                all_candidates.append((x, y, r))
    
    # 3. Detección por Hough (solo dentro de la cancha)
    hough_candidates = detect_ball_by_hough(frame)
    for x, y, r in hough_candidates:
        if point_in_polygon((x, y), polygon):
            all_candidates.append((x, y, r))
    
    if not all_candidates:
        return None
    
    # Filtrar por velocidad válida si hay estado previo
    valid_candidates = []
    for x, y, r in all_candidates:
        if ball_state.position is not None:
            dist_px = distance((x, y), ball_state.position)
            velocity = dist_px * px_to_m * fps
            if velocity <= MAX_BALL_VELOCITY:
                valid_candidates.append((x, y, r))
        else:
            valid_candidates.append((x, y, r))
    
    if not valid_candidates:
        return None
    
    # Si hay estado previo, priorizar candidatos cerca de la predicción
    if ball_state.position is not None:
        pred_x, pred_y = ball_state.predict()
        
        # Calcular score para cada candidato
        scored_candidates = []
        for x, y, r in valid_candidates:
            dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
            score = 1.0 / (1.0 + dist)
            scored_candidates.append((score, x, y, r))
        
        scored_candidates.sort(reverse=True)
        
        # Retornar el mejor candidato dentro del radio de búsqueda
        for score, x, y, r in scored_candidates:
            dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
            if dist < search_radius:
                return (x, y, r)
        
        # Si no hay candidato cercano, permitir cualquier candidato válido
        # (la pelota puede haber rebotado o cambiado de dirección bruscamente)
        if valid_candidates:
            # Retornar el mejor candidato aunque esté lejos de la predicción
            return scored_candidates[0][1], scored_candidates[0][2], scored_candidates[0][3]
    
    # Si no hay estado previo, retornar el candidato más pequeño
    valid_candidates.sort(key=lambda c: c[2])
    return valid_candidates[0] if valid_candidates else None


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    pt = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula distancia euclidiana entre dos puntos."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def analyze_with_ball_tracking(
    video_path: str,
    corners_path: str,
    output_dir: str = "runs/spike3_ball",
    max_frames: int = 500,
    conf_threshold: float = 0.5,
    detect_players: bool = True
) -> Dict:
    """
    Analiza el video con tracking de pelota y jugadores.
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
    court_height_px = np.sqrt((corners['BL'][0] - corners['TL'][0])**2 + 
                              (corners['BL'][1] - corners['TL'][1])**2)
    
    px_to_m = 10.0 / court_width_px
    
    print(f"\n📏 Ancho de cancha: {court_width_px:.1f}px ≈ 10m")
    
    # Cargar modelo para jugadores
    if detect_players:
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
    out = cv2.VideoWriter(f"{output_dir}/visualization_ball.mp4", fourcc, fps, (width, height))
    
    # Estado de la pelota
    ball_state = BallState()
    
    # Métricas de la pelota
    ball_metrics = {
        'total_detections': 0,
        'positions': [],
        'velocities': [],
        'max_velocity': 0,
        'bounces': []
    }
    
    frame_count = 0
    prev_frame = None
    
    print(f"\n🔍 Analizando {max_frames} frames con tracking de pelota...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar cancha
        cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
        
        # Detectar pelota (con nuevos parámetros)
        ball_detection = detect_ball_combined(
            frame, ball_state, frame_count, polygon, 
            prev_frame, px_to_m, fps
        )
        
        if ball_detection:
            bx, by, br = ball_detection
            
            # Verificar que esté dentro o cerca de la cancha
            in_court = point_in_polygon((bx, by), polygon)
            
            # Actualizar estado
            ball_state.update(frame_count, bx, by)
            ball_metrics['total_detections'] += 1
            ball_metrics['positions'].append((frame_count, bx, by))
            
            # Calcular velocidad
            if ball_state.velocity:
                vx, vy = ball_state.velocity
                velocity_mps = np.sqrt(vx**2 + vy**2) * px_to_m * fps
                ball_metrics['velocities'].append((frame_count, velocity_mps))
                ball_metrics['max_velocity'] = max(ball_metrics['max_velocity'], velocity_mps)
                
                # Detectar posible rebote (cambio brusco de dirección)
                if len(ball_metrics['velocities']) >= 3:
                    prev_v = ball_metrics['velocities'][-2][1]
                    if abs(velocity_mps - prev_v) > 5:  # Cambio significativo
                        ball_metrics['bounces'].append((frame_count, bx, by))
            
            # Dibujar pelota
            color = (0, 255, 255) if in_court else (0, 165, 255)  # Cyan o naranja
            cv2.circle(frame, (int(bx), int(by)), int(br), color, 2)
            cv2.circle(frame, (int(bx), int(by)), 3, color, -1)
            
            # Dibujar vector de velocidad
            if ball_state.velocity:
                vx, vy = ball_state.velocity
                scale = 3  # Escalar para visualización
                cv2.line(frame, 
                        (int(bx), int(by)), 
                        (int(bx + vx * scale), int(by + vy * scale)),
                        (0, 255, 0), 2)
        else:
            # Usar predicción de Kalman
            if ball_state.position is not None and (frame_count - ball_state.last_seen_frame) < MAX_FRAMES_MISSING_BALL:
                pred_x, pred_y = ball_state.predict()
                # Dibujar predicción con color diferente
                cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (128, 128, 128), 1)
                cv2.putText(frame, "?", (int(pred_x) - 5, int(pred_y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Dibujar trayectoria de la pelota
        positions = ball_state.positions_history[-100:]  # Últimos 100 frames
        for i in range(1, len(positions)):
            alpha = i / len(positions)
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.line(frame, 
                    (int(positions[i-1][1]), int(positions[i-1][2])),
                    (int(positions[i][1]), int(positions[i][2])),
                    color, 1)
        
        # Panel de métricas
        y_offset = 30
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        detection_rate = ball_metrics['total_detections'] / (frame_count + 1) * 100
        cv2.putText(frame, f"Ball detections: {ball_metrics['total_detections']} ({detection_rate:.1f}%)", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        if ball_metrics['velocities']:
            y_offset += 25
            avg_vel = np.mean([v[1] for v in ball_metrics['velocities']])
            max_vel = ball_metrics['max_velocity']
            cv2.putText(frame, f"Ball velocity: {avg_vel:.1f} m/s (max: {max_vel:.1f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        if ball_metrics['bounces']:
            y_offset += 25
            cv2.putText(frame, f"Bounces detected: {len(ball_metrics['bounces'])}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"   Frame {frame_count}: Ball {'detected' if ball_detection else 'not detected'}")
        
        # Guardar frame actual para la siguiente iteración
        prev_frame = frame.copy()
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Calcular métricas finales
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS CON TRACKING DE PELOTA")
    print(f"{'='*60}")
    
    detection_rate = ball_metrics['total_detections'] / frame_count * 100
    print(f"\n🎾 Pelota:")
    print(f"   Detecciones: {ball_metrics['total_detections']}/{frame_count} ({detection_rate:.1f}%)")
    
    if ball_metrics['velocities']:
        velocities = [v[1] for v in ball_metrics['velocities']]
        print(f"   Velocidad promedio: {np.mean(velocities):.1f} m/s")
        print(f"   Velocidad máxima: {max(velocities):.1f} m/s")
    
    if ball_metrics['bounces']:
        print(f"   Rebotes detectados: {len(ball_metrics['bounces'])}")
    
    # Guardar métricas (convertir tipos numpy a tipos nativos de Python)
    output_metrics = {
        'video_path': video_path,
        'frames_analyzed': int(frame_count),
        'fps': float(fps),
        'court_corners': corners,
        'ball': {
            'total_detections': int(ball_metrics['total_detections']),
            'detection_rate': float(detection_rate),
            'positions': [(int(f), float(x), float(y)) for f, x, y in ball_metrics['positions'][-1000:]],
            'velocities': [(int(f), float(v)) for f, v in ball_metrics['velocities']],
            'max_velocity': float(ball_metrics['max_velocity']),
            'bounces': [(int(f), float(x), float(y)) for f, x, y in ball_metrics['bounces']]
        }
    }
    
    with open(f"{output_dir}/metrics_ball.json", 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    print(f"\n✅ Video guardado en: {output_dir}/visualization_ball.mp4")
    print(f"📁 Métricas guardadas en: {output_dir}/metrics_ball.json")
    
    return {
        'frames_analyzed': frame_count,
        'ball_metrics': output_metrics['ball']
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python spike3_ball_tracking.py <video_path> <corners_path> [output_dir]")
        print("\nEjemplo:")
        print('  python spike3_ball_tracking.py "test-videos/ProPadel2.mp4" "runs/court_propadel2/court_corners.json"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    corners_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "runs/spike3_ball"
    
    print("🎾 Spike 3 - Tracking de Pelota\n")
    print("="*60)
    
    results = analyze_with_ball_tracking(
        video_path=video_path,
        corners_path=corners_path,
        output_dir=output_dir,
        max_frames=500,  # ~16 segundos
        detect_players=False
    )