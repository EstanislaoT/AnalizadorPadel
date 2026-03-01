#!/usr/bin/env python3
"""
Spike 3 - Tracking de Pelota con YOLOS

Usa YOLOS (YOLO for Object Detection with Transformers) de HuggingFace
para detectar la pelota de pádel.

YOLOS demostró mejor rendimiento que YOLOv8 para pelotas pequeñas.
"""

import cv2
import numpy as np
import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

os.makedirs('runs/spike3_ball', exist_ok=True)

# Cargar modelo YOLOS globalmente para evitar recargas
_yolos_model = None
_yolos_processor = None


def get_yolos_model():
    """Carga el modelo YOLOS de HuggingFace."""
    global _yolos_model, _yolos_processor
    
    if _yolos_model is None:
        from transformers import YolosImageProcessor, YolosForObjectDetection
        import torch
        
        print("🔄 Cargando YOLOS...")
        _yolos_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
        _yolos_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
        
        # Mover a GPU si está disponible
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _yolos_model = _yolos_model.to(device)
        print(f"   Modelo cargado en: {device}")
    
    return _yolos_processor, _yolos_model


def detect_ball_yolos(frame: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
    """
    Detecta la pelota usando YOLOS.
    
    Args:
        frame: Frame en formato BGR (OpenCV)
        confidence_threshold: Umbral mínimo de confianza
    
    Returns:
        Lista de detecciones con (x, y, radius, confidence)
    """
    import torch
    
    processor, model = get_yolos_model()
    
    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preparar input
    inputs = processor(images=frame_rgb, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Inferencia
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-procesar
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]
    
    balls = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        
        # COCO class 37 = sports ball
        if 'ball' in label_name.lower() or label.item() == 37:
            box = box.tolist()
            x1, y1, x2, y2 = box
            
            # Centro y radio aproximado
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            radius = (width + height) / 4
            
            balls.append({
                'x': cx, 'y': cy, 'radius': radius,
                'confidence': score.item(),
                'width': width, 'height': height
            })
    
    # Ordenar por confianza
    balls.sort(key=lambda x: x['confidence'], reverse=True)
    
    return balls


@dataclass
class BallState:
    """Estado de la pelota con Kalman Filter."""
    position: Optional[Tuple[float, float]] = None
    velocity: Optional[Tuple[float, float]] = None
    last_seen_frame: int = -1
    positions_history: List[Tuple[int, float, float]] = field(default_factory=list)
    kalman: Optional[cv2.KalmanFilter] = None
    
    def __post_init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
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
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    
    def update(self, frame_idx: int, x: float, y: float):
        """Actualiza el estado con nueva detección."""
        self.position = (x, y)
        self.last_seen_frame = frame_idx
        self.positions_history.append((frame_idx, x, y))
        
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)
        
        if len(self.positions_history) >= 2:
            prev_frame, prev_x, prev_y = self.positions_history[-2]
            dt = frame_idx - prev_frame
            if dt > 0:
                self.velocity = ((x - prev_x) / dt, (y - prev_y) / dt)
    
    def predict(self) -> Tuple[float, float]:
        """Predice siguiente posición."""
        prediction = self.kalman.predict()
        return (prediction[0, 0], prediction[1, 0])


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    pt = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def analyze_with_yolos(
    video_path: str,
    corners_path: str,
    output_dir: str = "runs/spike3_ball",
    max_frames: int = 500,
    confidence_threshold: float = 0.3
) -> Dict:
    """
    Analiza el video con YOLOS para tracking de pelota.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar esquinas
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    print(f"📐 Cancha cargada:")
    print(f"   TL: {corners['TL']}")
    print(f"   TR: {corners['TR']}")
    print(f"   BR: {corners['BR']}")
    print(f"   BL: {corners['BL']}")
    
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    # Dimensiones
    court_width_px = np.sqrt((corners['TR'][0] - corners['TL'][0])**2 + 
                             (corners['TR'][1] - corners['TL'][1])**2)
    px_to_m = 10.0 / court_width_px
    
    # Abrir video
    print(f"\n📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{output_dir}/yolos_tracking.mp4", fourcc, fps, (width, height))
    
    # Estado de la pelota
    ball_state = BallState()
    
    # Métricas
    metrics = {
        'total_detections': 0,
        'positions': [],
        'velocities': [],
        'max_velocity': 0
    }
    
    frame_count = 0
    
    print(f"\n🔍 Analizando {max_frames} frames con YOLOS...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dibujar cancha
        cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
        
        # Detectar pelota
        detections = detect_ball_yolos(frame, confidence_threshold)
        
        # Filtrar por posición en cancha
        in_court_detections = [d for d in detections if point_in_polygon((d['x'], d['y']), polygon)]
        
        if in_court_detections:
            best = in_court_detections[0]  # Ya ordenado por confianza
            
            ball_state.update(frame_count, best['x'], best['y'])
            metrics['total_detections'] += 1
            metrics['positions'].append((frame_count, best['x'], best['y']))
            
            # Calcular velocidad
            if ball_state.velocity:
                vx, vy = ball_state.velocity
                velocity_mps = np.sqrt(vx**2 + vy**2) * px_to_m * fps
                metrics['velocities'].append((frame_count, velocity_mps))
                metrics['max_velocity'] = max(metrics['max_velocity'], velocity_mps)
            
            # Dibujar detección
            color = (0, 255, 0)  # Verde
            cv2.circle(frame, (int(best['x']), int(best['y'])), int(best['radius']), color, 2)
            cv2.circle(frame, (int(best['x']), int(best['y'])), 3, color, -1)
            
            # Dibujar confianza
            cv2.putText(frame, f"{best['confidence']:.0%}", 
                       (int(best['x']) + 10, int(best['y']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # Usar predicción de Kalman si tenemos estado previo
            if ball_state.position is not None and (frame_count - ball_state.last_seen_frame) < 10:
                pred_x, pred_y = ball_state.predict()
                cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (128, 128, 128), 1)
        
        # Dibujar trayectoria
        positions = ball_state.positions_history[-50:]
        for i in range(1, len(positions)):
            cv2.line(frame, 
                    (int(positions[i-1][1]), int(positions[i-1][2])),
                    (int(positions[i][1]), int(positions[i][2])),
                    (0, 255, 255), 1)
        
        # Panel de métricas
        y = 30
        cv2.putText(frame, f"Frame: {frame_count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y += 30
        rate = metrics['total_detections'] / (frame_count + 1) * 100
        cv2.putText(frame, f"Ball detections: {metrics['total_detections']} ({rate:.1f}%)", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if metrics['velocities']:
            y += 25
            avg_vel = np.mean([v[1] for v in metrics['velocities']])
            cv2.putText(frame, f"Avg velocity: {avg_vel:.1f} m/s", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        
        if frame_count % 50 == 0:
            print(f"   Frame {frame_count}: {len(in_court_detections)} balls detected")
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Resultados
    print(f"\n{'='*60}")
    print("📊 RESULTADOS YOLOS")
    print(f"{'='*60}")
    
    rate = metrics['total_detections'] / frame_count * 100
    print(f"\n🎾 Pelota:")
    print(f"   Detecciones: {metrics['total_detections']}/{frame_count} ({rate:.1f}%)")
    
    if metrics['velocities']:
        velocities = [v[1] for v in metrics['velocities']]
        print(f"   Velocidad promedio: {np.mean(velocities):.1f} m/s")
        print(f"   Velocidad máxima: {max(velocities):.1f} m/s")
    
    # Guardar métricas
    output_metrics = {
        'video_path': video_path,
        'frames_analyzed': int(frame_count),
        'fps': float(fps),
        'model': 'YOLOS (hustvl/yolos-small)',
        'confidence_threshold': confidence_threshold,
        'ball': {
            'total_detections': int(metrics['total_detections']),
            'detection_rate': float(rate),
            'positions': [(int(f), float(x), float(y)) for f, x, y in metrics['positions'][-500:]],
            'max_velocity': float(metrics['max_velocity'])
        }
    }
    
    with open(f"{output_dir}/yolos_metrics.json", 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    print(f"\n✅ Video guardado en: {output_dir}/yolos_tracking.mp4")
    print(f"📁 Métricas guardadas en: {output_dir}/yolos_metrics.json")
    
    return output_metrics


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python spike3_ball_yolos.py <video_path> <corners_path> [output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    corners_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "runs/spike3_ball"
    
    print("🎾 Spike 3 - Tracking de Pelota con YOLOS")
    print("=" * 60)
    
    results = analyze_with_yolos(
        video_path=video_path,
        corners_path=corners_path,
        output_dir=output_dir,
        max_frames=500,
        confidence_threshold=0.3
    )