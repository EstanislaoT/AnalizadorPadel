#!/usr/bin/env python3
"""
Spike 3 - Debug interactivo de detección de pelota

1. Muestra un frame del video
2. El usuario hace clic en la pelota real
3. Prueba diferentes configuraciones y muestra cuáles detectan la pelota
"""

import cv2
import numpy as np
import json
import sys
import os

# Crear directorio de salida
os.makedirs('runs/spike3_ball', exist_ok=True)


def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Extrae un frame específico del video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def detect_ball_by_color(frame: np.ndarray, h_range: tuple, s_range: tuple, v_range: tuple) -> list:
    """Detecta la pelota por color HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array([h_range[0], s_range[0], v_range[0]])
    upper = np.array([h_range[1], s_range[1], v_range[1]])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            candidates.append({
                'x': x, 'y': y, 'radius': radius,
                'area': area, 'circularity': circularity
            })
    
    return candidates, mask


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    pt = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def distance(p1, p2):
    """Calcula distancia euclidiana."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


class BallDebug:
    def __init__(self, video_path, corners_path, frame_idx=100):
        self.video_path = video_path
        self.corners_path = corners_path
        self.frame_idx = frame_idx
        
        # Cargar esquinas
        with open(corners_path, 'r') as f:
            corners = json.load(f)
        
        points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
        self.polygon = points.reshape((-1, 1, 2))
        
        # Posición de la pelota marcada por el usuario
        self.ball_position = None
        self.frame = None
        
        # Configuraciones de color a probar
        self.color_configs = [
            # (nombre, H range, S range, V range)
            ('Amarillo estándar', (20, 40), (100, 255), (100, 255)),
            ('Amarillo brillante', (20, 35), (80, 255), (150, 255)),
            ('Amarillo oscuro', (20, 45), (50, 200), (80, 200)),
            ('Verde fluorescente', (35, 85), (100, 255), (100, 255)),
            ('Amarillo neón', (25, 35), (80, 255), (180, 255)),
            ('Amarillo amplio', (15, 50), (50, 255), (80, 255)),
            ('Amarillo muy amplio', (10, 60), (30, 255), (50, 255)),
            ('Verde-amarillo', (30, 50), (80, 255), (100, 255)),
        ]
        
    def load_frame(self):
        """Carga el frame actual."""
        self.frame = extract_frame(self.video_path, self.frame_idx)
        return self.frame is not None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ball_position = (x, y)
            print(f"\n🎾 Pelota marcada en: ({x}, {y})")
            self.analyze_detection()
    
    def analyze_detection(self):
        """Analiza qué configuraciones detectan la pelota marcada."""
        if self.ball_position is None:
            print("⚠️ Primero marca la pelota con un clic")
            return
        
        print("\n" + "="*60)
        print("📊 ANÁLISIS DE DETECCIÓN")
        print("="*60)
        
        best_configs = []
        
        for name, h_range, s_range, v_range in self.color_configs:
            candidates, mask = detect_ball_by_color(self.frame, h_range, s_range, v_range)
            
            # Filtrar por radio y circularidad
            filtered = [c for c in candidates if 3 <= c['radius'] <= 15 and c['circularity'] > 0.4]
            
            # Buscar el candidato más cercano a la pelota marcada
            best_match = None
            min_dist = float('inf')
            
            for c in filtered:
                d = distance((c['x'], c['y']), self.ball_position)
                if d < min_dist:
                    min_dist = d
                    best_match = c
            
            # Verificar si detectó la pelota (distancia < 20px)
            detected = min_dist < 20 if best_match else False
            
            if detected:
                best_configs.append({
                    'name': name,
                    'config': (h_range, s_range, v_range),
                    'distance': min_dist,
                    'candidate': best_match,
                    'total': len(filtered)
                })
            
            status = "✅ DETECTA" if detected else "❌ No detecta"
            print(f"\n{name}:")
            print(f"  Candidatos: {len(filtered)}")
            if best_match:
                print(f"  Más cercano: ({best_match['x']:.0f}, {best_match['y']:.0f}), dist={min_dist:.1f}px")
            print(f"  {status}")
        
        # Mostrar mejores configuraciones
        if best_configs:
            best_configs.sort(key=lambda x: x['distance'])
            print("\n" + "="*60)
            print("🏆 MEJORES CONFIGURACIONES:")
            for i, cfg in enumerate(best_configs[:3]):
                print(f"\n{i+1}. {cfg['name']} (dist={cfg['distance']:.1f}px)")
                h, s, v = cfg['config']
                print(f"   H: {h}, S: {s}, V: {v}")
                print(f"   Candidato: ({cfg['candidate']['x']:.0f}, {cfg['candidate']['y']:.0f}), "
                      f"r={cfg['candidate']['radius']:.1f}")
        else:
            print("\n⚠️ NINGUNA configuración detectó la pelota")
            print("   Posibles causas:")
            print("   - La pelota no es amarilla/verde en este frame")
            print("   - El radio esperado (3-15px) no es correcto")
            print("   - La pelota está parcialmente oculta")
    
    def run(self):
        """Ejecuta el debug interactivo."""
        print("🎾 Spike 3 - Debug Interactivo de Pelota")
        print("="*60)
        print("\nInstrucciones:")
        print("  - Haz CLIC IZQUIERDO en la pelota para marcarla")
        print("  - Presiona 'n' para siguiente frame")
        print("  - Presiona 'p' para frame anterior")
        print("  - Presiona 'q' para salir")
        print("  - Presiona 's' para guardar el frame")
        
        cv2.namedWindow('Ball Debug')
        cv2.setMouseCallback('Ball Debug', self.mouse_callback)
        
        self.load_frame()
        
        while True:
            if self.frame is None:
                print("❌ No se pudo cargar el frame")
                break
            
            # Mostrar frame
            display = self.frame.copy()
            
            # Dibujar cancha
            cv2.polylines(display, [self.polygon], True, (255, 255, 0), 2)
            
            # Dibujar pelota marcada
            if self.ball_position:
                cv2.circle(display, self.ball_position, 15, (0, 255, 0), 2)
                cv2.circle(display, self.ball_position, 3, (0, 255, 0), -1)
                cv2.putText(display, f"Ball: {self.ball_position}", 
                           (self.ball_position[0] + 10, self.ball_position[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Mostrar frame actual
            cv2.putText(display, f"Frame: {self.frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Ball Debug', display)
            
            # Esperar tecla
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.frame_idx += 10
                self.ball_position = None
                self.load_frame()
                print(f"\n➡️ Frame {self.frame_idx}")
            elif key == ord('p'):
                self.frame_idx = max(0, self.frame_idx - 10)
                self.ball_position = None
                self.load_frame()
                print(f"\n⬅️ Frame {self.frame_idx}")
            elif key == ord('s'):
                path = f'runs/spike3_ball/debug_frame_{self.frame_idx}.jpg'
                cv2.imwrite(path, display)
                print(f"💾 Frame guardado en: {path}")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python spike3_ball_interactive.py <video_path> <corners_path> [frame_idx]")
        print("\nEjemplo:")
        print('  python spike3_ball_interactive.py "test-videos/ProPadel2.mp4" "runs/court_propadel2_test/court_corners.json" 100')
        sys.exit(1)
    
    video_path = sys.argv[1]
    corners_path = sys.argv[2]
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    debug = BallDebug(video_path, corners_path, frame_idx)
    debug.run()