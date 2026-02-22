#!/usr/bin/env python3
"""
Herramienta de Calibración Visual de Cancha v2

El usuario puede arrastrar directamente cada una de las 4 esquinas de la cancha
para ajustarla al video. El sistema mantiene la perspectiva en tiempo real.

Mejoras:
- Puntos transparentes con cruz para mejor precisión
- Líneas de servicio completas
- Línea divisoria de lados
- Puntos de red ajustables

Uso:
    python spike_court_calibration_v2.py <video_path> [output_dir]
"""

import cv2
import numpy as np
import json
import os
import sys


class CourtCalibratorV2:
    """Calibración de cancha con arrastre directo de esquinas."""
    
    def __init__(self, video_path: str, output_dir: str = "runs/court_calibrated"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar primer frame del video
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, frame = self.cap.read()
        self.cap.release()
        
        if not ret:
            raise ValueError(f"No se pudo leer el video: {video_path}")
        
        self.original_frame = frame.copy()
        
        # 4 puntos de la cancha (TL, TR, BR, BL)
        cx, cy = self.width // 2, self.height // 2
        margin = min(self.width, self.height) // 4
        
        self.corners = {
            'TL': [cx - margin, cy - margin * 2],
            'TR': [cx + margin, cy - margin * 2],
            'BR': [cx + margin, cy + margin * 2],
            'BL': [cx - margin, cy + margin * 2]
        }
        
        # Puntos de la red (ajustables por el usuario)
        self.net_corners = None
        self.dragging_net_corner = None
        self.net_corner_radius = 12
        
        # Estado de arrastre
        self.dragging_corner = None
        self.corner_radius = 15
        
        # Ventana
        self.window_name = "Calibración de Cancha v2 - Arrastra las esquinas"
        
        # Calcular puntos de red iniciales
        self._calculate_initial_net_points()
    
    def _calculate_initial_net_points(self):
        """Calcula los puntos iniciales de la red."""
        src_points = np.array([
            self.corners['TL'], self.corners['TR'],
            self.corners['BR'], self.corners['BL']
        ], dtype=np.float32)
        
        dst_points = np.array([
            [0, 0], [10, 0], [10, 20], [0, 20]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(dst_points, src_points)
        
        if H is not None:
            net_left_dst = np.array([[0, 10]], dtype=np.float32).reshape(-1, 1, 2)
            net_right_dst = np.array([[10, 10]], dtype=np.float32).reshape(-1, 1, 2)
            net_left = cv2.perspectiveTransform(net_left_dst, H).reshape(2)
            net_right = cv2.perspectiveTransform(net_right_dst, H).reshape(2)
            self.net_corners = {
                'NL': [int(net_left[0]), int(net_left[1])],
                'NR': [int(net_right[0]), int(net_right[1])]
            }
        else:
            self.net_corners = {
                'NL': [(self.corners['TL'][0] + self.corners['BL'][0]) // 2,
                       (self.corners['TL'][1] + self.corners['BL'][1]) // 2],
                'NR': [(self.corners['TR'][0] + self.corners['BR'][0]) // 2,
                       (self.corners['TR'][1] + self.corners['BR'][1]) // 2]
            }
    
    def get_corners_array(self) -> np.ndarray:
        """Retorna las esquinas como array numpy."""
        return np.array([
            self.corners['TL'], self.corners['TR'],
            self.corners['BR'], self.corners['BL']
        ], dtype=np.int32)
    
    def get_corner_at(self, x: int, y: int) -> str:
        """Retorna el nombre de la esquina si el punto está cerca."""
        # Primero verificar esquinas de la cancha
        for name, pt in self.corners.items():
            dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
            if dist <= self.corner_radius:
                return ('corner', name)
        
        # Luego verificar puntos de la red
        if self.net_corners:
            for name, pt in self.net_corners.items():
                dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                if dist <= self.net_corner_radius:
                    return ('net', name)
        
        return None
    
    def draw_transparent_point_with_cross(self, frame, x, y, radius, color, label):
        """
        Dibuja un punto transparente con una cruz en el medio.
        Facilita la precisión al usuario.
        """
        x, y = int(x), int(y)
        
        # Crear overlay para el círculo semi-transparente
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Borde del círculo
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
        
        # Cruz en el centro (más precisa)
        cross_size = radius - 2
        cv2.line(frame, (x - cross_size, y), (x + cross_size, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y - cross_size), (x, y + cross_size), (255, 255, 255), 2)
        
        # Etiqueta
        cv2.putText(frame, label, (x - 10, y - radius - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja la cancha con las esquinas actuales."""
        points = self.get_corners_array()
        
        # Overlay semi-transparente para la cancha
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # Borde de la cancha
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # Calcular homografía para las líneas
        src_points = np.array([
            self.corners['TL'], self.corners['TR'],
            self.corners['BR'], self.corners['BL']
        ], dtype=np.float32)
        
        dst_points = np.array([
            [0, 0], [10, 0], [10, 20], [0, 20]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(dst_points, src_points)
        
        if H is not None:
            # === LÍNEA DE LA RED (usando puntos ajustables) ===
            if self.net_corners:
                net_left = (self.net_corners['NL'][0], self.net_corners['NL'][1])
                net_right = (self.net_corners['NR'][0], self.net_corners['NR'][1])
                cv2.line(frame, net_left, net_right, (255, 255, 255), 3)
                cv2.putText(frame, "RED", 
                           ((net_left[0] + net_right[0]) // 2 - 20, net_left[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # === LÍNEA DIVISORIA VERTICAL (divide ambos lados) ===
            # Centro de la línea superior y centro de la línea inferior
            center_top_dst = np.array([[5, 0]], dtype=np.float32).reshape(-1, 1, 2)
            center_bottom_dst = np.array([[5, 20]], dtype=np.float32).reshape(-1, 1, 2)
            center_top = cv2.perspectiveTransform(center_top_dst, H).reshape(2)
            center_bottom = cv2.perspectiveTransform(center_bottom_dst, H).reshape(2)
            cv2.line(frame, (int(center_top[0]), int(center_top[1])),
                    (int(center_bottom[0]), int(center_bottom[1])), (200, 200, 200), 1)
            
            # === LÍNEAS DE SERVICIO ===
            # CORREGIDO: Línea de servicio está a 3.05m del fondo, 6.95m de la red
            # Línea de servicio superior: Y = 3.05m
            # Línea de servicio inferior: Y = 16.95m (20 - 3.05)
            # Ambas líneas van de lado a lado COMPLETO (X=0 hasta X=10)
            
            # Línea de servicio superior - COMPLETA de lado a lado
            service_top_left_dst = np.array([[0, 3.05]], dtype=np.float32).reshape(-1, 1, 2)
            service_top_right_dst = np.array([[10, 3.05]], dtype=np.float32).reshape(-1, 1, 2)
            service_top_left = cv2.perspectiveTransform(service_top_left_dst, H).reshape(2)
            service_top_right = cv2.perspectiveTransform(service_top_right_dst, H).reshape(2)
            cv2.line(frame, (int(service_top_left[0]), int(service_top_left[1])),
                    (int(service_top_right[0]), int(service_top_right[1])), (255, 255, 0), 2)
            
            # Línea de servicio inferior - COMPLETA de lado a lado
            service_bottom_left_dst = np.array([[0, 16.95]], dtype=np.float32).reshape(-1, 1, 2)
            service_bottom_right_dst = np.array([[10, 16.95]], dtype=np.float32).reshape(-1, 1, 2)
            service_bottom_left = cv2.perspectiveTransform(service_bottom_left_dst, H).reshape(2)
            service_bottom_right = cv2.perspectiveTransform(service_bottom_right_dst, H).reshape(2)
            cv2.line(frame, (int(service_bottom_left[0]), int(service_bottom_left[1])),
                    (int(service_bottom_right[0]), int(service_bottom_right[1])), (255, 255, 0), 2)
            
            # === LÍNEAS LATERALES DE SERVICIO ===
            # Conectan las esquinas con las líneas de servicio
            # Lado superior izquierdo (TL a línea de servicio)
            cv2.line(frame, 
                    (int(self.corners['TL'][0]), int(self.corners['TL'][1])),
                    (int(service_top_left[0]), int(service_top_left[1])), 
                    (255, 255, 0), 2)
            # Lado superior derecho (TR a línea de servicio)
            cv2.line(frame,
                    (int(self.corners['TR'][0]), int(self.corners['TR'][1])),
                    (int(service_top_right[0]), int(service_top_right[1])),
                    (255, 255, 0), 2)
            # Lado inferior izquierdo (BL a línea de servicio)
            cv2.line(frame,
                    (int(self.corners['BL'][0]), int(self.corners['BL'][1])),
                    (int(service_bottom_left[0]), int(service_bottom_left[1])),
                    (255, 255, 0), 2)
            # Lado inferior derecho (BR a línea de servicio)
            cv2.line(frame,
                    (int(self.corners['BR'][0]), int(self.corners['BR'][1])),
                    (int(service_bottom_right[0]), int(service_bottom_right[1])),
                    (255, 255, 0), 2)
        
        # === DIBUJAR ESQUINAS DE LA CANCHA (transparentes con cruz) ===
        corner_colors = {
            'TL': (0, 0, 255),    # Rojo
            'TR': (255, 0, 0),    # Azul
            'BR': (255, 0, 255),  # Magenta
            'BL': (0, 255, 255)   # Amarillo
        }
        
        for name, pt in self.corners.items():
            self.draw_transparent_point_with_cross(
                frame, pt[0], pt[1], self.corner_radius, corner_colors[name], name
            )
        
        # === DIBUJAR PUNTOS DE LA RED (transparentes con cruz) ===
        if self.net_corners:
            net_colors = {
                'NL': (0, 165, 255),  # Naranja
                'NR': (255, 165, 0)   # Cyan
            }
            for name, pt in self.net_corners.items():
                self.draw_transparent_point_with_cross(
                    frame, pt[0], pt[1], self.net_corner_radius, net_colors[name], name
                )
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja instrucciones."""
        instructions = [
            "CONTROLES:",
            "  Arrastra los puntos para ajustar",
            "  R: Reiniciar",
            "  ENTER: Guardar",
            "  ESC: Cancelar",
            "",
            "Cancha:",
            "  TL(rojo) TR(azul)",
            "  BR(magenta) BL(amarillo)",
            "",
            "Red:",
            "  NL(naranja) NR(cyan)",
        ]
        
        y = 25
        for text in instructions:
            cv2.putText(frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 18
        
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            result = self.get_corner_at(x, y)
            if result:
                corner_type, corner_name = result
                if corner_type == 'corner':
                    self.dragging_corner = corner_name
                elif corner_type == 'net':
                    self.dragging_net_corner = corner_name
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_corner = None
            self.dragging_net_corner = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_corner:
                self.corners[self.dragging_corner] = [x, y]
                # Recalcular puntos de red cuando se mueven las esquinas
                self._calculate_initial_net_points()
            elif self.dragging_net_corner:
                self.net_corners[self.dragging_net_corner] = [x, y]
    
    def run(self) -> dict:
        """Ejecuta la herramienta de calibración."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("🎯 CALIBRACIÓN VISUAL DE CANCHA v2")
        print("="*60)
        print("\nArrastra los puntos para ajustar la cancha.")
        print("Presiona ENTER cuando estés satisfecho.\n")
        
        while True:
            frame = self.original_frame.copy()
            frame = self.draw_court(frame)
            frame = self.draw_instructions(frame)
            
            cv2.imshow(self.window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == ord('\n'):  # ENTER
                corners_output = {
                    'TL': self.corners['TL'],
                    'TR': self.corners['TR'],
                    'BR': self.corners['BR'],
                    'BL': self.corners['BL']
                }
                
                # Guardar imagen
                result_frame = self.original_frame.copy()
                result_frame = self.draw_court(result_frame)
                cv2.imwrite(f"{self.output_dir}/court_calibration.jpg", result_frame)
                
                # Guardar JSON
                with open(f"{self.output_dir}/court_corners.json", 'w') as f:
                    json.dump(corners_output, f, indent=2)
                
                print(f"\n✅ Cancha calibrada:")
                print(f"   TL: {self.corners['TL']}")
                print(f"   TR: {self.corners['TR']}")
                print(f"   BR: {self.corners['BR']}")
                print(f"   BL: {self.corners['BL']}")
                print(f"\n📁 Guardado en {self.output_dir}/")
                
                cv2.destroyAllWindows()
                return corners_output
                
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
                
            elif key == ord('r') or key == ord('R'):
                cx, cy = self.width // 2, self.height // 2
                margin = min(self.width, self.height) // 4
                self.corners = {
                    'TL': [cx - margin, cy - margin * 2],
                    'TR': [cx + margin, cy - margin * 2],
                    'BR': [cx + margin, cy + margin * 2],
                    'BL': [cx - margin, cy + margin * 2]
                }
                self._calculate_initial_net_points()
                print("   Reiniciado.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python spike_court_calibration_v2.py <video_path> [output_dir]")
        print("\nEjemplo:")
        print('  python spike_court_calibration_v2.py "test-videos/ProPadel2.mp4" "runs/court_propadel2"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "runs/court_calibrated"
    
    print("🎯 Calibración de Cancha v2 - Arrastra las esquinas\n")
    
    calibrator = CourtCalibratorV2(video_path, output_dir)
    corners = calibrator.run()
    
    if corners:
        print("\n" + "="*60)
        print("✅ CALIBRACIÓN GUARDADA")
        print(f"   Usá {output_dir}/court_corners.json para el análisis")
        print("="*60)