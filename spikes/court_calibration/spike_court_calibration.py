#!/usr/bin/env python3
"""
Herramienta de Calibración Visual de Cancha

Permite al usuario superponer un modelo de cancha de pádel sobre el video
y ajustarlo visualmente mediante:
- Mover (arrastrar)
- Escalar (scroll)
- Perspectiva (teclas Q/E)

Uso:
    python spike_court_calibration.py <video_path> [output_dir]
"""

import cv2
import numpy as np
import json
import os
import sys


class CourtCalibrator:
    """Herramienta interactiva para calibrar la cancha de pádel."""
    
    # Dimensiones estándar de cancha de pádel (proporción 2:1)
    COURT_ASPECT_RATIO = 2.0  # largo / ancho = 20m / 10m
    
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
        
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        
        # Parámetros del modelo de cancha (iniciales)
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.scale = min(self.width, self.height) // 4  # Tamaño inicial
        self.perspective = 0.0  # -5 a 5 (inclinación hacia atrás/adelante) - rango ampliado
        
        # Estado de interacción
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_offset = (0, 0)
        
        # Ventana
        self.window_name = "Calibración de Cancha - Superponer modelo"
        
    def get_court_points(self) -> np.ndarray:
        """Calcula los 4 puntos de la cancha según parámetros actuales."""
        # Ancho y largo del modelo
        half_width = self.scale
        half_height = self.scale * self.COURT_ASPECT_RATIO / 2
        
        # Aplicar perspectiva (inclinación)
        # perspective > 0: parte superior más pequeña (cámara baja)
        # perspective < 0: parte superior más grande (cámara alta)
        perspective_factor = 1.0 + self.perspective * 0.5
        
        top_width = half_width / perspective_factor
        bottom_width = half_width * perspective_factor
        
        # Puntos en orden: TL, TR, BR, BL
        points = np.array([
            [self.center_x - top_width, self.center_y - half_height],      # TL
            [self.center_x + top_width, self.center_y - half_height],      # TR
            [self.center_x + bottom_width, self.center_y + half_height],   # BR
            [self.center_x - bottom_width, self.center_y + half_height],   # BL
        ], dtype=np.float32)
        
        return points
    
    def draw_court_model(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja el modelo de cancha sobre el frame."""
        points = self.get_court_points()
        points_int = points.astype(np.int32)
        
        # Crear overlay
        overlay = frame.copy()
        
        # Rellenar cancha con color semi-transparente
        cv2.fillPoly(overlay, [points_int], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Dibujar borde de la cancha
        cv2.polylines(frame, [points_int], True, (0, 255, 0), 2)
        
        # Dibujar líneas interiores de la cancha
        # Línea de la RED - Horizontal en el centro de la cancha
        # La red está en la línea central que divide la cancha en dos mitades
        center_left_y = points_int[0][1] + (points_int[3][1] - points_int[0][1]) // 2
        center_right_y = points_int[1][1] + (points_int[2][1] - points_int[1][1]) // 2
        
        # Puntos de la línea de red (horizontal, en el medio)
        net_left = (points_int[0][0] + (points_int[3][0] - points_int[0][0]) // 2, 
                    (points_int[0][1] + points_int[3][1]) // 2)
        net_right = (points_int[1][0] + (points_int[2][0] - points_int[1][0]) // 2,
                     (points_int[1][1] + points_int[2][1]) // 2)
        
        # Línea de red más gruesa y de color diferente (blanco) - HORIZONTAL
        cv2.line(frame, net_left, net_right, (255, 255, 255), 4)
        cv2.putText(frame, "RED", ((net_left[0] + net_right[0]) // 2 - 20, net_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Línea vertical central (división de lados)
        top_center = ((points_int[0][0] + points_int[1][0]) // 2,
                      (points_int[0][1] + points_int[1][1]) // 2)
        bottom_center = ((points_int[2][0] + points_int[3][0]) // 2,
                         (points_int[2][1] + points_int[3][1]) // 2)
        cv2.line(frame, top_center, bottom_center, (200, 200, 200), 1)
        
        # Líneas de servicio
        # Línea de servicio superior
        service_y_top = int(points_int[0][1] + (points_int[3][1] - points_int[0][1]) * 0.25)
        service_y_bottom = int(points_int[0][1] + (points_int[3][1] - points_int[0][1]) * 0.75)
        
        # Calcular ancho en las líneas de servicio
        ratio_top = (service_y_top - points_int[0][1]) / (points_int[3][1] - points_int[0][1])
        ratio_bottom = (service_y_bottom - points_int[0][1]) / (points_int[3][1] - points_int[0][1])
        
        width_at_service_top = points_int[0][0] + ratio_top * (points_int[3][0] - points_int[0][0])
        width_at_service_top2 = points_int[1][0] + ratio_top * (points_int[2][0] - points_int[1][0])
        
        width_at_service_bottom = points_int[0][0] + ratio_bottom * (points_int[3][0] - points_int[0][0])
        width_at_service_bottom2 = points_int[1][0] + ratio_bottom * (points_int[2][0] - points_int[1][0])
        
        cv2.line(frame, (int(width_at_service_top), service_y_top),
                (int(width_at_service_top2), service_y_top), (255, 255, 0), 1)
        cv2.line(frame, (int(width_at_service_bottom), service_y_bottom),
                (int(width_at_service_bottom2), service_y_bottom), (255, 255, 0), 1)
        
        # Etiquetar esquinas
        labels = ['TL', 'TR', 'BR', 'BL']
        colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
        for i, (pt, label, color) in enumerate(zip(points_int, labels, colors)):
            cv2.circle(frame, tuple(pt), 5, color, -1)
            cv2.putText(frame, label, (pt[0] - 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja instrucciones en pantalla."""
        instructions = [
            "CONTROLES:",
            "  Click + Arrastrar: Mover cancha",
            "  W/S: Aumentar/Disminuir escala",
            "  Q/A: Perspectiva hacia adelante",
            "  E/D: Perspectiva hacia atrás",
            "  R: Reiniciar",
            "  ENTER: Guardar y salir",
            "  ESC: Cancelar",
            "",
            f"  Escala: {self.scale:.0f}",
            f"  Perspectiva: {self.perspective:.2f}",
        ]
        
        y = 30
        for text in instructions:
            cv2.putText(frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x - self.drag_offset[0], y - self.drag_offset[1])
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.center_x = x - self.drag_start[0]
            self.center_y = y - self.drag_start[1]
            self.drag_offset = (x - self.drag_start[0], y - self.drag_start[1])
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll para escalar - usar flags directamente
            # flags > 0: scroll hacia arriba (agrandar)
            # flags < 0: scroll hacia abajo (achicar)
            if flags > 0:
                # Scroll hacia arriba - agrandar
                self.scale = min(self.scale + 15, max(self.width, self.height) * 2)
            else:
                # Scroll hacia abajo - achicar
                self.scale = max(self.scale - 15, 50)
    
    def run(self) -> dict:
        """Ejecuta la herramienta de calibración."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("🎯 CALIBRACIÓN VISUAL DE CANCHA")
        print("="*60)
        print("\nAlinea el modelo verde con la cancha del video.")
        print("Presiona ENTER cuando estés satisfecho.\n")
        
        while True:
            # Copiar frame original
            frame = self.original_frame.copy()
            
            # Dibujar modelo de cancha
            frame = self.draw_court_model(frame)
            
            # Dibujar instrucciones
            frame = self.draw_instructions(frame)
            
            cv2.imshow(self.window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == ord('\n'):  # ENTER
                # Guardar y salir
                points = self.get_court_points()
                corners = {
                    'TL': [float(points[0][0]), float(points[0][1])],
                    'TR': [float(points[1][0]), float(points[1][1])],
                    'BR': [float(points[2][0]), float(points[2][1])],
                    'BL': [float(points[3][0]), float(points[3][1])],
                    'scale': self.scale,
                    'perspective': self.perspective
                }
                
                # Guardar imagen con resultado
                result_frame = self.original_frame.copy()
                result_frame = self.draw_court_model(result_frame)
                cv2.imwrite(f"{self.output_dir}/court_calibration.jpg", result_frame)
                
                # Guardar coordenadas
                with open(f"{self.output_dir}/court_corners.json", 'w') as f:
                    json.dump(corners, f, indent=2)
                
                print(f"\n✅ Cancha calibrada:")
                print(f"   TL: ({corners['TL'][0]:.1f}, {corners['TL'][1]:.1f})")
                print(f"   TR: ({corners['TR'][0]:.1f}, {corners['TR'][1]:.1f})")
                print(f"   BR: ({corners['BR'][0]:.1f}, {corners['BR'][1]:.1f})")
                print(f"   BL: ({corners['BL'][0]:.1f}, {corners['BL'][1]:.1f})")
                print(f"\n📁 Guardado en {self.output_dir}/")
                
                cv2.destroyAllWindows()
                return corners
                
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
                
            elif key == ord('q') or key == ord('Q'):
                # Perspectiva hacia adelante (parte superior más pequeña)
                self.perspective = min(self.perspective + 0.2, 10.0)
            elif key == ord('a') or key == ord('A'):
                # Perspectiva hacia adelante (parte superior más pequeña)
                self.perspective = min(self.perspective + 0.2, 10.0)
            elif key == ord('e') or key == ord('E'):
                # Perspectiva hacia atrás (parte superior más grande)
                self.perspective = max(self.perspective - 0.2, -10.0)
            elif key == ord('d') or key == ord('D'):
                # Perspectiva hacia atrás (parte superior más grande)
                self.perspective = max(self.perspective - 0.2, -10.0)
            elif key == ord('w') or key == ord('W'):
                # Aumentar escala
                self.scale = min(self.scale + 20, max(self.width, self.height))
            elif key == ord('s') or key == ord('S'):
                # Disminuir escala
                self.scale = max(self.scale - 20, 50)
            elif key == ord('r') or key == ord('R'):
                # Reiniciar
                self.center_x = self.width // 2
                self.center_y = self.height // 2
                self.scale = min(self.width, self.height) // 4
                self.perspective = 0.0
                self.drag_offset = (0, 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python spike_court_calibration.py <video_path> [output_dir]")
        print("\nEjemplo:")
        print('  python spike_court_calibration.py "test-videos/VideoPadelAmateur1.mp4" "runs/court_amateur1"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "runs/court_calibrated"
    
    print("🎯 Herramienta de Calibración Visual de Cancha\n")
    
    calibrator = CourtCalibrator(video_path, output_dir)
    corners = calibrator.run()
    
    if corners:
        print("\n" + "="*60)
        print("✅ CALIBRACIÓN GUARDADA")
        print(f"   Usá {output_dir}/court_corners.json para el análisis")
        print("="*60)