#!/usr/bin/env python3
"""
Script para seleccionar las esquinas de la cancha de un video.

Uso:
    python spike1_select_court.py <video_path> [output_dir]
    
Ejemplo:
    python spike1_select_court.py "test-videos/FinalPadelPrueba1.mp4" "runs/court_finalpadel"
"""

import cv2
import numpy as np
import os
import json
import sys

# Variables globales
points = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    """Callback para capturar clics del mouse."""
    global points, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"   Punto {len(points)}: ({x}, {y})")
            
            # Dibujar punto
            cv2.circle(current_frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(current_frame, str(len(points)), (x, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Dibujar líneas entre puntos
            if len(points) > 1:
                cv2.line(current_frame, points[-2], points[-1], (255, 0, 0), 2)
            
            # Cerrar el polígono si tenemos 4 puntos
            if len(points) == 4:
                cv2.line(current_frame, points[-1], points[0], (255, 0, 0), 2)
                
                # Rellenar el polígono
                overlay = current_frame.copy()
                pts = np.array(points, np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, current_frame, 0.7, 0, current_frame)
            
            cv2.imshow("Seleccionar Cancha", current_frame)


def select_court_manually(video_path: str, output_dir: str = "runs/manual_court"):
    """
    Permite al usuario seleccionar los 4 vértices de la cancha manualmente.
    """
    global points, current_frame
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error al leer video")
        return None
    
    current_frame = frame.copy()
    
    print("\n" + "="*60)
    print("🎯 SELECCIÓN MANUAL DE CANCHA")
    print("="*60)
    print("\nInstrucciones:")
    print("   1. Haz clic en los 4 vértices de la cancha")
    print("   2. Orden: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    print("   3. Presiona 'r' para reiniciar")
    print("   4. Presiona 'q' para confirmar y guardar")
    print("="*60 + "\n")
    
    cv2.namedWindow("Seleccionar Cancha")
    cv2.setMouseCallback("Seleccionar Cancha", mouse_callback)
    cv2.imshow("Seleccionar Cancha", current_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') and len(points) == 4:
            break
        elif key == ord('r'):
            # Reiniciar
            points = []
            current_frame = frame.copy()
            cv2.imshow("Seleccionar Cancha", current_frame)
            print("\n   Reiniciado. Selecciona los 4 puntos nuevamente.")
    
    cv2.destroyAllWindows()
    
    if len(points) == 4:
        # Guardar resultado
        os.makedirs(output_dir, exist_ok=True)
        
        result = frame.copy()
        
        # Dibujar polígono final
        pts = np.array(points, np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 0), 3)
        
        for i, pt in enumerate(points):
            cv2.circle(result, pt, 10, (0, 0, 255), -1)
            label = ['TL', 'TR', 'BR', 'BL'][i]
            cv2.putText(result, label, (pt[0], pt[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Rellenar
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        cv2.imwrite(f"{output_dir}/court_selection.jpg", result)
        
        # Guardar coordenadas
        corners = {
            'TL': list(points[0]),
            'TR': list(points[1]),
            'BR': list(points[2]),
            'BL': list(points[3])
        }
        
        with open(f"{output_dir}/court_corners.json", "w") as f:
            json.dump(corners, f, indent=2)
        
        print(f"\n✅ Cancha seleccionada:")
        print(f"   TL: {points[0]}")
        print(f"   TR: {points[1]}")
        print(f"   BR: {points[2]}")
        print(f"   BL: {points[3]}")
        print(f"\n📁 Guardado en {output_dir}/")
        
        return corners
    
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python spike1_select_court.py <video_path> [output_dir]")
        print("\nEjemplo:")
        print('  python spike1_select_court.py "test-videos/FinalPadelPrueba1.mp4" "runs/court_finalpadel"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "runs/manual_court"
    
    print("🏃 Selección Manual de Cancha\n")
    
    corners = select_court_manually(video_path, output_dir)
    
    if corners:
        print("\n" + "="*60)
        print("✅ SELECCIÓN GUARDADA")
        print(f"   Usá {output_dir}/court_corners.json para el análisis")
        print("="*60)