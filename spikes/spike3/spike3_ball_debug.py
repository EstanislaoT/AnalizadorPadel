#!/usr/bin/env python3
"""
Spike 3 - Debug de detección de pelota

Extrae un frame del video y prueba diferentes configuraciones
para detectar correctamente la pelota de pádel.
"""

import cv2
import numpy as np
import json
import sys


def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Extrae un frame específico del video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def detect_ball_by_color(frame: np.ndarray, h_range: tuple, s_range: tuple, v_range: tuple) -> list:
    """
    Detecta la pelota por color HSV con rangos personalizables.
    
    Returns:
        Lista de (x, y, radius, mask) de posibles pelotas
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array([h_range[0], s_range[0], v_range[0]])
    upper = np.array([h_range[1], s_range[1], v_range[1]])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:  # Muy pequeño
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Verificar circularidad
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            candidates.append({
                'x': x, 'y': y, 'radius': radius,
                'area': area, 'circularity': circularity
            })
    
    return candidates, mask


def detect_circles(frame: np.ndarray, min_radius: int, max_radius: int, 
                   param1: int, param2: int) -> list:
    """
    Detecta círculos usando HoughCircles con parámetros personalizables.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        return [(x, y, r) for x, y, r in np.round(circles[0, :]).astype(int)]
    return []


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    pt = (int(point[0]), int(point[1]))
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def main():
    if len(sys.argv) < 3:
        print("Uso: python spike3_ball_debug.py <video_path> <corners_path> [frame_idx]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    corners_path = sys.argv[2]
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Cargar esquinas de la cancha
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    print(f"📹 Extrayendo frame {frame_idx} de {video_path}")
    frame = extract_frame(video_path, frame_idx)
    
    if frame is None:
        print("❌ No se pudo extraer el frame")
        return
    
    # Dibujar cancha
    cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
    
    print(f"\n📐 Dimensiones del frame: {frame.shape[1]}x{frame.shape[0]}")
    print(f"\n🔍 Probando diferentes configuraciones de color HSV...")
    
    # Probar diferentes rangos de color
    color_configs = [
        # (nombre, H range, S range, V range)
        ('Amarillo estándar', (20, 40), (100, 255), (100, 255)),
        ('Amarillo brillante', (20, 35), (80, 255), (150, 255)),
        ('Amarillo oscuro', (20, 45), (50, 200), (80, 200)),
        ('Verde fluorescente', (35, 85), (100, 255), (100, 255)),
        ('Naranja claro', (10, 25), (100, 255), (150, 255)),
        ('Amarillo neón', (25, 35), (80, 255), (180, 255)),
        ('Amarillo amplio', (15, 50), (50, 255), (80, 255)),
    ]
    
    # Crear imagen de debug combinada
    debug_h = 300
    debug_w = 400
    
    all_results = []
    
    for name, h_range, s_range, v_range in color_configs:
        candidates, mask = detect_ball_by_color(frame, h_range, s_range, v_range)
        
        # Filtrar por radio típico de pelota (3-15 px)
        filtered = [c for c in candidates if 3 <= c['radius'] <= 15]
        
        # Filtrar por circularidad
        circular = [c for c in filtered if c['circularity'] > 0.4]
        
        # Filtrar por posición dentro de cancha
        in_court = [c for c in circular if point_in_polygon((c['x'], c['y']), polygon)]
        
        all_results.append({
            'name': name,
            'config': (h_range, s_range, v_range),
            'total': len(candidates),
            'filtered': len(filtered),
            'circular': len(circular),
            'in_court': len(in_court),
            'best': in_court[0] if in_court else (circular[0] if circular else None),
            'mask': mask
        })
        
        print(f"\n  {name}:")
        print(f"    Total detecciones: {len(candidates)}")
        print(f"    Con radio válido (3-15px): {len(filtered)}")
        print(f"    Circulares (circularidad > 0.4): {len(circular)}")
        print(f"    Dentro de cancha: {len(in_court)}")
        
        if in_court:
            best = in_court[0]
            print(f"    ✅ Mejor candidato: ({best['x']:.1f}, {best['y']:.1f}), r={best['radius']:.1f}, "
                  f"area={best['area']:.1f}, circ={best['circularity']:.2f}")
    
    # Detectar círculos con Hough
    print(f"\n🔍 Detectando círculos con HoughCircles...")
    
    hough_configs = [
        # (param1, param2, min_r, max_r)
        (50, 20, 3, 15),
        (50, 30, 3, 15),
        (70, 25, 3, 12),
        (50, 15, 2, 20),
    ]
    
    for p1, p2, min_r, max_r in hough_configs:
        circles = detect_circles(frame, min_r, max_r, p1, p2)
        in_court_circles = [c for c in circles if point_in_polygon((c[0], c[1]), polygon)]
        print(f"  param1={p1}, param2={p2}, r=[{min_r},{max_r}]: {len(circles)} círculos, {len(in_court_circles)} en cancha")
        
        for x, y, r in in_court_circles[:3]:  # Mostrar primeros 3
            print(f"    Círculo: ({x}, {y}), r={r}")
    
    # Crear visualización con el mejor resultado
    best_result = max(all_results, key=lambda r: r['in_court'] if r['in_court'] > 0 else -1)
    
    if best_result['best']:
        # Dibujar mejor detección
        best = best_result['best']
        cv2.circle(frame, (int(best['x']), int(best['y'])), int(best['radius']), (0, 255, 0), 2)
        cv2.circle(frame, (int(best['x']), int(best['y'])), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"Best: {best_result['name']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"({best['x']:.0f}, {best['y']:.0f}) r={best['radius']:.0f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Crear panel de máscaras
    mask_panel = np.zeros((debug_h, len(color_configs) * debug_w), dtype=np.uint8)
    for i, result in enumerate(all_results):
        mask_resized = cv2.resize(result['mask'], (debug_w, debug_h))
        mask_panel[:, i*debug_w:(i+1)*debug_w] = mask_resized
        
        # Agregar texto
        cv2.putText(mask_panel, result['name'], (i*debug_w + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        cv2.putText(mask_panel, f"In court: {result['in_court']}", (i*debug_w + 5, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    
    # Mostrar resultados
    cv2.imshow('Frame with detection', frame)
    cv2.imshow('Color masks', mask_panel)
    
    print(f"\n⌨️ Presiona 'q' para salir, 'n' para siguiente frame, 'p' para frame anterior")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            frame_idx += 10
            print(f"\n➡️ Frame {frame_idx}")
            # Repetir análisis
            frame = extract_frame(video_path, frame_idx)
            if frame is not None:
                cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
                # ... repetir análisis
                cv2.imshow('Frame with detection', frame)
        elif key == ord('p'):
            frame_idx = max(0, frame_idx - 10)
            print(f"\n⬅️ Frame {frame_idx}")
            frame = extract_frame(video_path, frame_idx)
            if frame is not None:
                cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
                cv2.imshow('Frame with detection', frame)
    
    cv2.destroyAllWindows()
    
    # Guardar frame con detección
    cv2.imwrite('runs/spike3_ball/debug_frame.jpg', frame)
    print(f"\n💾 Frame guardado en: runs/spike3_ball/debug_frame.jpg")


if __name__ == "__main__":
    main()