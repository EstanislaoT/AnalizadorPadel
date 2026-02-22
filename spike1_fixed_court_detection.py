#!/usr/bin/env python3
"""
Spike 1 - Detección de Cancha con Cámara Fija

Ventajas:
1. Cámara FIJA - la cancha no se mueve entre frames
2. Color de piso CONSTANTE - mismo color en todo el video  
3. Forma de TRAPEZOIDE - proporción conocida de pádel

Estrategia:
1. Analizar el primer frame para detectar el color del piso
2. Crear máscara binaria con ese color
3. Encontrar el contorno más grande (la cancha)
4. Ajustar a un trapezoide
5. Usar la misma máscara para todos los frames
"""

import cv2
import numpy as np
import os

def analyze_court_color(frame):
    """
    Analiza el frame para detectar el color dominante del piso de la cancha.
    
    Estrategia:
    - Dividir el frame en una grilla
    - Encontrar regiones con color similar (potencialmente el piso)
    - El grupo más grande es probablemente la cancha
    """
    h, w = frame.shape[:2]
    
    # Convertir a HSV para mejor detección de color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Dividir en una grilla 5x5
    grid_h, grid_w = h // 5, w // 5
    
    # Recopilar colores de cada celda
    cell_colors = []
    for i in range(5):
        for j in range(5):
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w
            
            cell = hsv[y1:y2, x1:x2]
            
            # Color medio de la celda
            mean_h = np.mean(cell[:, :, 0])
            mean_s = np.mean(cell[:, :, 1])
            mean_v = np.mean(cell[:, :, 2])
            
            # Desviación estándar (baja = color uniforme = potencial piso)
            std_h = np.std(cell[:, :, 0])
            std_s = np.std(cell[:, :, 1])
            std_v = np.std(cell[:, :, 2])
            
            cell_colors.append({
                'pos': (i, j),
                'mean': (mean_h, mean_s, mean_v),
                'std': (std_h, std_s, std_v),
                'uniformity': std_h + std_s + std_v  # Más bajo = más uniforme
            })
    
    # Encontrar celdas con color uniforme (potencial piso)
    # Ordenar por uniformidad (menor = mejor)
    cell_colors.sort(key=lambda x: x['uniformity'])
    
    # Tomar las celdas del centro (probablemente cancha)
    center_cells = [c for c in cell_colors if 1 <= c['pos'][0] <= 3 and 1 <= c['pos'][1] <= 3]
    
    if center_cells:
        # Usar el color promedio de las celdas centrales más uniformes
        center_cells.sort(key=lambda x: x['uniformity'])
        best_cells = center_cells[:4]  # 4 mejores celdas centrales
        
        avg_h = np.mean([c['mean'][0] for c in best_cells])
        avg_s = np.mean([c['mean'][1] for c in best_cells])
        avg_v = np.mean([c['mean'][2] for c in best_cells])
        
        # Rangos de tolerancia
        h_range = 30
        s_range = 50
        v_range = 60
        
        return {
            'h': (avg_h - h_range, avg_h + h_range),
            's': (avg_s - s_range, avg_s + s_range),
            'v': (avg_v - v_range, avg_v + v_range),
            'mean': (avg_h, avg_s, avg_v)
        }
    
    return None


def create_court_mask(frame, color_range):
    """
    Crea una máscara binaria del piso de la cancha usando el color detectado.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear máscara con rangos de color
    lower = np.array([max(0, color_range['h'][0]), 
                      max(0, color_range['s'][0]), 
                      max(0, color_range['v'][0])])
    upper = np.array([min(180, color_range['h'][1]), 
                      min(255, color_range['s'][1]), 
                      min(255, color_range['v'][1])])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operaciones morfológicas para limpiar
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def find_court_trapezoid(mask):
    """
    Encuentra el trapezoide de la cancha en la máscara.
    
    Busca el contorno más grande y aproxima a un polígono de 4 lados.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Tomar el contorno más grande
    largest = max(contours, key=cv2.contourArea)
    
    # Aproximar a polígono
    perimeter = cv2.arcLength(largest, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    # Si tenemos 4 vértices, usarlos directamente
    if len(approx) == 4:
        corners = approx.reshape(4, 2)
    else:
        # Si no, encontrar el bounding rect rotado o convex hull
        # Usar convex hull
        hull = cv2.convexHull(largest)
        
        # Aproximar el hull a 4 vértices
        epsilon2 = 0.03 * cv2.arcLength(hull, True)
        approx2 = cv2.approxPolyDP(hull, epsilon2, True)
        
        if len(approx2) == 4:
            corners = approx2.reshape(4, 2)
        else:
            # Fallback: usar las 4 esquinas extremas del contorno
            # Encontrar puntos extremos
            leftmost = tuple(largest[largest[:, :, 0].argmin()][0])
            rightmost = tuple(largest[largest[:, :, 0].argmax()][0])
            topmost = tuple(largest[largest[:, :, 1].argmin()][0])
            bottommost = tuple(largest[largest[:, :, 1].argmax()][0])
            
            # Combinar y ordenar
            points = np.array([leftmost, rightmost, topmost, bottommost])
            
            # Ordenar como trapezoide: TL, TR, BR, BL
            center = np.mean(points, axis=0)
            
            corners = []
            for p in points:
                if p[0] < center[0] and p[1] < center[1]:
                    corners.append(p)  # TL
                elif p[0] >= center[0] and p[1] < center[1]:
                    corners.append(p)  # TR
                elif p[0] >= center[0] and p[1] >= center[1]:
                    corners.append(p)  # BR
                else:
                    corners.append(p)  # BL
            
            # Si no encontramos 4 esquinas, usar minAreaRect
            if len(corners) != 4:
                rect = cv2.minAreaRect(largest)
                corners = cv2.boxPoints(rect)
    
    # Ordenar las esquinas: TL, TR, BR, BL
    corners = order_corners(corners)
    
    return corners


def order_corners(corners):
    """
    Ordena las esquinas: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    corners = np.array(corners, dtype=np.float32)
    
    # Centro
    center = np.mean(corners, axis=0)
    
    ordered = [None, None, None, None]
    
    for p in corners:
        if p[0] < center[0] and p[1] < center[1]:
            ordered[0] = p  # TL
        elif p[0] >= center[0] and p[1] < center[1]:
            ordered[1] = p  # TR
        elif p[0] >= center[0] and p[1] >= center[1]:
            ordered[2] = p  # BR
        else:
            ordered[3] = p  # BL
    
    # Si alguno quedó None, usar fallback
    if any(x is None for x in ordered):
        # Ordenar por Y primero
        corners = corners[corners[:, 1].argsort()]
        top = corners[:2]
        bottom = corners[2:]
        
        # Ordenar por X
        top = top[top[:, 0].argsort()]
        bottom = bottom[bottom[:, 0].argsort()]
        
        ordered = [top[0], top[1], bottom[1], bottom[0]]
    
    return np.array(ordered, dtype=np.float32)


def detect_fixed_court(video_path: str):
    """
    Detecta la cancha en el primer frame y usa esa detección para todo el video.
    """
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Leer primer frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer video")
        return None
    
    print("\n🔍 Analizando primer frame...")
    
    # Detectar color del piso
    color_range = analyze_court_color(first_frame)
    
    if color_range:
        print(f"\n🎨 Color del piso detectado:")
        print(f"   H: {color_range['h']}")
        print(f"   S: {color_range['s']}")
        print(f"   V: {color_range['v']}")
        print(f"   Mean: H={color_range['mean'][0]:.1f}, S={color_range['mean'][1]:.1f}, V={color_range['mean'][2]:.1f}")
    else:
        print("⚠️ No se pudo detectar color del piso")
        return None
    
    # Crear máscara
    mask = create_court_mask(first_frame, color_range)
    
    # Encontrar trapezoide
    corners = find_court_trapezoid(mask)
    
    if corners is not None:
        print(f"\n📐 Esquinas de la cancha detectadas:")
        print(f"   TL: ({corners[0][0]:.0f}, {corners[0][1]:.0f})")
        print(f"   TR: ({corners[1][0]:.0f}, {corners[1][1]:.0f})")
        print(f"   BR: ({corners[2][0]:.0f}, {corners[2][1]:.0f})")
        print(f"   BL: ({corners[3][0]:.0f}, {corners[3][1]:.0f})")
    else:
        print("⚠️ No se pudo detectar el trapezoide")
        return None
    
    # Guardar resultados visuales
    os.makedirs("runs/fixed_court", exist_ok=True)
    
    # Frame con máscara
    result_frame = first_frame.copy()
    
    # Dibujar máscara semi-transparente
    overlay = result_frame.copy()
    overlay[mask > 0] = [0, 255, 0]
    cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
    
    # Dibujar trapezoide
    polygon = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(result_frame, [polygon], True, (255, 0, 0), 3)
    
    # Dibujar esquinas numeradas
    for i, corner in enumerate(corners):
        cv2.circle(result_frame, tuple(corner.astype(int)), 10, (0, 0, 255), -1)
        cv2.putText(result_frame, str(i+1), tuple(corner.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite("runs/fixed_court/court_detection.jpg", result_frame)
    cv2.imwrite("runs/fixed_court/court_mask.jpg", mask)
    
    print(f"\n📁 Resultados guardados en runs/fixed_court/")
    
    cap.release()
    
    return {
        'color_range': color_range,
        'corners': corners.tolist(),
        'mask': mask
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Detección de Cancha con Cámara Fija\n")
    print("="*60)
    
    result = detect_fixed_court(video_path)
    
    if result:
        print("\n" + "="*60)
        print("✅ DETECCIÓN DE CANCHA EXITOSA")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ No se pudo detectar la cancha")
        print("="*60)