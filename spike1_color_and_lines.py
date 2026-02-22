#!/usr/bin/env python3
"""
Spike 1 - Detección de Cancha: Color + Líneas

Problema anterior: Detección por color incluía vidrios y público.

Nueva estrategia:
1. Detectar color del piso (área amplia)
2. Detectar líneas BLANCAS de la cancha (bordes reales)
3. Intersectar: solo el área que está dentro de las líneas
"""

import cv2
import numpy as np
import os

def detect_court_color(frame):
    """Detecta el color del piso de la cancha."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Dividir en grilla 5x5
    grid_h, grid_w = h // 5, w // 5
    
    cell_colors = []
    for i in range(5):
        for j in range(5):
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w
            
            cell = hsv[y1:y2, x1:x2]
            
            mean_h = np.mean(cell[:, :, 0])
            mean_s = np.mean(cell[:, :, 1])
            mean_v = np.mean(cell[:, :, 2])
            
            std_h = np.std(cell[:, :, 0])
            std_s = np.std(cell[:, :, 1])
            std_v = np.std(cell[:, :, 2])
            
            cell_colors.append({
                'pos': (i, j),
                'mean': (mean_h, mean_s, mean_v),
                'uniformity': std_h + std_s + std_v
            })
    
    # Celdas del centro
    center_cells = [c for c in cell_colors if 1 <= c['pos'][0] <= 3 and 1 <= c['pos'][1] <= 3]
    center_cells.sort(key=lambda x: x['uniformity'])
    best_cells = center_cells[:4]
    
    avg_h = np.mean([c['mean'][0] for c in best_cells])
    avg_s = np.mean([c['mean'][1] for c in best_cells])
    avg_v = np.mean([c['mean'][2] for c in best_cells])
    
    return {
        'h': (avg_h - 30, avg_h + 30),
        's': (avg_s - 50, avg_s + 50),
        'v': (avg_v - 60, avg_v + 60),
        'mean': (avg_h, avg_s, avg_v)
    }


def create_color_mask(frame, color_range):
    """Crea máscara basada en color del piso."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower = np.array([max(0, color_range['h'][0]), 
                      max(0, color_range['s'][0]), 
                      max(0, color_range['v'][0])])
    upper = np.array([min(180, color_range['h'][1]), 
                      min(255, color_range['s'][1]), 
                      min(255, color_range['v'][1])])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def detect_court_lines(frame):
    """
    Detecta las líneas BLANCAS de la cancha.
    
    Las canchas de pádel tienen líneas blancas que delimitan el área de juego.
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detectar blanco (alto valor, baja saturación)
    # Líneas blancas: alto V, bajo S
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Detectar bordes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Solo en áreas donde hay blanco potencial
    masked_gray = cv2.bitwise_and(gray, gray, mask=white_mask)
    
    # Canny
    edges = cv2.Canny(masked_gray, 50, 150)
    
    # Hough para líneas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=20)
    
    return lines, white_mask


def find_court_polygon_from_lines(lines, frame_shape):
    """
    Encuentra el polígono de la cancha a partir de las líneas detectadas.
    
    Busca las 4 líneas principales que forman el rectángulo de juego.
    """
    h, w = frame_shape[:2]
    
    if lines is None or len(lines) < 4:
        return None
    
    # Separar líneas por orientación
    horizontal = []
    vertical = []
    diagonal = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if angle < np.pi/8:  # ~0 grados - horizontal
            horizontal.append((x1, y1, x2, y2, length))
        elif angle > np.pi/2 - np.pi/8:  # ~90 grados - vertical
            vertical.append((x1, y1, x2, y2, length))
        else:
            diagonal.append((x1, y1, x2, y2, length, angle))
    
    # Si hay muchas líneas diagonales, pueden ser los bordes de la cancha en perspectiva
    # Tomar las líneas más largas
    
    all_lines = []
    for l in horizontal:
        all_lines.append(('H', l[0], l[1], l[2], l[3], l[4]))
    for l in vertical:
        all_lines.append(('V', l[0], l[1], l[2], l[3], l[4]))
    for l in diagonal:
        all_lines.append(('D', l[0], l[1], l[2], l[3], l[4], l[5]))
    
    # Ordenar por longitud
    all_lines.sort(key=lambda x: x[5], reverse=True)
    
    # Tomar las 8 líneas más largas
    top_lines = all_lines[:8]
    
    # Encontrar las 4 esquinas
    # Buscar intersecciones entre líneas
    
    points = []
    for i, line1 in enumerate(top_lines):
        for j, line2 in enumerate(top_lines):
            if i >= j:
                continue
            
            # Calcular intersección
            x1, y1, x2, y2 = line1[1], line1[2], line1[3], line1[4]
            x3, y3, x4, y4 = line2[1], line2[2], line2[3], line2[4]
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1:
                continue
            
            ix = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            iy = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
            
            # Verificar que está dentro del frame
            if 0 <= ix <= w and 0 <= iy <= h:
                points.append((ix, iy))
    
    if len(points) < 4:
        return None
    
    # Clusterizar puntos para encontrar las 4 esquinas
    points = np.array(points)
    
    # K-means con k=4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(np.float32(points), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Ordenar esquinas
    corners = order_corners(centers)
    
    return corners


def order_corners(corners):
    """Ordena esquinas: TL, TR, BR, BL."""
    corners = np.array(corners, dtype=np.float32)
    center = np.mean(corners, axis=0)
    
    ordered = [None, None, None, None]
    
    for p in corners:
        if p[0] < center[0] and p[1] < center[1]:
            ordered[0] = p
        elif p[0] >= center[0] and p[1] < center[1]:
            ordered[1] = p
        elif p[0] >= center[0] and p[1] >= center[1]:
            ordered[2] = p
        else:
            ordered[3] = p
    
    if any(x is None for x in ordered):
        corners = corners[corners[:, 1].argsort()]
        top = corners[:2]
        bottom = corners[2:]
        top = top[top[:, 0].argsort()]
        bottom = bottom[bottom[:, 0].argsort()]
        ordered = [top[0], top[1], bottom[1], bottom[0]]
    
    return np.array(ordered, dtype=np.float32)


def detect_court_combined(video_path: str):
    """
    Detecta la cancha combinando color + líneas.
    """
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    ret, frame = cap.read()
    if not ret:
        return None
    
    print("\n🔍 Paso 1: Detectando color del piso...")
    color_range = detect_court_color(frame)
    print(f"   Color HSV medio: H={color_range['mean'][0]:.1f}, S={color_range['mean'][1]:.1f}, V={color_range['mean'][2]:.1f}")
    
    print("\n🔍 Paso 2: Creando máscara de color...")
    color_mask = create_color_mask(frame, color_range)
    color_coverage = 100 * np.count_nonzero(color_mask) / (width * height)
    print(f"   Cobertura de color: {color_coverage:.1f}%")
    
    print("\n🔍 Paso 3: Detectando líneas de la cancha...")
    lines, white_mask = detect_court_lines(frame)
    if lines is not None:
        print(f"   Líneas detectadas: {len(lines)}")
    else:
        print("   No se detectaron líneas")
    
    print("\n🔍 Paso 4: Encontrando polígono de líneas...")
    line_corners = find_court_polygon_from_lines(lines, frame.shape)
    
    os.makedirs("runs/color_lines", exist_ok=True)
    
    result = frame.copy()
    
    if line_corners is not None:
        print(f"\n📐 Esquinas detectadas por líneas:")
        print(f"   TL: ({line_corners[0][0]:.0f}, {line_corners[0][1]:.0f})")
        print(f"   TR: ({line_corners[1][0]:.0f}, {line_corners[1][1]:.0f})")
        print(f"   BR: ({line_corners[2][0]:.0f}, {line_corners[2][1]:.0f})")
        print(f"   BL: ({line_corners[3][0]:.0f}, {line_corners[3][1]:.0f})")
        
        # Crear máscara del polígono de líneas
        polygon = line_corners.reshape((-1, 1, 2)).astype(np.int32)
        line_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(line_mask, [polygon], 255)
        
        # Intersectar máscara de color con polígono de líneas
        final_mask = cv2.bitwise_and(color_mask, line_mask)
        
        # Dibujar
        cv2.polylines(result, [polygon], True, (255, 0, 0), 3)
        for i, c in enumerate(line_corners):
            cv2.circle(result, tuple(c.astype(int)), 10, (0, 0, 255), -1)
            cv2.putText(result, str(i+1), tuple(c.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Overlay del área final
        overlay = result.copy()
        overlay[final_mask > 0] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        final_coverage = 100 * np.count_nonzero(final_mask) / (width * height)
        print(f"\n📊 Cobertura final: {final_coverage:.1f}%")
    else:
        print("\n⚠️ No se pudieron detectar las líneas de la cancha")
        # Usar solo máscara de color
        overlay = result.copy()
        overlay[color_mask > 0] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    
    cv2.imwrite("runs/color_lines/court_detection.jpg", result)
    cv2.imwrite("runs/color_lines/color_mask.jpg", color_mask)
    cv2.imwrite("runs/color_lines/white_mask.jpg", white_mask)
    
    print(f"\n📁 Resultados guardados en runs/color_lines/")
    
    cap.release()
    
    return {
        'color_range': color_range,
        'lines_count': len(lines) if lines is not None else 0,
        'corners': line_corners.tolist() if line_corners is not None else None
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - Detección de Cancha: Color + Líneas\n")
    print("="*60)
    
    result = detect_court_combined(video_path)
    
    print("\n" + "="*60)
    if result and result['corners']:
        print("✅ DETECCIÓN COMPLETADA")
    else:
        print("⚠️ DETECCIÓN PARCIAL - solo color, sin líneas")
    print("="*60)