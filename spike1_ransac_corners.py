#!/usr/bin/env python3
"""
Spike 1 - Detección de Cancha con RANSAC y Esquinas

Dos estrategias:
1. RANSAC: Detectar líneas con Hough y encontrar el mejor rectángulo
2. Esquinas (Harris/ORB): Detectar esquinas prominentes y encontrar las 4 de la cancha
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_court_ransac(frame):
    """
    Estrategia 1: RANSAC para encontrar el mejor rectángulo.
    
    1. Detectar todas las líneas con Hough
    2. Agrupar líneas horizontales y verticales
    3. Usar RANSAC para encontrar el mejor rectángulo
    """
    h, w = frame.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detectar líneas con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=20)
    
    if lines is None or len(lines) < 4:
        return None
    
    # Separar líneas horizontales y verticales
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if angle < np.pi/6:  # Horizontal
            horizontal.append((x1, y1, x2, y2, length))
        elif angle > np.pi/3:  # Vertical
            vertical.append((x1, y1, x2, y2, length))
    
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    
    # RANSAC: Encontrar las mejores 2 horizontales y 2 verticales
    # Criterio: líneas más largas que formen un rectángulo coherente
    
    # Ordenar por longitud
    horizontal = sorted(horizontal, key=lambda x: x[4], reverse=True)[:10]
    vertical = sorted(vertical, key=lambda x: x[4], reverse=True)[:10]
    
    best_rect = None
    best_score = 0
    
    # Probar combinaciones de 2 horizontales y 2 verticales
    for i, h1 in enumerate(horizontal):
        for j, h2 in enumerate(horizontal):
            if i >= j:
                continue
            for k, v1 in enumerate(vertical):
                for l, v2 in enumerate(vertical):
                    if k >= l:
                        continue
                    
                    # Obtener las 4 intersecciones
                    h1_y = (h1[1] + h1[3]) / 2
                    h2_y = (h2[1] + h2[3]) / 2
                    v1_x = (v1[0] + v1[2]) / 2
                    v2_x = (v2[0] + v2[2]) / 2
                    
                    # Asegurar orden correcto
                    top_y = min(h1_y, h2_y)
                    bottom_y = max(h1_y, h2_y)
                    left_x = min(v1_x, v2_x)
                    right_x = max(v1_x, v2_x)
                    
                    # Verificar que es un rectángulo válido
                    width = right_x - left_x
                    height = bottom_y - top_y
                    
                    if width < w * 0.3 or height < h * 0.3:
                        continue
                    if width > w * 0.95 or height > h * 0.95:
                        continue
                    
                    # Calcular score (área normalizada + longitud de líneas)
                    area = width * height
                    line_lengths = h1[4] + h2[4] + v1[4] + v2[4]
                    score = area / (w * h) + line_lengths / (w + h)
                    
                    if score > best_score:
                        best_score = score
                        best_rect = np.array([
                            [left_x, top_y],
                            [right_x, top_y],
                            [right_x, bottom_y],
                            [left_x, bottom_y]
                        ], dtype=np.float32)
    
    return best_rect


def detect_court_harris(frame):
    """
    Estrategia 2: Harris corners para encontrar esquinas de la cancha.
    
    1. Detectar esquinas con Harris
    2. Filtrar esquinas en la zona esperada
    3. Encontrar las 4 más prominentes
    """
    h, w = frame.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar esquinas con Harris
    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, blockSize=5, ksize=5, k=0.04)
    
    # Umbral para obtener esquinas significativas
    threshold = 0.01 * corners.max()
    corner_mask = corners > threshold
    
    # Obtener coordenadas de esquinas
    y_coords, x_coords = np.where(corner_mask)
    
    if len(x_coords) < 4:
        return None
    
    # Filtrar esquinas muy cercanas entre sí
    points = np.column_stack((x_coords, y_coords))
    
    # Agrupar puntos cercanos
    from scipy.cluster.hierarchy import fclusterdata
    try:
        clusters = fclusterdata(points, t=20, criterion='distance')
        unique_clusters = np.unique(clusters)
        
        # Obtener el centro de cada cluster
        cluster_centers = []
        for c in unique_clusters:
            cluster_points = points[clusters == c]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        
        points = np.array(cluster_centers)
    except:
        pass
    
    if len(points) < 4:
        return None
    
    # Encontrar las 4 esquinas de la cancha
    # Usar k-means con k=4 para encontrar 4 grupos
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    points_float = np.float32(points)
    
    if len(points_float) < 4:
        return None
    
    # Si hay exactamente 4 puntos, usarlos directamente
    if len(points_float) == 4:
        corners_4 = points_float
    else:
        # K-means con k=4
        _, labels, centers = cv2.kmeans(points_float, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        corners_4 = centers
    
    # Ordenar las esquinas: TL, TR, BR, BL
    center = np.mean(corners_4, axis=0)
    
    top_left = None
    top_right = None
    bottom_right = None
    bottom_left = None
    
    for point in corners_4:
        x, y = point
        if x < center[0] and y < center[1]:
            if top_left is None or y < top_left[1]:
                top_left = point
        elif x >= center[0] and y < center[1]:
            if top_right is None or y < top_right[1]:
                top_right = point
        elif x >= center[0] and y >= center[1]:
            if bottom_right is None or y > bottom_right[1]:
                bottom_right = point
        else:
            if bottom_left is None or y > bottom_left[1]:
                bottom_left = point
    
    if any(x is None for x in [top_left, top_right, bottom_right, bottom_left]):
        return None
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def detect_court_orb(frame):
    """
    Estrategia 2b: ORB para detectar puntos clave y encontrar esquinas.
    """
    h, w = frame.shape[:2]
    
    # Crear detector ORB
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detectar keypoints
    keypoints = orb.detect(frame, None)
    
    if len(keypoints) < 4:
        return None
    
    # Obtener coordenadas
    points = np.array([kp.pt for kp in keypoints])
    
    # Filtrar puntos en los bordes del frame (margen 15%)
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    
    mask = ((points[:, 0] > margin_x) & (points[:, 0] < w - margin_x) &
            (points[:, 1] > margin_y) & (points[:, 1] < h - margin_y))
    
    points = points[mask]
    
    if len(points) < 4:
        return None
    
    # K-means para encontrar 4 grupos
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    points_float = np.float32(points)
    
    _, labels, centers = cv2.kmeans(points_float, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Ordenar esquinas
    center = np.mean(centers, axis=0)
    
    corners_ordered = [None, None, None, None]
    
    for point in centers:
        x, y = point
        if x < center[0] and y < center[1]:
            corners_ordered[0] = point
        elif x >= center[0] and y < center[1]:
            corners_ordered[1] = point
        elif x >= center[0] and y >= center[1]:
            corners_ordered[2] = point
        else:
            corners_ordered[3] = point
    
    if any(x is None for x in corners_ordered):
        return None
    
    return np.array(corners_ordered, dtype=np.float32)


def point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono."""
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0


def analyze_with_strategy(video_path: str, strategy: str = 'ransac', max_frames: int = 100, conf_threshold: float = 0.5):
    """
    Analiza el video usando la estrategia especificada.
    """
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {width}x{height}")
    
    os.makedirs(f"runs/{strategy}_analysis", exist_ok=True)
    
    # Detectar cancha en el primer frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer video")
        return None
    
    if strategy == 'ransac':
        corners = detect_court_ransac(first_frame)
    elif strategy == 'harris':
        corners = detect_court_harris(first_frame)
    elif strategy == 'orb':
        corners = detect_court_orb(first_frame)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    
    if corners is None:
        print(f"⚠️ No se pudo detectar la cancha con {strategy}")
        # Fallback a zona fija
        margin_x = int(width * 0.12)
        margin_y = int(height * 0.08)
        corners = np.array([
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [width - margin_x, height - margin_y],
            [margin_x, height - margin_y]
        ], dtype=np.float32)
    
    print(f"\n📐 Esquinas detectadas ({strategy}):")
    print(f"   TL: ({corners[0][0]:.0f}, {corners[0][1]:.0f})")
    print(f"   TR: ({corners[1][0]:.0f}, {corners[1][1]:.0f})")
    print(f"   BR: ({corners[2][0]:.0f}, {corners[2][1]:.0f})")
    print(f"   BL: ({corners[3][0]:.0f}, {corners[3][1]:.0f})")
    
    polygon = corners.reshape((-1, 1, 2)).astype(np.int32)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    final_counts = []
    
    print(f"\n🔍 Analizando {max_frames} frames...")
    
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=[0], verbose=False)
        
        frame_detections = []
        
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    in_court = point_in_polygon((cx, cy), polygon)
                    
                    frame_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'confidence': conf,
                        'in_court': in_court
                    })
        
        # Filtrar
        filtered = [d for d in frame_detections if d['in_court'] and d['confidence'] >= conf_threshold]
        
        if len(filtered) > 4:
            filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
        
        final_counts.append(len(filtered))
        
        # Guardar frames de muestra
        if frame_idx % 25 == 0:
            sample = frame.copy()
            
            # Dibujar polígono
            cv2.polylines(sample, [polygon], True, (255, 0, 0), 3)
            
            # Dibujar detecciones
            for d in frame_detections:
                x1, y1, x2, y2 = d['bbox']
                conf = d['confidence']
                
                if d['in_court'] and conf >= conf_threshold:
                    color = (0, 255, 0)  # Verde
                elif d['in_court']:
                    color = (0, 255, 255)  # Amarillo
                else:
                    color = (0, 0, 255)  # Rojo
                
                cv2.rectangle(sample, (x1, y1), (x2, y2), color, 2)
            
            cv2.putText(sample, f"Filtered: {len(filtered)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite(f"runs/{strategy}_analysis/frame_{frame_idx:04d}.jpg", sample)
            print(f"   Frame {frame_idx}: {len(filtered)} jugadores")
    
    cap.release()
    
    # Estadísticas
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS CON {strategy.upper()}")
    print(f"{'='*60}")
    
    frames_with_4 = sum(1 for c in final_counts if c == 4)
    frames_with_3_5 = sum(1 for c in final_counts if 3 <= c <= 5)
    
    print(f"\n🎯 Resultados:")
    print(f"   Frames con 4 jugadores: {frames_with_4}/{max_frames} ({100*frames_with_4/max_frames:.1f}%)")
    print(f"   Frames con 3-5 jugadores: {frames_with_3_5}/{max_frames} ({100*frames_with_3_5/max_frames:.1f}%)")
    print(f"   Promedio: {np.mean(final_counts):.2f} jugadores/frame")
    
    return {
        'strategy': strategy,
        'frames_with_4': frames_with_4,
        'frames_with_3_5': frames_with_3_5,
        'avg': round(np.mean(final_counts), 2),
        'corners': corners.tolist()
    }


if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    print("🏃 Spike 1 - RANSAC y Esquinas\n")
    
    print("="*60)
    print("ESTRATEGIA 1: RANSAC")
    print("="*60)
    ransac_results = analyze_with_strategy(video_path, strategy='ransac', max_frames=100)
    
    print("\n" + "="*60)
    print("ESTRATEGIA 2: HARRIS CORNERS")
    print("="*60)
    harris_results = analyze_with_strategy(video_path, strategy='harris', max_frames=100)
    
    print("\n" + "="*60)
    print("ESTRATEGIA 2b: ORB")
    print("="*60)
    orb_results = analyze_with_strategy(video_path, strategy='orb', max_frames=100)
    
    # Comparar resultados
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE ESTRATEGIAS")
    print("="*60)
    print(f"\n{'Estrategia':<15} {'Frames 4':<12} {'Frames 3-5':<12} {'Promedio'}")
    print("-"*60)
    print(f"{'RANSAC':<15} {ransac_results['frames_with_4']}%{'':<8} {ransac_results['frames_with_3_5']}%{'':<8} {ransac_results['avg']}")
    print(f"{'Harris':<15} {harris_results['frames_with_4']}%{'':<8} {harris_results['frames_with_3_5']}%{'':<8} {harris_results['avg']}")
    print(f"{'ORB':<15} {orb_results['frames_with_4']}%{'':<8} {orb_results['frames_with_3_5']}%{'':<8} {orb_results['avg']}")