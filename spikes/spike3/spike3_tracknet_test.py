#!/usr/bin/env python3
"""
Spike 3 - Prueba de TrackNet para detección de pelota

Intenta usar TrackNet (modelo pre-entrenado de tenis) para
detectar la pelota de pádel.

Si TrackNet no está disponible, usa un enfoque alternativo con
YOLOv8 entrenado para detectar objetos pequeños.
"""

import cv2
import numpy as np
import sys
import os

# Crear directorio de salida
os.makedirs('runs/spike3_ball', exist_ok=True)


def check_torch_available():
    """Verifica si PyTorch está disponible."""
    try:
        import torch
        print(f"✅ PyTorch disponible: {torch.__version__}")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            return True, 'cuda'
        else:
            print("⚠️ CUDA no disponible, usando CPU (más lento)")
            return True, 'cpu'
    except ImportError:
        print("❌ PyTorch no está instalado")
        return False, None


def check_tracknet_available():
    """Verifica si TrackNet está disponible."""
    # Buscar en varios lugares
    possible_paths = [
        'TrackNetv2',
        'tracknet',
        'models/tracknet',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ TrackNet encontrado en: {path}")
            return path
    
    print("❌ TrackNet no encontrado")
    print("   Para instalar TrackNet:")
    print("   git clone https://github.com/Chang-Chia-Chi/TrackNetv2")
    print("   cd TrackNetv2")
    print("   pip install -r requirements.txt")
    return None


def extract_frames(video_path: str, start_frame: int, num_frames: int = 3) -> list:
    """Extrae frames consecutivos del video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def detect_ball_yolo(frame: np.ndarray) -> list:
    """
    Detecta la pelota usando YOLOv8.
    
    YOLO puede detectar 'sports ball' (clase 32 en COCO).
    """
    try:
        from ultralytics import YOLO
        
        # Cargar modelo
        model = YOLO('yolov8m.pt')
        
        # Detectar
        results = model(frame, classes=[32], conf=0.3, verbose=False)  # 32 = sports ball
        
        balls = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                # Centro y radio aproximado
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                radius = (x2 - x1 + y2 - y1) / 4
                
                balls.append({
                    'x': cx, 'y': cy, 'radius': radius,
                    'confidence': conf
                })
        
        return balls
    except Exception as e:
        print(f"Error en detección YOLO: {e}")
        return []


def detect_ball_template_matching(frame: np.ndarray, template_size: int = 15) -> list:
    """
    Detecta la pelota usando template matching con patrones circulares.
    
    Crea templates de círculos de diferentes tamaños y busca
    coincidencias en la imagen.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Crear templates circulares
    candidates = []
    
    for radius in range(3, 15):
        # Crear template circular
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.circle(template, (template_size//2, template_size//2), radius, 255, -1)
        
        # Aplicar template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Encontrar máximos locales
        threshold = 0.7
        loc = np.where(result >= threshold)
        
        for pt in zip(*loc[::-1]):
            candidates.append({
                'x': pt[0] + template_size // 2,
                'y': pt[1] + template_size // 2,
                'radius': radius,
                'score': result[pt[1], pt[0]]
            })
    
    # Filtrar candidatos duplicados (non-maximum suppression)
    if candidates:
        # Ordenar por score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # NMS simple
        keep = []
        for c in candidates:
            is_duplicate = False
            for k in keep:
                dist = np.sqrt((c['x'] - k['x'])**2 + (c['y'] - k['y'])**2)
                if dist < 10:  # Umbral de distancia
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(c)
        
        return keep[:10]  # Top 10
    
    return []


def detect_ball_background_subtraction(frames: list) -> list:
    """
    Detecta la pelota usando sustracción de fondo.
    
    La pelota es un objeto en movimiento pequeño.
    """
    if len(frames) < 2:
        return []
    
    # Crear sustractor de fondo
    fgbg = cv2.createBackgroundSubtractorMOG2(history=3, varThreshold=50)
    
    # Aplicar a los frames
    masks = []
    for frame in frames:
        fgmask = fgbg.apply(frame)
        masks.append(fgmask)
    
    # La última máscara tiene los objetos en movimiento
    last_mask = masks[-1]
    
    # Encontrar contornos
    contours, _ = cv2.findContours(last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 500:  # Tamaño típico de pelota
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 3 <= radius <= 15:
                candidates.append({
                    'x': x, 'y': y, 'radius': radius,
                    'area': area
                })
    
    return candidates


def main():
    if len(sys.argv) < 3:
        print("Uso: python spike3_tracknet_test.py <video_path> <corners_path> [frame_idx]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    corners_path = sys.argv[2]
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    print("🎾 Spike 3 - Prueba de TrackNet para Pelota")
    print("=" * 60)
    
    # Verificar disponibilidad
    torch_available, device = check_torch_available()
    tracknet_path = check_tracknet_available()
    
    print(f"\n📊 Estado:")
    print(f"   PyTorch: {'✅' if torch_available else '❌'}")
    print(f"   TrackNet: {'✅' if tracknet_path else '❌'}")
    print(f"   Device: {device if torch_available else 'N/A'}")
    
    # Extraer frames
    print(f"\n📹 Extrayendo frames desde {frame_idx}...")
    frames = extract_frames(video_path, frame_idx, 3)
    
    if not frames:
        print("❌ No se pudieron extraer frames")
        return
    
    print(f"   Extraídos: {len(frames)} frames")
    
    # Frame a analizar (el del medio)
    main_frame = frames[len(frames) // 2]
    
    # Intentar diferentes métodos de detección
    all_detections = []
    
    # Método 1: YOLO (sports ball)
    print("\n🔍 Método 1: YOLOv8 (sports ball)...")
    yolo_detections = detect_ball_yolo(main_frame)
    print(f"   Detecciones: {len(yolo_detections)}")
    for d in yolo_detections[:3]:
        print(f"   - ({d['x']:.0f}, {d['y']:.0f}), r={d['radius']:.1f}, conf={d['confidence']:.2f}")
    all_detections.extend([('YOLO', d) for d in yolo_detections])
    
    # Método 2: Template matching
    print("\n🔍 Método 2: Template matching circular...")
    template_detections = detect_ball_template_matching(main_frame)
    print(f"   Detecciones: {len(template_detections)}")
    for d in template_detections[:3]:
        print(f"   - ({d['x']:.0f}, {d['y']:.0f}), r={d['radius']:.1f}, score={d['score']:.2f}")
    all_detections.extend([('Template', d) for d in template_detections])
    
    # Método 3: Sustracción de fondo
    print("\n🔍 Método 3: Sustracción de fondo...")
    bg_detections = detect_ball_background_subtraction(frames)
    print(f"   Detecciones: {len(bg_detections)}")
    for d in bg_detections[:3]:
        print(f"   - ({d['x']:.0f}, {d['y']:.0f}), r={d['radius']:.1f}")
    all_detections.extend([('BGSub', d) for d in bg_detections])
    
    # Visualizar resultados
    display = main_frame.copy()
    
    # Dibujar todas las detecciones
    colors = {'YOLO': (0, 255, 0), 'Template': (255, 0, 0), 'BGSub': (0, 0, 255)}
    
    for method, det in all_detections:
        color = colors.get(method, (128, 128, 128))
        cv2.circle(display, (int(det['x']), int(det['y'])), int(det['radius']), color, 2)
        cv2.putText(display, method, (int(det['x']) + 5, int(det['y']) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Leyenda
    y = 30
    for method, color in colors.items():
        cv2.putText(display, f"{method}: {len([d for m, d in all_detections if m == method])}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 25
    
    # Guardar resultado
    output_path = 'runs/spike3_ball/tracknet_test.jpg'
    cv2.imwrite(output_path, display)
    print(f"\n💾 Resultado guardado en: {output_path}")
    
    # Mostrar resultado
    cv2.imshow('Detection Results', display)
    print("\n⌨️ Presiona 'q' para salir")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Sugerir próximos pasos
    print("\n" + "=" * 60)
    print("📌 PRÓXIMOS PASOS:")
    print("=" * 60)
    
    if not torch_available:
        print("1. Instalar PyTorch:")
        print("   pip install torch torchvision")
    
    if not tracknet_path:
        print("2. Instalar TrackNet:")
        print("   git clone https://github.com/Chang-Chia-Chi/TrackNetv2")
        print("   cd TrackNetv2 && pip install -r requirements.txt")
    
    if yolo_detections:
        print("\n✅ YOLO detectó 'sports ball' - puede ser útil como baseline")
    
    if not all_detections:
        print("\n⚠️ Ningún método detectó la pelota")
        print("   Considerar:")
        print("   - Etiquetar datos manualmente para entrenar modelo específico")
        print("   - Usar servicios de anotación como CVAT, LabelImg")


if __name__ == "__main__":
    main()