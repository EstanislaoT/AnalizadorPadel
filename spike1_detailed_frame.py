#!/usr/bin/env python3
"""
Análisis detallado de un solo frame para identificar cada detección.
"""

import cv2
from ultralytics import YOLO

def analyze_single_frame(video_path: str, frame_number: int = 0):
    """
    Analiza un frame y muestra las coordenadas exactas de cada detección.
    """
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame")
        return
    
    height, width = frame.shape[:2]
    print(f"Frame {frame_number} - Resolución: {width}x{height}")
    
    # Zona de cancha
    court_left = int(width * 0.15)
    court_right = width - court_left
    court_top = int(height * 0.12)
    court_bottom = height - court_top
    
    results = model(frame, classes=[0], verbose=False)
    
    print(f"\n📍 DETECCIONES EN FRAME {frame_number}:")
    print("="*60)
    
    detections = []
    for r in results:
        for i, box in enumerate(r.boxes):
            if box.cls[0] == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Calcular posición relativa
                pos_x = "CENTRO" if court_left < cx < court_right else ("IZQUIERDA" if cx < court_left else "DERECHA")
                pos_y = "CENTRO" if court_top < cy < court_bottom else ("ARRIBA" if cy < court_top else "ABAJO")
                
                is_inside = (court_left <= cx <= court_right and court_top <= cy <= court_bottom)
                
                detection = {
                    'id': i + 1,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'confidence': conf,
                    'position': f"{pos_y}-{pos_x}",
                    'inside_court': is_inside
                }
                detections.append(detection)
                
                status = "✅ DENTRO" if is_inside else "❌ FUERA"
                print(f"\n   Persona #{i+1}:")
                print(f"   ├── BBox: ({x1}, {y1}) → ({x2}, {y2})")
                print(f"   ├── Centro: ({cx}, {cy})")
                print(f"   ├── Confianza: {conf:.3f}")
                print(f"   ├── Posición: {pos_y}-{pos_x}")
                print(f"   └── Estado: {status}")
    
    # Dibujar en la imagen
    output = frame.copy()
    
    # Zona de cancha
    cv2.rectangle(output, (court_left, court_top), (court_right, court_bottom), (255, 0, 0), 2)
    cv2.putText(output, "CANCH A", (court_left + 10, court_top + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Cada detección con número
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['center']
        
        color = (0, 255, 0) if det['inside_court'] else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, (cx, cy), 8, color, -1)
        cv2.putText(output, f"#{det['id']}", (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Guardar
    output_path = f"runs/analysis/frame_{frame_number}_detailed.jpg"
    cv2.imwrite(output_path, output)
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN:")
    print(f"   Total detecciones: {len(detections)}")
    print(f"   Dentro de cancha: {sum(1 for d in detections if d['inside_court'])}")
    print(f"   Fuera de cancha: {sum(1 for d in detections if not d['inside_court'])}")
    print(f"\n📁 Imagen guardada: {output_path}")
    
    cap.release()
    
    return detections

if __name__ == "__main__":
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    analyze_single_frame(video_path, frame_number=0)