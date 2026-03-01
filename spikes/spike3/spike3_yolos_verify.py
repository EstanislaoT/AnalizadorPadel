#!/usr/bin/env python3
"""
Spike 3 - Verificación de detecciones YOLOS

Muestra qué está detectando YOLOS y permite al usuario marcar
la posición real de la pelota para comparar.
"""

import cv2
import numpy as np
import sys
import json

# Cargar modelo YOLOS
_yolos_model = None
_yolos_processor = None


def get_yolos_model():
    global _yolos_model, _yolos_processor
    if _yolos_model is None:
        from transformers import YolosImageProcessor, YolosForObjectDetection
        import torch
        print("🔄 Cargando YOLOS...")
        _yolos_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
        _yolos_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _yolos_model = _yolos_model.to(device)
    return _yolos_processor, _yolos_model


def detect_all_objects(frame, threshold=0.2):
    """Detecta TODOS los objetos (no solo pelotas) para ver qué encuentra YOLOS."""
    import torch
    
    processor, model = get_yolos_model()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    inputs = processor(images=frame_rgb, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        box = box.tolist()
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append({
            'x': cx, 'y': cy,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': score.item(),
            'label': label_name,
            'label_id': label.item()
        })
    
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections


def main():
    if len(sys.argv) < 3:
        print("Uso: python spike3_yolos_verify.py <video_path> <corners_path> [frame_idx]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    corners_path = sys.argv[2]
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Cargar cancha
    with open(corners_path, 'r') as f:
        corners = json.load(f)
    points = np.array([corners['TL'], corners['TR'], corners['BR'], corners['BL']], dtype=np.int32)
    polygon = points.reshape((-1, 1, 2))
    
    # Extraer frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    print(f"📹 Frame {frame_idx}")
    
    # Detectar TODOS los objetos
    print("\n🔍 Detectando objetos con YOLOS...")
    detections = detect_all_objects(frame, threshold=0.2)
    
    print(f"\n📊 Total detecciones: {len(detections)}")
    
    # Mostrar por tipo
    labels_count = {}
    for d in detections:
        labels_count[d['label']] = labels_count.get(d['label'], 0) + 1
    
    print("\nPor tipo:")
    for label, count in sorted(labels_count.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    
    # Crear visualización
    display = frame.copy()
    cv2.polylines(display, [polygon], True, (255, 255, 0), 2)
    
    # Dibujar TODAS las detecciones
    for i, d in enumerate(detections):
        # Color según tipo
        if 'ball' in d['label'].lower():
            color = (0, 255, 0)  # Verde para pelotas
        elif 'person' in d['label'].lower():
            color = (255, 0, 0)  # Azul para personas
        elif 'sports' in d['label'].lower():
            color = (0, 255, 255)  # Cyan para sports equipment
        else:
            color = (128, 128, 128)  # Gris para otros
        
        # Dibujar caja
        cv2.rectangle(display, (int(d['x1']), int(d['y1'])), (int(d['x2']), int(d['y2'])), color, 2)
        
        # Etiqueta
        label_text = f"{i+1}. {d['label']} {d['confidence']:.0%}"
        cv2.putText(display, label_text, (int(d['x1']), int(d['y1']) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Lista de detecciones en consola
    print("\n📋 Lista de detecciones:")
    for i, d in enumerate(detections[:20]):  # Primeras 20
        in_court = cv2.pointPolygonTest(polygon, (int(d['x']), int(d['y'])), False) >= 0
        court_str = "✅ EN CANCHA" if in_court else "❌ fuera"
        print(f"  {i+1}. {d['label']}: ({d['x']:.0f}, {d['y']:.0f}) conf={d['confidence']:.0%} {court_str}")
    
    # Guardar
    output_path = 'runs/spike3_ball/yolos_verify.jpg'
    cv2.imwrite(output_path, display)
    print(f"\n💾 Imagen guardada en: {output_path}")
    
    # Mostrar
    cv2.imshow('YOLOS Detections', display)
    print("\n⌨️ Presiona 'q' para salir")
    
    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()