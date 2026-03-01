#!/usr/bin/env python3
"""
Spike 3 - Prueba con modelos de HuggingFace para detección de pelota

Busca y prueba modelos pre-entrenados que puedan detectar
pelotas pequeñas (tenis, fútbol, etc.)
"""

import cv2
import numpy as np
import sys
import os

os.makedirs('runs/spike3_ball', exist_ok=True)


def test_owlvit(frame):
    """
    Prueba OWL-ViT (Open-World Localization with Vision Transformers).
    
    Permite detectar objetos con descripciones de texto libre.
    """
    try:
        from transformers import pipeline
        
        print("\n🔍 Probando OWL-ViT...")
        
        # Cargar modelo
        detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")
        
        # Detectar con texto libre
        predictions = detector(
            frame,
            candidate_labels=["yellow tennis ball", "small ball", "sports ball", "yellow ball"],
            threshold=0.1
        )
        
        balls = []
        for pred in predictions:
            box = pred['box']
            cx = (box['xmin'] + box['xmax']) / 2
            cy = (box['ymin'] + box['ymax']) / 2
            radius = (box['xmax'] - box['xmin'] + box['ymax'] - box['ymin']) / 4
            balls.append({
                'x': cx, 'y': cy, 'radius': radius,
                'confidence': pred['score'],
                'label': pred['label']
            })
        
        return balls
    except Exception as e:
        print(f"Error con OWL-ViT: {e}")
        return []


def test_detr(frame):
    """
    Prueba DETR (DEtection TRansformer) de Facebook.
    
    Modelo de detección de objetos con transformers.
    """
    try:
        from transformers import DetrImageProcessor, DetrForObjectDetection
        import torch
        
        print("\n🔍 Probando DETR...")
        
        # Cargar modelo
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Preparar imagen
        inputs = processor(images=frame, return_tensors="pt")
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-procesar
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
        
        balls = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[label.item()]
            # COCO tiene 'sports ball' como clase 37
            if 'ball' in label_name.lower() or label.item() == 37:
                box = box.tolist()
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                radius = (box[2] - box[0] + box[3] - box[1]) / 4
                balls.append({
                    'x': cx, 'y': cy, 'radius': radius,
                    'confidence': score.item(),
                    'label': label_name
                })
        
        return balls
    except Exception as e:
        print(f"Error con DETR: {e}")
        return []


def test_yolos(frame):
    """
    Prueba YOLOS (YOLO for Object Detection with Transformers).
    
    Versión de YOLO basada en transformers.
    """
    try:
        from transformers import YolosImageProcessor, YolosForObjectDetection
        import torch
        
        print("\n🔍 Probando YOLOS...")
        
        # Cargar modelo
        processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
        model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
        
        # Preparar imagen
        inputs = processor(images=frame, return_tensors="pt")
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-procesar
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
        
        balls = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = model.config.id2label[label.item()]
            # COCO tiene 'sports ball' como clase 37
            if 'ball' in label_name.lower() or label.item() == 37:
                box = box.tolist()
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                radius = (box[2] - box[0] + box[3] - box[1]) / 4
                balls.append({
                    'x': cx, 'y': cy, 'radius': radius,
                    'confidence': score.item(),
                    'label': label_name
                })
        
        return balls
    except Exception as e:
        print(f"Error con YOLOS: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Uso: python spike3_huggingface_test.py <video_path> [frame_idx]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print("🎾 Spike 3 - Prueba con HuggingFace")
    print("=" * 60)
    
    # Extraer frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ No se pudo extraer el frame")
        return
    
    print(f"📹 Frame {frame_idx} extraído: {frame.shape[1]}x{frame.shape[0]}")
    
    # Frame RGB para HuggingFace
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    all_detections = []
    
    # Probar OWL-ViT (zero-shot)
    owlvit_detections = test_owlvit(frame_rgb)
    print(f"   OWL-ViT: {len(owlvit_detections)} detecciones")
    for d in owlvit_detections[:3]:
        print(f"   - {d['label']}: ({d['x']:.0f}, {d['y']:.0f}), conf={d['confidence']:.2f}")
    all_detections.extend([('OWL-ViT', d) for d in owlvit_detections])
    
    # Probar DETR
    detr_detections = test_detr(frame_rgb)
    print(f"   DETR: {len(detr_detections)} detecciones")
    for d in detr_detections[:3]:
        print(f"   - {d['label']}: ({d['x']:.0f}, {d['y']:.0f}), conf={d['confidence']:.2f}")
    all_detections.extend([('DETR', d) for d in detr_detections])
    
    # Probar YOLOS
    yolos_detections = test_yolos(frame_rgb)
    print(f"   YOLOS: {len(yolos_detections)} detecciones")
    for d in yolos_detections[:3]:
        print(f"   - {d['label']}: ({d['x']:.0f}, {d['y']:.0f}), conf={d['confidence']:.2f}")
    all_detections.extend([('YOLOS', d) for d in yolos_detections])
    
    # Visualizar
    display = frame.copy()
    colors = {'OWL-ViT': (0, 255, 0), 'DETR': (255, 0, 0), 'YOLOS': (0, 0, 255)}
    
    for method, det in all_detections:
        color = colors.get(method, (128, 128, 128))
        cv2.circle(display, (int(det['x']), int(det['y'])), int(det['radius']), color, 2)
        cv2.putText(display, f"{method}", (int(det['x']) + 5, int(det['y']) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Leyenda
    y = 30
    for method, color in colors.items():
        count = len([d for m, d in all_detections if m == method])
        cv2.putText(display, f"{method}: {count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 25
    
    # Guardar
    output_path = 'runs/spike3_ball/huggingface_test.jpg'
    cv2.imwrite(output_path, display)
    print(f"\n💾 Resultado guardado en: {output_path}")
    
    # Mostrar
    cv2.imshow('HuggingFace Detection', display)
    print("\n⌨️ Presiona 'q' para salir")
    
    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()