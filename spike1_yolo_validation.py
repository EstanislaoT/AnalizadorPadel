#!/usr/bin/env python3
"""
Spike 1 - Validación de YOLO en Videos de Pádel

Objetivo: Confirmar que YOLO v8 detecta correctamente los 4 jugadores 
en videos reales de pádel con cámara cenital.

Criterio de éxito: Detección correcta de los 4 jugadores en > 85% de los frames.
"""

import cv2
import time
from ultralytics import YOLO
from collections import Counter

def validate_yolo_on_padel(video_path: str, max_seconds: int = 30):
    """
    Ejecuta detección YOLO en un video de pádel y analiza resultados.
    
    Args:
        video_path: Ruta al video de prueba
        max_seconds: Máximo de segundos a procesar (default 30)
    """
    print(f"🔄 Cargando modelo YOLO v8 nano...")
    model = YOLO('yolov8n.pt')  # nano model, más rápido
    
    print(f"📹 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: No se puede abrir el video")
        return None
    
    # Info del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   FPS: {fps:.2f}")
    print(f"   Resolución: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duración: {total_frames/fps:.1f} segundos")
    
    # Limitar a max_seconds
    frames_to_process = min(int(fps * max_seconds), total_frames)
    print(f"\n🎯 Procesando {frames_to_process} frames ({max_seconds}s)...")
    
    # Estadísticas
    person_counts = []
    frames_with_4_players = 0
    frames_with_3_plus_players = 0
    frames_with_2_plus_players = 0
    total_confidence = []
    
    start_time = time.time()
    
    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Frame {frame_idx} no se pudo leer")
            break
        
        # Ejecutar YOLO (solo clase 0 = persona)
        results = model(frame, classes=[0], verbose=False)
        
        # Contar personas detectadas
        person_count = 0
        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:  # persona
                    person_count += 1
                    total_confidence.append(float(box.conf[0]))
        
        person_counts.append(person_count)
        
        if person_count == 4:
            frames_with_4_players += 1
        if person_count >= 3:
            frames_with_3_plus_players += 1
        if person_count >= 2:
            frames_with_2_plus_players += 1
        
        # Progreso cada 100 frames
        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps_processed = (frame_idx + 1) / elapsed
            print(f"   Frame {frame_idx + 1}/{frames_to_process} | FPS: {fps_processed:.1f} | Último: {person_count} personas")
    
    cap.release()
    
    # Análisis de resultados
    elapsed_time = time.time() - start_time
    processed_frames = len(person_counts)
    
    print(f"\n{'='*60}")
    print("📊 RESULTADOS DEL ANÁLISIS")
    print(f"{'='*60}")
    
    print(f"\n⏱️  Tiempo de procesamiento: {elapsed_time:.1f}s")
    print(f"   FPS de procesamiento: {processed_frames/elapsed_time:.1f}")
    
    print(f"\n👤 DETECCIÓN DE PERSONAS:")
    print(f"   Frames procesados: {processed_frames}")
    print(f"   Frames con exactamente 4 jugadores: {frames_with_4_players} ({100*frames_with_4_players/processed_frames:.1f}%)")
    print(f"   Frames con 3+ personas: {frames_with_3_plus_players} ({100*frames_with_3_plus_players/processed_frames:.1f}%)")
    print(f"   Frames con 2+ personas: {frames_with_2_plus_players} ({100*frames_with_2_plus_players/processed_frames:.1f}%)")
    
    # Distribución de conteos
    count_dist = Counter(person_counts)
    print(f"\n📈 Distribución de detecciones:")
    for count in sorted(count_dist.keys()):
        pct = 100 * count_dist[count] / processed_frames
        bar = "█" * int(pct / 2)
        print(f"   {count} personas: {count_dist[count]:4d} frames ({pct:5.1f}%) {bar}")
    
    # Confianza promedio
    if total_confidence:
        avg_conf = sum(total_confidence) / len(total_confidence)
        print(f"\n🎯 Confianza promedio: {avg_conf:.3f}")
    
    # Veredicto
    success_rate = 100 * frames_with_4_players / processed_frames
    print(f"\n{'='*60}")
    if success_rate >= 85:
        print(f"✅ SPIKE 1 EXITOSO: {success_rate:.1f}% frames con 4 jugadores (≥85% requerido)")
    elif frames_with_3_plus_players / processed_frames >= 0.85:
        print(f"⚠️  SPIKE 1 PARCIAL: {success_rate:.1f}% con 4 jugadores, pero {100*frames_with_3_plus_players/processed_frames:.1f}% con 3+")
        print("   Recomendación: Revisar falsos negativos o jugadores ocluidos")
    else:
        print(f"❌ SPIKE 1 FALLIDO: {success_rate:.1f}% frames con 4 jugadores (≥85% requerido)")
        print("   Plan B: Probar YOLO v8 medium/large o fine-tuning")
    print(f"{'='*60}")
    
    return {
        "success_rate": success_rate,
        "frames_processed": processed_frames,
        "processing_fps": processed_frames/elapsed_time,
        "avg_confidence": avg_conf if total_confidence else 0
    }


def run_detection_with_output(video_path: str, max_seconds: int = 10, output_path: str = "runs/spike1_output.mp4"):
    """
    Ejecuta detección YOLO y genera video con bounding boxes.
    Útil para verificación visual.
    """
    print(f"🔄 Generando video con detecciones...")
    
    model = YOLO('yolov8n.pt')
    
    # Ejecutar detección con save=True para guardar resultado
    results = model(
        video_path,
        classes=[0],  # solo personas
        save=True,
        vid_stride=2,  # procesar 1 de cada 2 frames para acelerar
        name="spike1",
        project="runs/detect",
        verbose=True
    )
    
    print(f"\n✅ Video guardado en: runs/detect/spike1/")
    return results


if __name__ == "__main__":
    import sys
    
    video_path = "test-videos/Final Reserve Cup Miami 2026 Coello _ Chingotto vs Tapia _ Galán Full Match Partidazo - Flash Padel (720p, h264).mp4"
    
    # Ejecutar análisis estadístico
    print("🏃 Iniciando Spike 1 - Validación de YOLO en Pádel\n")
    stats = validate_yolo_on_padel(video_path, max_seconds=30)
    
    # Preguntar si generar video con detecciones
    print("\n" + "="*60)
    print("Para verificación visual, ejecuta:")
    print("  python3 spike1_yolo_validation.py --video-output")
    print("="*60)