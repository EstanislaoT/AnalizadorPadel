#!/usr/bin/env python3
"""
Script Python para procesar videos de pádel.
Este script es llamado desde .NET via subprocess.

Uso: python3 process_video.py <video_path> <output_path> <models_path>

Args:
    video_path: Ruta al video de entrada
    output_path: Ruta donde guardar resultados JSON
    models_path: Ruta a los modelos YOLO
"""

import sys
import json
import os
import time
from pathlib import Path
from ultralytics import YOLO


def main():
    """Función principal de procesamiento."""
    try:
        # Parsear argumentos
        if len(sys.argv) != 4:
            raise ValueError("Se requieren 3 argumentos: video_path, output_path, models_path")
        
        video_path = sys.argv[1]
        output_path = sys.argv[2]
        models_path = sys.argv[3]
        
        # Validar inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        if not os.path.exists(models_path):
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {models_path}")
        
        print(f"🎾 Procesando video: {video_path}")
        print(f"📁 Output: {output_path}")
        print(f"🤖 Models: {models_path}")
        
        # Iniciar timer
        start_time = time.time()
        
        # Cargar modelo YOLO
        model_path = os.path.join(models_path, "yolov8m.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {model_path}")
        
        print("🔄 Cargando modelo YOLO...")
        model = YOLO(model_path)
        
        # Procesar video (versión simplificada para el spike)
        print("🔍 Analizando video...")
        results = model(video_path, classes=[0])  # Solo clase 'person'
        
        # Extraer estadísticas básicas
        total_frames = len(results)
        detections_per_frame = []
        players_detected = set()
        
        for frame_idx, result in enumerate(results):
            frame_detections = len(result.boxes)
            detections_per_frame.append(frame_detections)
            
            # Contar jugadores únicos (simplificado)
            if frame_detections >= 4:
                players_detected.add('player1')
                players_detected.add('player2')
                players_detected.add('player3')
                players_detected.add('player4')
        
        # Calcular métricas
        processing_time = time.time() - start_time
        avg_detections = sum(detections_per_frame) / len(detections_per_frame) if detections_per_frame else 0
        frames_with_4_players = sum(1 for d in detections_per_frame if d >= 4)
        detection_rate = (frames_with_4_players / total_frames * 100) if total_frames > 0 else 0
        
        # Preparar resultado
        result_data = {
            "status": "success",
            "video_path": video_path,
            "processing_time_seconds": round(processing_time, 2),
            "total_frames": total_frames,
            "players_detected": len(players_detected),
            "avg_detections_per_frame": round(avg_detections, 2),
            "frames_with_4_players": frames_with_4_players,
            "detection_rate_percent": round(detection_rate, 2),
            "model_used": "yolov8m.pt",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Guardar resultados
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Output para .NET
        print(f"✅ Procesamiento completado en {processing_time:.2f}s")
        print(f"📊 Detección: {detection_rate:.1f}% frames con 4+ jugadores")
        
        # Retornar JSON por stdout para que .NET lo capture
        print(json.dumps(result_data))
        
        return 0
        
    except Exception as e:
        # Manejo de errores
        error_data = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Imprimir error por stderr
        print(f"❌ ERROR: {error_data['error_type']}: {error_data['error_message']}", file=sys.stderr)
        
        # También imprimir JSON por stderr para .NET
        print(json.dumps(error_data), file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())