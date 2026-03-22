# 🤖 Modelos de Machine Learning

Este directorio contiene los modelos de YOLO utilizados para la detección de jugadores y objetos.

## Modelos Requeridos

| Modelo | Tamaño | Uso | Descarga |
|--------|--------|-----|----------|
| `yolov8m.pt` | ~50MB | Detección de jugadores (Spike 1, 2) | `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt` |
| `yolo11m.pt` | ~50MB | Versión más reciente de YOLO | `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt` |

## Nota

Los archivos `.pt` están excluidos del repositorio mediante `.gitignore` debido a su tamaño.
El script los descargará automáticamente la primera vez que se ejecute si no están presentes.

## Uso en los Scripts

```python
from ultralytics import YOLO

# El modelo se descarga automáticamente si no existe
model = YOLO('yolov8m.pt')  # Busca en el directorio actual
model = YOLO('../models/yolov8m.pt')  # Ruta relativa desde spikes/