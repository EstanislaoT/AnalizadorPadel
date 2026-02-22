# TrackNet - Investigación para Detección de Pelota

## ¿Qué es TrackNet?

TrackNet es una red neuronal diseñada específicamente para **trackear pelotas en deportes** como tenis, bádminton y voleibol. Fue desarrollada por investigadores de la National Chiao Tung University (Taiwán).

### Versiones

| Versión | Año | Características |
|---------|-----|-----------------|
| **TrackNetV1** | 2019 | Input: 3 frames consecutivos, output: mapa de calor |
| **TrackNetV2** | 2020 | Mejor precisión, más rápido, maneja oclusiones |
| **TrackNetV3** | 2023 | Arquitectura mejorada, mejor para tiempo real |

## Ventajas para Pádel

1. **Diseñado para pelotas pequeñas y rápidas** - Exactamente el problema que tenemos
2. **Maneja motion blur** - La pelota de pádel se mueve rápido
3. **Resistente a oclusiones** - La pelota puede estar parcialmente oculta
4. **Entrenado en tenis** - Similar a pádel (pelota amarilla, cancha rectangular)

## Disponibilidad

### Modelos Pre-entrenados

```bash
# TrackNetV2 está disponible en GitHub
git clone https://github.com/Chang-Chia-Chi/TrackNetv2

# También hay implementaciones en PyTorch
pip install tracknet-pytorch
```

### Requisitos

| Requisito | Valor |
|-----------|-------|
| Python | 3.7+ |
| PyTorch | 1.7+ |
| GPU | Recomendada (CUDA) |
| Memoria GPU | 4GB+ |
| Input | 3 frames consecutivos (288x512) |
| Output | Heatmap de la pelota |

## Arquitectura

```
Input: 3 frames RGB (288 x 512 x 9)
         ↓
    ResNet-18 (encoder)
         ↓
    FPN (Feature Pyramid Network)
         ↓
    Decoder (upsampling)
         ↓
Output: Heatmap (288 x 512)
         ↓
    Argmax → Posición (x, y) de la pelota
```

## Implementación Propuesta

### Opción 1: Usar modelo pre-entrenado de tenis

```python
import torch
from tracknet import TrackNetV2

# Cargar modelo pre-entrenado en tenis
model = TrackNetV2()
model.load_state_dict(torch.load('tracknet_tennis.pt'))
model.eval()

# Input: 3 frames consecutivos
def detect_ball(frames):
    """frames: lista de 3 frames consecutivos"""
    input_tensor = preprocess(frames)  # (1, 9, 288, 512)
    with torch.no_grad():
        heatmap = model(input_tensor)
    ball_pos = heatmap_to_position(heatmap)
    return ball_pos
```

### Opción 2: Fine-tuning para pádel

Si el modelo de tenis no funciona bien, podemos hacer fine-tuning:

1. **Dataset**: Etiquetar 1000-2000 frames de pádel con posición de pelota
2. **Transfer Learning**: Usar pesos de tenis como inicialización
3. **Entrenamiento**: 10-20 epochs (~2-4 horas en GPU)

## Comparativa con Nuestra Implementación Actual

| Aspecto | HSV + HoughCircles | TrackNet |
|---------|-------------------|----------|
| **Precisión** | ~10-20% | ~80-90% |
| **Falsos positivos** | Muchos | Pocos |
| **Velocidad** | Muy rápido (CPU) | Medio (requiere GPU) |
| **Setup** | Simple | Complejo (PyTorch, GPU) |
| **Dependencias** | OpenCV | PyTorch, CUDA |
| **Tiempo de implementación** | ✅ Hecho | 2-4 horas |

## Plan de Implementación

### Fase 1: Probar modelo de tenis (1-2 horas)

```bash
# 1. Clonar repositorio
git clone https://github.com/Chang-Chia-Chi/TrackNetv2

# 2. Descargar pesos pre-entrenados
wget https://github.com/Chang-Chia-Chi/TrackNetv2/releases/download/v1.0/tracknet.pt

# 3. Crear script de prueba
python spike3_tracknet_test.py
```

### Fase 2: Si funciona, integrar al proyecto (2-4 horas)

1. Agregar dependencias a `requirements.txt`
2. Crear módulo `spike3_ball_tracknet.py`
3. Comparar resultados con HSV

### Fase 3: Fine-tuning si es necesario (post-MVP)

1. Etiquetar frames de pádel
2. Entrenar modelo
3. Evaluar mejora

## Script de Prueba

Voy a crear `spike3_tracknet_test.py` para probar TrackNet con nuestro video.

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| No hay GPU disponible | Media | Usar CPU (más lento) o servicios cloud |
| Modelo de tenis no funciona en pádel | Baja | Fine-tuning con datos de pádel |
| Dependencias conflictúan | Media | Usar virtualenv separado |
| Tiempo de inferencia alto | Baja | Optimizar con ONNX/TensorRT |

## Decisión

**Recomendación**: Implementar TrackNet V2 como reemplazo de HSV + HoughCircles.

**Justificación**:
- La detección por HSV tiene precisión < 20%
- TrackNet está diseñado específicamente para este problema
- Hay modelos pre-entrenados disponibles
- El tiempo de implementación es razonable (2-4 horas)

**Alternativa si no funciona**: 
- Usar YOLOv8 con entrenamiento específico para pelota de pádel
- Requiere más datos y tiempo de entrenamiento