# Spike 1 - Resultados de Validación de YOLO en Videos de Pádel

**Fecha**: 21 Febrero 2026  
**Objetivo**: Confirmar que YOLO detecta correctamente los 4 jugadores en videos reales de pádel con cámara cenital.

## Resumen Ejecutivo

**Resultado Final**: ✅ **EXITOSO**

| Métrica | Sin Filtrado | Con YOLOv8m |
|---------|-------------|-------------|
| Frames con 4 jugadores | 1.6% | **99.0%** |
| Frames con 3-5 jugadores | ~20% | **100%** |
| Promedio jugadores/frame | 6.4 | 3.99 |
| Mejora | - | **62x mejor** |

---

## Comparativa de Modelos YOLO

| Modelo | Frames con 4 | Promedio | Tamaño | Recomendación |
|--------|-------------|----------|--------|---------------|
| YOLOv8n | 37% | 3.30 | 6.3MB | Baseline |
| YOLOv26n | 0% | 2.10 | 5.3MB | ❌ No usar |
| YOLOv8s | 73% | 3.73 | 21.5MB | ✅ Bueno |
| YOLOv11n | 42% | 3.35 | 5.4MB | ✅ Rápido |
| YOLOv11s | 82% | 3.82 | 18.4MB | ✅ Muy bueno |
| YOLOv26s | 58% | 3.57 | 19.5MB | ⚠️ Regular |
| **YOLOv8m** | **99%** | **3.99** | 49.7MB | ✅ **GANADOR** |
| YOLOv11m | 93% | 3.93 | 38.8MB | ✅ Alternativa eficiente |
| YOLOv26m | 89% | 3.89 | 42.2MB | ⚠️ Regular |

### Ganador: YOLOv8m

**99% de frames con exactamente 4 jugadores detectados.**

---

## Solución Final Implementada

### 1. Selección Manual de Cancha
Como la cámara está fija, el usuario selecciona los 4 vértices de la cancha una sola vez:
```python
# Archivo: spike1_manual_selection.py
# Genera: runs/manual_court/court_corners.json
```

### 2. Filtrado por Posición de Pies
Se verifica si los **pies** del jugador (parte inferior del bbox) están dentro del polígono:
```python
cy_bottom = y2  # Coordenada Y inferior (pies)
in_court = point_in_polygon((cx, cy_bottom), polygon)
```

### 3. Non-Maximum Suppression (NMS)
Elimina detecciones duplicadas del mismo jugador:
```python
def nms(detections, iou_threshold=0.3):
    # Elimina detecciones con IoU > threshold
```

### 4. YOLOv8m con Tracking
Modelo medium con tracking temporal:
```python
model = YOLO('yolov8m.pt')
results = model.track(frame, classes=[0], persist=True, conf=0.5)
```

---

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `spike1_manual_selection.py` | Selección interactiva de los 4 vértices de la cancha |
| `spike1_tracking.py` | Filtrado de jugadores con YOLOv8m + Tracking |
| `spike1_examples.py` | Generador de ejemplos visuales |
| `runs/manual_court/court_corners.json` | Coordenadas de la cancha |

---

## Análisis de Detección por Jugador

Con YOLOv8m, los 4 jugadores se detectan consistentemente:

| Track ID | Detecciones | Porcentaje |
|----------|-------------|------------|
| ID 1 | 100/100 | 100% |
| ID 2 | 99/100 | 99% |
| ID 3 | 100/100 | 100% |
| ID 4 | 100/100 | 100% |

**Nota**: El jugador más lejano (del fondo) es el más difícil de detectar, pero YOLOv8m lo detecta en el 100% de los frames.

---

## Lecciones Aprendidas

### Lo que funcionó
- ✅ **Selección manual de cancha** (cámara fija)
- ✅ **Filtrado por pies** (no centro del bbox)
- ✅ **Non-Maximum Suppression (NMS)**
- ✅ **YOLOv8m** - modelo medium para objetos lejanos
- ✅ **Tracking temporal** con `model.track()`

### Lo que NO funcionó
- ❌ Detección automática de piso por color
- ❌ Detección automática de líneas (Hough)
- ❌ YOLO segmentación para detectar cancha
- ❌ Polígonos de perspectiva automática
- ❌ Filtrado por centro del bbox
- ❌ **YOLOv26** en todas sus variantes (n, s, m)

### Hallazgos clave
- Los jugadores del fondo son más difíciles de detectar (más lejanos/más pequeños)
- YOLOv8m supera a YOLOv11m y YOLOv26m
- Los modelos "small" (s) son un buen balance entre velocidad y precisión
- El tracking de YOLO no "inventa" detecciones, solo mantiene IDs

---

## Recomendaciones

### Para Producción
1. **Modelo recomendado**: YOLOv8m (99% precisión) o YOLOv11m (93% precisión, 22% más pequeño)
2. **Confianza mínima**: 0.5
3. **IoU threshold para NMS**: 0.3
4. **Máximo jugadores**: 4 (limitar después de filtrar)

### Para Mejorar Performance
1. Considerar YOLOv11m si el tamaño del modelo es crítico (38MB vs 49MB)
2. Implementar cache de detecciones para frames consecutivos similares
3. Considerar GPU para procesamiento en tiempo real

---

## Conclusión

**Spike 1 - ESTADO**: ✅ **EXITOSO**

- ✅ **99% de frames con exactamente 4 jugadores**
- ✅ 100% de frames con 3-5 jugadores
- ✅ Detección de cancha correcta con selección manual
- ✅ NMS elimina duplicados
- ✅ Tracking temporal mantiene IDs consistentes

**La combinación YOLOv8m + filtrado por pies + NMS logra detección casi perfecta.**

---

## Próximos Pasos

1. **Spike 2**: Análisis de movimiento de jugadores
2. **Spike 3**: Detección de eventos de juego (golpes, pelota)
3. **MVP**: Integración en pipeline de procesamiento