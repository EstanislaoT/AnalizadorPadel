# Spike 3 - Resultados: Detección de Pelota de Pádel

**Fecha**: 28 Febrero 2026  
**Objetivo**: Evaluar métodos para detectar la pelota de pádel en videos.

## Resumen Ejecutivo

**Resultado Final**: ⚠️ **PARCIALMENTE EXITOSO**

| Método | Tasa de Detección | Precisión | Conclusión |
|--------|-------------------|-----------|------------|
| YOLO (sports ball) | 0% | N/A | ❌ No funciona |
| HSV (color) | 100% | ~30% | ⚠️ Muchos falsos positivos |
| TrackNet | No probado | N/A | Pendiente de instalación |

---

## Prueba 1: YOLO con Clase Sports Ball

### Configuración
- **Modelo**: YOLOv8m
- **Clase COCO**: 37 (sports ball)
- **Video**: ProPadel2.mp4 (300 frames)
- **Umbral confianza**: 0.25

### Resultados
```
Total detecciones: 0
Frames con detección: 0/300 (0.0%)
```

### Análisis
La clase "sports ball" de COCO está entrenada principalmente con:
- Pelotas de fútbol (grandes, blanco/negro)
- Pelotas de básquet (naranjas, grandes)
- Pelotas de tenis en contexto de partido

**Por qué falló con pádel**:
- La pelota de pádel es más pequeña que las pelotas en COCO
- La vista cenital no es común en el dataset de entrenamiento
- La pelota de pádel se mueve muy rápido (motion blur)

---

## Prueba 2: Detección por Color HSV

### Configuración
- **Rango HSV**: H(20-50), S(100-255), V(100-255)
- **Video**: ProPadel2.mp4 (300 frames)
- **Radio esperado**: 3-20 píxeles

### Resultados
```
Total detecciones: 3229
Detecciones en cancha: 968
Frames con detección: 300/300 (100.0%)
```

### Análisis
**Problema detectado**: Muchos falsos positivos

El color HSV detecta:
- ✅ La pelota amarilla/verde
- ❌ Líneas de la cancha (amarillas)
- ❌ Ropa de jugadores (colores similares)
- ❌ Reflejos en el vidrio
- ❌ Publicidad en fondos

**Ratio de precisión estimado**: 968/3229 ≈ 30%

### Métricas Guardadas
- Video: `spikes/spikes/runs/spike3_ball_yolo/ball_detection_yolo.mp4`
- JSON: `spikes/spikes/runs/spike3_ball_yolo/metrics.json`

---

## Conclusiones

### ❌ YOLO Sports Ball - No Viable
El modelo pre-entrenado de YOLO no detecta pelotas de pádel. No es una opción para el MVP.

### ⚠️ HSV - Viable con Mejoras
La detección por color funciona pero requiere filtros adicionales:

**Mejoras propuestas**:
1. **Filtrar por forma**: Solo aceptar objetos circulares (circularidad > 0.7)
2. **Filtrar por tamaño**: Ajustar rango de radio según posición en cancha
3. **Filtrar por movimiento**: Solo considerar objetos en movimiento
4. **Filtrar por posición**: Excluir líneas de cancha (conocidas)
5. **Tracking temporal**: Usar Kalman filter para predecir posición

### 🔄 TrackNet - Pendiente
Requiere instalación adicional:
```bash
git clone https://github.com/Chang-Chia-Chi/TrackNetv2
cd TrackNetv2 && pip install -r requirements.txt
```

---

## Recomendación para MVP

### Opción A: HSV Mejorado (Recomendada)
**Tiempo estimado**: 1-2 días

Implementar detección HSV con:
- Filtro de circularidad estricto
- Tracking con Kalman filter
- Validación por movimiento entre frames
- Exclusión de zonas de línea de cancha

**Criterio de éxito**: Reducir falsos positivos a < 20% manteniendo > 60% de detección real

### Opción B: TrackNet
**Tiempo estimado**: 2-3 días

Instalar y configurar TrackNetV2:
- Clonar repositorio
- Descargar pesos pre-entrenados de tenis
- Adaptar para tamaño de frame del video
- Probar con videos de pádel

**Riesgo**: TrackNet está entrenado para tenis, puede no generalizar bien a pádel

### Opción C: Dataset Personalizado (Post-MVP)
**Tiempo estimado**: 1-2 semanas

1. Etiquetar 500-1000 frames de pádel con posición de pelota
2. Fine-tuning de YOLOv8 para pelota de pádel
3. Evaluar precisión

---

## Próximos Pasos Inmediatos

1. **Implementar HSV mejorado** (Opción A)
   - Crear `spike3_ball_hsv_improved.py`
   - Agregar filtros de forma y movimiento
   - Probar con videos disponibles

2. **Evaluar resultados**
   - Medir precisión con validación manual
   - Comparar con baseline actual

3. **Documentar decisión final**
   - Actualizar PLANNING.md con método elegido

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `spike3_ball_yolo.py` | Script de prueba YOLO vs HSV |
| `spikes/spikes/runs/spike3_ball_yolo/ball_detection_yolo.mp4` | Video con detecciones |
| `spikes/spikes/runs/spike3_ball_yolo/metrics.json` | Métricas de detección |

---

**Spike 3 - ESTADO**: ❌ **FRACASO PARCIAL**

**Decisión**: No se continuará con la detección de pelota en el MVP. Se retoma en versión futura con:
- Dataset etiquetado específico para pádel
- Fine-tuning de modelo dedicado

*Última actualización: 28 de Febrero 2026*
