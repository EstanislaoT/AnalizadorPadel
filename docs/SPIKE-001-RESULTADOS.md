# Spike 1 - Resultados de Validación de YOLO en Videos de Pádel

**Fecha**: 19-21 Febrero 2026  
**Objetivo**: Confirmar que YOLO v8 detecta correctamente los 4 jugadores en videos reales de pádel con cámara cenital.

## Resumen Ejecutivo

| Métrica | Sin Filtrado | Con Filtrado por Zona | Con Filtrado + Confianza |
|---------|-------------|----------------------|-------------------------|
| Personas detectadas/frame | 6-11 | 6-7 | 4-5 (estimado) |
| Frames con 4 jugadores | 1.6% | 3.0% | Por validar |
| FPS de procesamiento | 51 | 51 | 51 |
| Confianza promedio | 0.53 | 0.53 | >0.5 |

**Veredicto**: ⚠️ PARCIALMENTE EXITOSO - Requiere filtrado adicional

---

## Análisis Detallado

### 1. Detección Sin Filtrado

Resultados del análisis de 30 segundos (900 frames):

```
📈 Distribución de detecciones:
   1 personas:  164 frames ( 18.2%)
   2 personas:   42 frames (  4.7%)
   3 personas:   21 frames (  2.3%)
   4 personas:   14 frames (  1.6%) ← Solo 1.6% con 4 exactas
   5 personas:   57 frames (  6.3%)
   6 personas:  134 frames ( 14.9%)
   7 personas:  222 frames ( 24.7%) ← Más común
   8 personas:  156 frames ( 17.3%)
   9 personas:   66 frames (  7.3%)
   10 personas:  22 frames (  2.4%)
   11 personas:   2 frames (  0.2%)
```

**Causa identificada**: El video de prueba incluye:
- 4 jugadores de pádel
- Árbitro de silla
- Espectadores visibles
- Personal de producción/cámaras

---

### 2. Detección con Filtrado por Zona

Se definió una zona de cancha centrada:
- Margen lateral: 15% (192px - 1088px de 1280px)
- Margen vertical: 12% (86px - 634px de 720px)

Resultados del análisis de 100 frames:

```
📍 Clasificación por zona:
   Detecciones DENTRO de cancha: 642 (96.1%)
   Detecciones FUERA de cancha: 26 (3.9%)

📈 Promedio por frame:
   Dentro de cancha: 6.4 personas/frame
   Fuera de cancha: 0.3 personas/frame

🎯 Resultados del filtrado:
   Frames con 4 personas DENTRO: 3/100 (3.0%)
   Frames con 3-5 personas DENTRO: 20/100 (20.0%)
```

**Hallazgo crítico**: La mayoría de las detecciones extras están DENTRO de la cancha, no fuera.

---

### 3. Análisis Frame por Frame

Se analizó el Frame 0 en detalle para identificar cada detección:

| # | Centro (x, y) | Confianza | Ubicación | Interpretación |
|---|---------------|-----------|-----------|----------------|
| 1 | (478, 420) | 0.804 | Centro-Centro | ✅ Jugador (alta confianza) |
| 2 | (772, 508) | 0.773 | Centro-Centro | ✅ Jugador (alta confianza) |
| 3 | (729, 260) | 0.673 | Centro-Centro | ✅ Jugador (media confianza) |
| 4 | (509, 258) | 0.577 | Centro-Centro | ✅ Jugador (media confianza) |
| 5 | (937, 202) | 0.429 | Centro-Centro | ❓ Posible falso positivo |
| 6 | (831, 249) | 0.282 | Centro-Centro | ❓ Probable falso positivo |

**Conclusión**: Las detecciones #5 y #6 tienen baja confianza y podrían ser falsos positivos.

---

## Solución Propuesta

### Filtrado por Confianza Mínima

Los 4 jugadores típicamente tienen confianza >0.5. Implementar:

```python
# Filtrar detecciones con confianza >= 0.5
filtered_detections = [d for d in detections if d['confidence'] >= 0.5]

# Si hay más de 4, tomar las 4 con mayor confianza
if len(filtered_detections) > 4:
    filtered_detections = sorted(filtered_detections, 
                                  key=lambda x: x['confidence'], 
                                  reverse=True)[:4]
```

### Validación Pendiente

Se requiere ejecutar un nuevo análisis con:
1. Filtrado por zona de cancha
2. Filtro de confianza mínima (conf >= 0.5)
3. Limitar a máximo 4 detecciones por frame

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `runs/detect/spike1/detection_output.mp4` | Video 10s con bounding boxes |
| `runs/analysis/frame_0000.jpg` | Frame con zona de cancha marcada |
| `runs/analysis/frame_0_detailed.jpg` | Frame con cada detección numerada |
| `runs/analysis/report.json` | Estadísticas del análisis |

## Scripts de Análisis

| Script | Propósito |
|--------|-----------|
| `spike1_yolo_validation.py` | Validación inicial sin filtrado |
| `spike1_video_generator.py` | Generador de video con detecciones |
| `spike1_detection_analysis.py` | Análisis por zona de cancha |
| `spike1_detailed_frame.py` | Análisis detallado de un frame |

---

## Próximos Pasos

1. **Validar filtrado por confianza**: Ejecutar análisis con umbral 0.5
2. **Probar con YOLO medium**: Comparar precisión vs velocidad
3. **Implementar detección de líneas**: Automatizar definición de zona de cancha
4. **Documentar ADR**: Crear ADR-004 con decisión sobre modelo de detección

---

## Conclusión

**Spike 1 - ESTADO**: ⚠️ **PARCIALMENTE EXITOSO**

- ✅ YOLO detecta personas correctamente (fps: 51, confianza media: 0.53)
- ✅ Los 4 jugadores se detectan con alta confianza (>0.5)
- ⚠️ Se detectan 2-7 personas adicionales por frame
- ⚠️ Requiere filtrado combinado: zona de cancha + confianza mínima
- ❌ Sin filtrado, solo 1.6% de frames tienen exactamente 4 detecciones

**Recomendación**: Proceder con implementación de filtrado doble (zona + confianza) y validar resultados.