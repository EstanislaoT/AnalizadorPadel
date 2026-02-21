# Spike 1 - Resultados de Validación de YOLO en Videos de Pádel

**Fecha**: 19-21 Febrero 2026  
**Objetivo**: Confirmar que YOLO v8 detecta correctamente los 4 jugadores en videos reales de pádel con cámara cenital.

## Resumen Ejecutivo

**Resultado Final**: ⚠️ **PARCIALMENTE EXITOSO**

| Métrica | Sin Filtrado | Con Filtrado Final |
|---------|-------------|-------------------|
| Frames con 4 jugadores | 1.6% | **39.0%** |
| Frames con 3-5 jugadores | ~20% | **93.0%** |
| Promedio jugadores/frame | 6.4 | 3.32 |
| Mejora | - | **24x mejor** |

---

## Enfoques Probados

### Comparativa de Resultados

| Enfoque | Frames con 4 | Frames con 3-5 | Problema |
|---------|--------------|----------------|----------|
| Sin filtrado | 1.6% | ~20% | Detecta 6-11 personas |
| Zona fija original | 3.0% | 20.0% | Zona no coincide con video |
| Color de piso (OpenCV) | 0.0% | 0.0% | Solo detectó un lado de la cancha |
| Detección de líneas (Hough) | 37.0% | 94.0% | Rectángulo cubría solo mitad derecha |
| Perspectiva automática | 0.0% | 0.0% | Polígono muy pequeño |
| YOLO segmentación | - | - | Solo detecta personas, no cancha |
| **Zona + Confianza >= 0.5** | **39.0%** | **93.0%** | Mejor resultado |

---

## Solución Final Implementada

```python
# Parámetros optimizados
margin_x = int(width * 0.12)   # 12% margen lateral
margin_y = int(height * 0.08)  # 8% margen vertical
conf_threshold = 0.5           # Confianza mínima

# Filtrar detecciones
filtered = [d for d in detections 
            if in_court(d) and d['confidence'] >= 0.5]

# Limitar a 4 detecciones con mayor confianza
if len(filtered) > 4:
    filtered = sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:4]
```

### Zona de Cancha Final
- TL: (153, 57)
- TR: (1127, 57)
- BR: (1127, 663)
- BL: (153, 663)

---

## Análisis de Errores

### Causa de detecciones extras (sin filtrado)
1. **4 jugadores de pádel** ✓ (objetivo)
2. **Árbitro de silla** - dentro/fuera de cancha
3. **Espectadores** - visibles en bordes del frame
4. **Personal de producción** - cámaras, operadores
5. **Falsos positivos de YOLO** - confianza baja (<0.4)

### Causa de falsos negativos (61% frames sin exactamente 4)
1. **YOLO no detecta al jugador** (principal causa)
   - Oclusiones parciales
   - Poses difíciles
   - Iluminación variable
2. **Jugador fuera de zona** (marginal)
3. **Confianza baja** (<0.5)

---

## Lecciones Aprendidas

### Lo que funcionó
- ✅ Filtro por zona rectangular simple
- ✅ Filtro de confianza >= 0.5
- ✅ Limitar a máximo 4 detecciones

### Lo que NO funcionó
- ❌ Detección automática de piso por color
- ❌ Detección automática de líneas (Hough)
- ❌ YOLO segmentación para detectar cancha
- ❌ Polígonos de perspectiva automática

### Limitaciones de YOLOv8n
- ~7% de falsos negativos en detección de jugadores
- Falsos positivos con baja confianza (filtrables)
- FPS: 51 (suficiente para tiempo real)

---

## Próximos Pasos

### Mejoras propuestas
1. **Probar YOLOv8 medium** - Mayor precisión, menor velocidad
2. **Tracking temporal** - Usar posición en frame anterior para predecir posición actual
3. **Detección de cancha manual** - Permitir al usuario marcar los 4 vértices
4. **Combinar con detección de pelota** - Validar que jugadores estén cerca de la acción

### Para el MVP
**Recomendación**: Proceder con el filtrado actual (zona + confianza), aceptando que:
- 39% de frames tendrán exactamente 4 jugadores
- 93% de frames tendrán entre 3-5 jugadores
- Se requerirá tracking temporal para mejorar precisión

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `spike1_yolo_validation.py` | Validación inicial sin filtrado |
| `spike1_video_generator.py` | Generador de video con detecciones |
| `spike1_detection_analysis.py` | Análisis por zona de cancha |
| `spike1_detailed_frame.py` | Análisis detallado de un frame |
| `spike1_court_floor_detection.py` | Detección de piso por color |
| `spike1_court_lines_detection.py` | Detección de líneas (mejor resultado individual) |
| `spike1_court_perspective.py` | Detección de perspectiva |
| `spike1_yolo_court_segmentation.py` | Segmentación YOLO |
| `spike1_final_filter.py` | **Enfoque final recomendado** |

---

## Conclusión

**Spike 1 - ESTADO**: ⚠️ **PARCIALMENTE EXITOSO**

- ✅ Filtrado mejora de 1.6% → 39% (24x mejor)
- ✅ 93% de frames tienen 3-5 jugadores
- ⚠️ Limitación principal: YOLO tiene ~7% falsos negativos
- ⚠️ Detección automática de cancha no funciona bien

**Recomendación**: 
1. Proceder con filtrado zona + confianza para MVP
2. Implementar tracking temporal como siguiente mejora
3. Considerar permitir definición manual de zona de cancha