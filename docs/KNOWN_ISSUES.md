# 🐛 Errores Conocidos / Known Issues

Este documento lista los errores y limitaciones conocidas del proyecto.

---

## Spike Court Calibration v2

### Issue #1: Cálculo de vértices desde puntos de red

- **Archivo**: `spike_court_calibration_v2.py`
- **Función afectada**: `_update_corner_with_homography()`
- **Descripción**: El cálculo de vértices a partir de los puntos de red tiene errores. Cuando se arrastra un punto de red, la esquina calculada mediante homografía puede generar posiciones incorrectas en algunos casos extremos.
- **Fecha detectado**: 22 de Febrero 2026
- **Prioridad**: Media
- **Estado**: Pendiente de investigación
- **Workaround**: Ajustar manualmente las esquinas de la cancha en lugar de usar los puntos de red

---

## Formato para nuevos issues

```markdown
### Issue #N: Título del issue

- **Archivo**: `ruta/al/archivo.py`
- **Función afectada**: `nombre_funcion()`
- **Descripción**: Descripción detallada del problema
- **Fecha detectado**: DD de Mes AAAA
- **Prioridad**: Alta/Media/Baja
- **Estado**: Pendiente/En investigación/Resuelto
- **Workaround**: Solución temporal si existe
```
