# 🔬 Spikes de Validación Técnica

Este directorio contiene los spikes (experimentos técnicos) realizados para validar riesgos antes de la implementación del MVP.

## Estructura

```
spikes/
├── court_calibration/    # Calibración de cancha y detección de esquinas
├── spike1/              # Validación de YOLO para detectar jugadores
├── spike2/              # Tracking de trayectorias y métricas de movimiento
└── spike3/              # Detección de pelota (fracaso parcial)
```

## Estado de los Spikes

| Spike | Estado | Resultado | Documentación |
|-------|--------|-----------|---------------|
| Court Calibration | ✅ Completado | Calibración manual funcionando | - |
| Spike 1 | ✅ Completado | YOLO detecta 4 jugadores | `docs/SPIKE-001-RESULTADOS.md` |
| Spike 2 | ✅ Completado | Tracking y métricas funcionando | `docs/SPIKE-002-RESULTADOS.md` |
| Spike 3 | ❌ Fracaso parcial | Detección de pelota no viable | `docs/SPIKE-003-RESULTADOS.md` |
| Spike 4 | ✅ Completado | Integración .NET→Python viable | `docs/SPIKE-004-RESULTADOS.md` |

## Descripción de cada Spike

### Court Calibration
Scripts para calibrar la cancha y detectar las 4 esquinas.
- `spike_court_calibration.py` - Versión inicial
- `spike_court_calibration_v2.py` - Versión mejorada con puntos de red

### Spike 1 - Validación de YOLO
Objetivo: Verificar que YOLO v8 detecta correctamente los 4 jugadores en videos de pádel con cámara cenital.

**Resultado**: ✅ Detección > 95% de frames con 4 jugadores correctamente identificados.

### Spike 2 - Tracking de Trayectorias
Objetivo: Extraer trayectorias de cada jugador y calcular métricas de movimiento (velocidad, distancia, heatmaps).

**Resultado**: ✅ Sistema completo de análisis de movimiento implementado.

### Spike 3 - Detección de Pelota
Objetivo: Detectar la pelota de pádel usando diferentes métodos.

**Resultado**: ❌ Fracaso parcial
- YOLO (sports ball): 0% detección
- HSV (color): 100% detección pero ~30% precisión (muchos falsos positivos)

**Decisión**: Postergado para versión futura. Se requiere dataset etiquetado específico para pádel.

## Ejecutar los Spikes

```bash
# Spike 1 - Detectar jugadores
cd spikes/spike1
python3 spike1_tracking.py

# Spike 2 - Análisis de movimiento
cd spikes/spike2
python3 spike2_trajectory.py

# Spike 3 - Detección de pelota
cd spikes/spike3
python3 spike3_ball_yolo.py
```

## Directorios Relacionados

- `docs/` - Documentación de resultados y planning
- `runs/` - Outputs generados por los scripts
- `test-videos/` - Videos de prueba