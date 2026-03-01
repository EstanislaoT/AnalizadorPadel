# ADR-005: Reorganización de Carpeta runs/

## Estado

Aprobado

## Contexto

Durante una revisión de la estructura del proyecto, se identificó que la carpeta `runs/` se encuentra en la raíz del proyecto, pero contiene exclusivamente resultados de ejecuciones de spikes (spike2_, spike3_, etc.). Esto crea una inconsistencia estructural con lo definido en ADR-004.

### Problema

- La carpeta `runs/` está en la raíz pero su contenido está relacionado con spikes
- Los nombres de las carpetas dentro de `runs/` sugieren su origen: `spike2_`, `spike3_`, etc.
- La estructura actual no sigue el principio de cohesión definido en ADR-004
- Los resultados de ejecuciones deberían estar cerca del código que las genera

### Restricciones

- No se debe perder ningún resultado de ejecución existente
- Los scripts que hacen referencia a estas rutas deben ser actualizados
- La estructura debe ser coherente con ADR-004

## Decisión

**Mover la carpeta `runs/` dentro de `spikes/`** para mantener la coherencia estructural.

### Nueva Estructura

```
AnalizadorPadel/
├── backend/
├── frontend/
├── scripts/
├── models/
├── infrastructure/
├── docs/
├── tests/
├── spikes/
│   ├── README.md
│   ├── spike1/
│   ├── spike2/
│   ├── spike3/
│   ├── spike4/
│   └── runs/              # ← MOVIDO AQUÍ
│       ├── spike2/
│       ├── spike3/
│       └── examples/
└── README.md
```

### Detalle de la Reorganización

#### Contenido actual de `runs/`:
```
runs/
├── court_amateur1/
├── court_finalpadel/
├── court_propadel2/
├── examples/
├── spike2/
├── spike2_amateur1/
├── spike2_finalpadel/
├── spike2_propadel2/
├── spike3_ball/
└── spike3_ball_yolo/
```

#### Nueva ubicación: `spikes/runs/`

#### Referencias a actualizar:
- Scripts Python que escriben en `runs/`
- Documentación que mencione la ruta `runs/`
- Configuraciones que apunten a `runs/`

## Consecuencias

### Positivas
- Coherencia estructural con ADR-004
- Los resultados de spikes están cerca del código que los genera
- Estructura más intuitiva para nuevos desarrolladores
- Facilita la limpieza y mantenimiento de resultados de spikes

### Negativas
- Se deben actualizar las referencias a las rutas
- Scripts existentes necesitan modificación
- Cambio de ubicación requiere comunicación al equipo

### Neutrales
- La carpeta `spikes/` crece en tamaño
- Se necesita mantener disciplina para no mezclar resultados con código

## Implementación

### Pasos a ejecutar:

1. **Backup**: Verificar contenido actual de `runs/`
2. **Mover**: `mv runs/ spikes/runs/`
3. **Actualizar referencias** en:
   - Scripts Python (`spikes/*/`)
   - Documentación (`docs/`)
   - Configuraciones
4. **Verificar**: Ejecutar algunos spikes para confirmar funcionamiento
5. **Documentar**: Actualizar README.md y documentación relevante

### Comandos de ejecución:

```bash
# 1. Verificar contenido
ls -la runs/

# 2. Mover carpeta
mv runs/ spikes/runs/

# 3. Buscar referencias a actualizar
grep -r "runs/" spikes/ --include="*.py"
grep -r "runs/" docs/ --include="*.md"

# 4. Actualizar referencias (según resultados del grep)
```

## Alternativas Consideradas

| Alternativa | Pros | Contras | ¿Por qué se descarta? |
|-------------|------|---------|----------------------|
| Mantener `runs/` en raíz | No hay cambios | Inconsistencia estructural | Va contra ADR-004 |
| Crear `results/` en raíz | Nombre más genérico | Mismo problema de ubicación | No resuelve cohesión |
| Eliminar `runs/` | Limpieza total | Pérdida de resultados históricos | Información valiosa |
| **Mover a `spikes/runs/`** | **Cohesión estructural** | **Requiere actualizar referencias** | **Mejor solución** |

## Referencias

- [ADR-004: Estructura del Proyecto](ADR-004-estructura-del-proyecto.md)
- [PLANNING.md - Configuración de Desarrollo](../PLANNING.md)
- Estructura actual del proyecto

---

*Fecha: 2026-03-01*
*Autores: Equipo de Arquitectura*
*Estado: Aprobado para implementación*