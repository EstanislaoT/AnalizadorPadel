# ADR-004: Estructura del Proyecto

## Estado

Propuesto

## Contexto

Se necesita definir la estructura de carpetas y organización del proyecto para el desarrollo del Analizador de Pádel. El proyecto tendrá múltiples componentes (backend, frontend, scripts Python) y necesitamos una estructura que facilite el desarrollo, mantenimiento y escalabilidad.

### Restricciones

- El proyecto incluye backend (.NET), frontend (React), y scripts (Python)
- Necesita soporte para Docker y desarrollo local
- Los spikes actuales están en carpeta `spikes/` y no deben mezclarse con código de producción

## Decisión

Se adopta la estructura de **Monorepo con carpetas separadas**:

```
AnalizadorPadel/
├── backend/                 # API .NET (Nuxt/ASP.NET Core)
├── frontend/                # React app
├── scripts/                 # Python processing (spikes evolucionados)
├── ml-models/               # Modelos ML (YOLO)
├── infrastructure/          # Docker, configs
├── docs/                    # Documentación
├── tests/                   # Tests de integración
├── SPEC.md                  # Especificación del producto
└── README.md
```

### Detalle de Carpetas

#### backend/
```
backend/
├── src/
│   └── AnalizadorPadel.Api/
│       ├── Controllers/
│       ├── Services/
│       ├── Models/
│       │   ├── Entities/
│       │   └── DTOs/
│       ├── Data/
│       ├── Middleware/
│       ├── Configuration/
│       └── Program.cs
├── tests/
│   └── AnalizadorPadel.Api.Tests/
├── AnalizadorPadel.sln
└── *.csproj
```

#### frontend/
```
frontend/
├── src/
│   ├── components/
│   ├── pages/
│   ├── services/
│   ├── hooks/
│   ├── store/
│   ├── types/
│   └── App.tsx
├── public/
├── package.json
├── vite.config.ts
└── tsconfig.json
```

#### scripts/
```
scripts/
├── process_video.py
├── requirements.txt
└── README.md
```

#### ml-models/
```
ml-models/
└── yolov8m.pt
```

#### infrastructure/
```
infrastructure/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── Dockerfile.python
└── nginx.conf
```

## Consecuencias

### Positivas
- Estructura clara y estándar de la industria
- Desarrollo paralelo de backend y frontend
- Fácil de configurar CI/CD
- Separa código de spikes de producción
- Escalable para agregar más servicios

### Negativas
- Requiere configurar múltiples proyectos en IDE
- Path resolution entre carpetas ( ../ml-models )

### Neutrales
- Necesita disciplina para mantener estructura
- Los scripts Python deben evolucionar de spikes a producción

## Alternativas Consideradas

| Alternativa | Pros | Contras | ¿Por qué se descarta? |
|-------------|------|---------|----------------------|
| Todo en un repo .NET | Todo en un lugar | Frontend dentro de .NET no es estándar | Complica mantenimiento |
| Repos separados (backend, frontend, python) | Totalmente independientes | Gestión de múltiples repos | Más complejo de operar |
| Estructura plana (todo en root) | Simple | Se mezcla spikes con producción | Falta organización |

## Referencias

- [PLANNING.md - Configuración de Desarrollo](../PLANNING.md)
- [TECHNICAL.md - Stack Tecnológico](../TECHNICAL.md)

---

*Fecha: 2026-02-28*
*Autores: Equipo de Arquitectura*
