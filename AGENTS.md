# 🎾 Analizador de Pádel - Contexto del Proyecto

> **Versión**: 1.1 MVP Simplificado  
> **Última actualización**: Marzo 2026  
> **Propósito**: Este documento proporciona el contexto completo del proyecto para asistentes de IA, permitiendo entender rápidamente la arquitectura, funcionalidades y decisiones técnicas.

---

## 📋 Descripción General

**Analizador de Pádel** es una aplicación web para análisis de partidos de pádel mediante procesamiento de video. Los usuarios suben videos de sus partidos y obtienen estadísticas detalladas, análisis de movimiento y visualizaciones de su juego.

**Público objetivo**: Jugadores de pádel amateur y semi-profesional que quieran mejorar su juego con datos objetivos.

---

## 🎯 Funcionalidades Principales

### 1. Subida de Videos
- Interfaz para subir videos (MP4, AVI, MOV)
- Límite de tamaño: 500MB
- Validación de formato y duración mínima (1 minuto)
- Almacenamiento local en `/uploads`

### 2. Procesamiento de Video
- Detección de jugadores usando **YOLO v8** vía Python subprocess
- Seguimiento de posiciones en la cancha
- Extracción de estadísticas de detección
- Procesamiento síncrono con timeout de 10 minutos

### 3. Análisis y Estadísticas
- Tiempo total del partido
- Número de frames analizados
- Tasa de detección de jugadores
- Heatmap de movimiento (placeholder en MVP)
- Estadísticas de procesamiento

### 4. Dashboard y Visualización
- Dashboard principal con estadísticas generales
- Lista de videos recientes
- Lista de análisis recientes
- Estados visuales (uploaded, processing, completed, failed)

---

## 🏗️ Arquitectura del Sistema

### Stack Tecnológico

| Capa | Tecnología | Versión |
|------|------------|---------|
| **Backend** | ASP.NET Core | 10 (LTS) |
| **Frontend** | React + TypeScript | 18 |
| **Base de Datos** | SQLite | - |
| **ML/Computer Vision** | Python + YOLO v8 | 3.10+ |
| **UI Components** | Material-UI (MUI) | v5 |
| **State Management** | Zustand | - |
| **Build Tool** | Vite | - |
| **Contenedores** | Docker + Docker Compose | - |
| **Web Server** | Nginx | - |

### Estructura del Proyecto

```
AnalizadorPadel/
├── backend/
│   ├── src/AnalizadorPadel.Api/          # API .NET con Minimal APIs
│   │   ├── Program.cs                    # Endpoints y configuración
│   │   ├── Data/PadelDbContext.cs        # EF Core + SQLite
│   │   ├── Models/Entities/              # VideoEntity, AnalysisEntity
│   │   ├── Models/DTOs/                  # ApiDtos.cs
│   │   └── Services/                     # VideoService, AnalysisService
│   └── tests/AnalizadorPadel.Api.Tests/  # Tests BDD (SpecFlow) + TDD
├── frontend/
│   ├── src/
│   │   ├── components/                   # Layout, VideoPlayer
│   │   ├── pages/                        # Dashboard, Videos, Analyses, Reports
│   │   ├── services/api/generated/       # Cliente TypeScript generado
│   │   └── store/                        # Zustand stores
│   └── package.json
├── python-scripts/
│   ├── process_video.py                  # Script YOLO para detección
│   └── tests/                            # Tests de Python
├── infrastructure/
│   ├── Dockerfile.api                    # Backend + Python + FFmpeg
│   ├── Dockerfile.frontend               # Nginx + React build
│   └── nginx.conf                        # Configuración reverse proxy
├── docker-compose.yml                    # Orquestación de servicios
├── docs/                                 # Documentación (ADR, PRODUCT, TECHNICAL)
└── spikes/                               # Experimentos técnicos
```

### Diagrama de Arquitectura MVP

```
         Internet (puerto 80)
                 │
                 ▼
     ┌───────────────────────┐
     │        Nginx          │
     │   (Reverse Proxy)     │
     └───────────┬───────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
       ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│ React (estático) │  │  Backend Kestrel │
│ /usr/share/nginx │  │  (.NET :5000)    │
└──────────────────┘  └────────┬─────────┘
                               │
           ┌───────────────────┴──────────────┐
           │                              │
           ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│    SQLite        │          │  Local Storage   │
│    Database      │          │   (/uploads)     │
└──────────────────┘          └──────────────────┘
```

---

## 🔌 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/videos` | Subir nuevo video |
| GET | `/api/videos` | Listar videos |
| GET | `/api/videos/{id}` | Obtener video por ID |
| GET | `/api/videos/{id}/stream` | Stream de video (Range requests) |
| DELETE | `/api/videos/{id}` | Eliminar video |
| POST | `/api/videos/{id}/analyse` | Iniciar análisis del video |
| GET | `/api/analyses` | Listar análisis |
| GET | `/api/analyses/{id}` | Obtener análisis por ID |
| GET | `/api/analyses/{id}/stats` | Estadísticas del análisis |
| GET | `/api/analyses/{id}/heatmap` | Datos del heatmap |
| GET | `/api/analyses/{id}/report` | Reporte del análisis |
| GET | `/api/dashboard/stats` | Estadísticas del dashboard |
| GET | `/api/health` | Health check |

---

## 🗄️ Modelo de Datos

### Entidades Principales

```csharp
// VideoEntity
public class VideoEntity
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string? Description { get; set; }
    public string FilePath { get; set; }
    public long FileSizeBytes { get; set; }
    public string FileExtension { get; set; }
    public string Status { get; set; }  // Uploaded, Processing, Completed, Failed
    public DateTime UploadedAt { get; set; }
    public int? AnalysisId { get; set; }
}

// AnalysisEntity
public class AnalysisEntity
{
    public int Id { get; set; }
    public int VideoId { get; set; }
    public string Status { get; set; }  // Pending, Running, Completed, Failed
    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public string? ErrorMessage { get; set; }
    
    // Resultados del análisis
    public int? TotalFrames { get; set; }
    public int? PlayersDetected { get; set; }
    public double? AvgDetectionsPerFrame { get; set; }
    public int? FramesWith4Players { get; set; }
    public double? DetectionRatePercent { get; set; }
    public double? ProcessingTimeSeconds { get; set; }
    public string? ModelUsed { get; set; }
}
```

---

## 🎥 Pipeline de Procesamiento de Video

```
Video (.mp4)
    │
    ▼
Backend .NET recibe el video
    │
    ▼
Guarda en /uploads + registro en SQLite
    │
    ▼
Llama a Python subprocess
    │
    ▼
YOLO v8 procesa el video
    ├── Detecta personas (clase 0)
    ├── Cuenta detecciones por frame
    └── Calcula métricas
    │
    ▼
Resultados JSON guardados en /outputs
    │
    ▼
Backend actualiza AnalysisEntity en SQLite
    │
    ▼
Frontend consulta y visualiza resultados
```

### Parámetros de Procesamiento

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Modelo YOLO | yolov8m.pt | Balance precisión/velocidad |
| Clases detectadas | [0] (personas) | Solo jugadores |
| Timeout | 10 minutos | Límite razonable para partido |
| Máximo tamaño | 500MB | Evita videos muy grandes |

---

## 🧪 Estrategia de Testing

### Enfoque Híbrido: BDD + TDD

| Tipo | Herramienta | Uso |
|------|-------------|-----|
| **BDD** | SpecFlow | Features de usuario (User Stories) |
| **TDD** | xUnit + FluentAssertions | Lógica de negocio |
| **Mocking** | Moq | Dependencias en tests unitarios |

### User Stories Implementadas

1. **US-1**: Subir un Video de Partido
2. **US-2**: Ver Estadísticas del Partido

### Estructura de Tests

```
backend/tests/AnalizadorPadel.Api.Tests/
├── BDD/
│   ├── Features/              # Archivos .feature (Gherkin)
│   │   ├── US-1-SubirVideo.feature
│   │   └── US-2-VerEstadisticas.feature
│   └── StepDefinitions/       # Implementación de steps
├── Integration/               # Tests de integración
│   ├── VideoEndpointsTests.cs
│   └── AnalysisEndpointsTests.cs
└── Unit/                      # Tests unitarios
    └── Services/
```

---

## 🐳 Despliegue con Docker

### Servicios

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `api` | 5001 (externo) / 5000 (interno) | Backend .NET + Python + FFmpeg |
| `frontend` | 80 | Nginx con React SPA |

### Volúmenes

- `padel-data`: Base de datos SQLite
- `padel-uploads`: Videos subidos
- `padel-outputs`: Resultados de análisis
- `padel-logs`: Logs de la aplicación

### Comandos

```bash
# Iniciar en producción
docker-compose up -d

# Ver logs
docker-compose logs -f

# Desarrollo local (sin Docker)
cd backend/src/AnalizadorPadel.Api && dotnet run  # :5000
cd frontend && npm run dev                         # :5173
```

---

## 📚 Documentación Adicional

| Documento | Ubicación | Descripción |
|-----------|-----------|-------------|
| **Producto** | `docs/PRODUCT.md` | Funcionalidades, casos de uso, roadmap |
| **Técnico** | `docs/TECHNICAL.md` | Stack, arquitectura detallada, módulos |
| **Planificación** | `docs/PLANNING.md` | Fases, setup, decisiones y métricas |
| **ADRs** | `docs/ADR/` | Architecture Decision Records |
| **Spikes** | `docs/SPIKE-*.md` | Resultados de experimentos técnicos |

---

## ⚠️ Limitaciones Conocidas del MVP

1. **Detección de pelota**: No implementada en MVP (requiere TrackNet)
2. **Procesamiento síncrono**: El usuario espera en la misma request
3. **Sin autenticación**: Cualquiera puede acceder (agregado en V2.0)
4. **Almacenamiento local**: SQLite + filesystem (migrar a PostgreSQL/S3 en V2.0)
5. **Heatmap**: Datos de placeholder (generación real post-MVP)
6. **Ángulo de cámara**: Optimizado para vista cenital estándar

---

## 🔮 Roadmap

### V1.0 — MVP (Actual)
- ✅ Subida y almacenamiento local de videos
- ✅ Procesamiento síncrono básico
- ✅ Detección de jugadores con YOLO v8
- ✅ Estadísticas básicas
- ❌ Sin autenticación

### V2.0 — Post-MVP
- Autenticación de usuarios (JWT)
- Procesamiento asíncrono con background workers
- Almacenamiento en S3
- Detección de pelota mejorada (TrackNet)
- Análisis de técnica de golpeo
- Comparación entre partidos

### V3.0 — Futuro
- IA para recomendaciones personalizadas
- Coach virtual
- Torneos virtuales
- Comunidad y rankings
- Integración con wearables

---

## 🔧 Comandos Útiles para Desarrollo

```bash
# Backend
cd backend/src/AnalizadorPadel.Api
dotnet run                    # Iniciar API
dotnet test                   # Ejecutar tests
dotnet ef migrations add <name>  # Crear migración

# Frontend
cd frontend
npm install                   # Instalar dependencias
npm run dev                   # Iniciar dev server
npm run test                  # Ejecutar tests
npm run codegen               # Generar tipos desde OpenAPI

# Docker
docker-compose up -d          # Iniciar todos los servicios
docker-compose logs -f api    # Ver logs del backend
docker-compose down -v        # Detener y eliminar volúmenes

# Tests E2E
cd e2e
npm install
npx playwright test           # Ejecutar tests E2E
```

---

## 📞 Información de Contacto y Contribución

- Ver `CONTRIBUTING.md` para guías de contribución
- Ver `docs/ADR/README.md` para decisiones arquitectónicas
- Issues conocidos: `docs/KNOWN_ISSUES.md`

---

*Este documento es el punto de entrada principal para entender el proyecto. Para detalles específicos, consultar la documentación enlazada.*
