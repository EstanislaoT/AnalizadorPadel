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
- Almacenamiento local en rutas runtime fuera del código fuente

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
│   │   └── ...                           # Configuración, servicios y modelos
│   └── tests/AnalizadorPadel.Api.Tests/  # Tests BDD (SpecFlow) + TDD
├── frontend/
│   ├── src/
│   │   ├── features/                     # Pantallas y lógica por dominio
│   │   ├── shared/                       # Componentes y servicios compartidos
│   │   └── test/                         # Mocks y setup de tests
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
├── ml-models/                            # Modelos y pesos de ML
├── test-videos/                          # Assets para pruebas manuales y test data
├── var/                                  # Estado mutable de ejecución
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
│    Database      │          │   (/var/*)       │
└──────────────────┘          └──────────────────┘
```

---

## ✅ Validación y Cambios Incrementales

- Validaciones de integración entre componentes deben hacerse con una comprobación real del flujo completo, no solo con builds, tests unitarios o checks aislados por servicio.
- Antes de concluir que un entorno, puerto o dependencia está bloqueado, intentar la ejecución real del flujo esperado y validar el resultado observable.
- Cuando se hagan cambios estructurales o refactors grandes, avanzar en fases pequeñas y validar después de cada una con las verificaciones principales del proyecto.
- No incluir en commits artefactos generados de build o tooling salvo que exista una razón explícita para versionarlos.

## 📌 Guías por Capa

- `frontend/AGENTS.md`: validación UI, estructura `features/shared/test`, integración con API y convenciones de desarrollo frontend.
- `backend/AGENTS.md`: endpoints, modelos, persistencia, rutas runtime, CORS y validación del backend.

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
```

# Desarrollo local
Ver AGENTS.md de cada capa para comandos y validaciones específicas


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

## 📞 Información de Contacto y Contribución

- Ver `CONTRIBUTING.md` para guías de contribución
- Ver `docs/ADR/README.md` para decisiones arquitectónicas
- Issues conocidos: `docs/KNOWN_ISSUES.md`

---

*Este documento es el punto de entrada principal para entender el proyecto. Para detalles específicos, consultar la documentación enlazada.*
