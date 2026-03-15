# 🗺️ Planificación — Analizador de Pádel

## 🚀 Plan de Implementación

### Fase 1: Setup Básico (Semanas 1-2)
- [x] Configurar estructura del proyecto
- [x] **Diseñar especificación OpenAPI (API First)**
- [x] **Configurar Scalar (Swagger) y generar documentación**
- [ ] **Configurar mock server con Prism para frontend** (NO IMPLEMENTADO)
- [x] Setup de Docker y docker-compose
- [x] Configurar base de datos SQLite
- [x] Crear API básica de videos (siguiendo contrato OpenAPI)
- [x] Setup de frontend React + TypeScript
- [x] Configurar CORS y endpoints básicos

### Fase 2: Subida y Almacenamiento (Semanas 3-4)
- [x] Implementar subida de videos
- [x] Configurar almacenamiento local (`/uploads`)
- [x] Crear API endpoints para videos
- [x] Implementar frontend de subida (input básico, sin drag & drop)
- [x] Implementar procesamiento síncrono

### Fase 3: Procesamiento de Video (Semanas 5-7)
- [x] FFmpeg disponible en Docker (instalado pero YOLO lo usa internamente)
- [x] Implementar scripts Python con YOLO v8 (procesamiento básico)
- [ ] Detección de pelota (OpenCV HSV) - EXCLUIDO DEL MVP (Spike 3)
- [x] Crear servicios modulares de análisis
- [x] Extraer y guardar estadísticas en BD

### Fase 4: Visualización y UI (Semanas 8-9)
- [ ] **Sprint 1: Estructura Base y Dashboard** (3 días)
  - [ ] Crear estructura de carpetas según ADR-004
  - [ ] Instalar dependencias (MUI, Chart.js, D3.js, Video.js, Zustand)
  - [ ] Crear layout principal con navegación
  - [ ] Implementar dashboard con estadísticas básicas
  - [ ] Crear endpoint `GET /api/dashboard/stats` en backend
- [ ] **Sprint 2: Reproductor de Video** (3 días)
  - [ ] Implementar endpoint stream de video en backend
  - [ ] Crear componente VideoPlayer con Video.js
  - [ ] Integrar reproductor con lista de videos
  - [ ] Agregar controles de reproducción
- [ ] **Sprint 3: Heatmap con D3.js** (3 días)
  - [ ] Crear componente Heatmap con D3.js
  - [ ] Implementar endpoint heatmap por jugador
  - [ ] Integrar heatmap con reproductor de video
  - [ ] Agregar controles de filtrado por jugador
- [ ] **Sprint 4: Visualizaciones Estadísticas** (3 días)
  - [ ] Crear componente StatisticsCharts con Chart.js
  - [ ] Implementar endpoint stats por jugador
  - [ ] Crear gráficos: detecciones, frames, tasa éxito
  - [ ] Integrar con dashboard
- [ ] **Sprint 5: Timeline y Frames** (2 días)
  - [ ] Crear componente Timeline con marcadores
  - [ ] Implementar endpoint timeline de eventos
  - [ ] Crear componente FrameViewer para frames específicos
  - [ ] Integrar timeline con reproductor
- [ ] **Sprint 6: Reportes PDF y Testing** (3 días)
  - [ ] Crear componente ReportDownloader
  - [ ] Implementar generación de PDF con QuestPDF
  - [ ] Crear tests BDD para US-2 (Ver Estadísticas)
  - [ ] Crear tests TDD para servicios de visualización

### Fase 5: Testing y Deploy (Semana 10)
- [ ] Testing unitario y de integración
- [ ] Optimización de rendimiento
- [ ] Configurar CI/CD (GitHub Actions)
- [ ] Deploy a producción
- [ ] Documentación final

---

## 🔧 Configuración de Desarrollo

### Prerrequisitos

| Herramienta | Versión | Notas |
|---|---|---|
| .NET SDK | 10.0 | Requerido para desarrollo backend local |
| Node.js | 18+ | Requerido para desarrollo frontend local |
| Docker | Latest | Obligatorio para producción |
| Docker Compose | Latest | Obligatorio para producción |
| FFmpeg | Latest | Solo si desarrollas sin Docker |
| Python | 3.10+ | Solo si desarrollas sin Docker |

### Puertos por Entorno

| Servicio | Desarrollo | Producción |
|---|---|---|
| Frontend | :5173 (Vite Dev Server) | :80 (Nginx) |
| Backend | :5000 (Kestrel directo) | :5000 (interno, via Nginx) |
| SQLite | :N/A (archivo local) | :N/A (archivo en volumen) |

### Setup Local (Desarrollo)

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd AnalizadorPadel

# 2. Iniciar base de datos
docker-compose up -d postgres

# 3. Backend (http://localhost:5000)
cd backend
dotnet restore
dotnet run

# 4. Frontend (http://localhost:5173)
cd frontend
npm install
npm run dev
```

### Setup con Docker (Producción)

```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# La app estará disponible en http://localhost
```

### Variables de Entorno

```env
# Backend
ConnectionStrings__DefaultConnection=Data Source=padel.db
UPLOADS_PATH=/app/uploads
PYTHON_SCRIPTS_PATH=/app/scripts
MAX_VIDEO_SIZE_MB=500
PROCESSING_TIMEOUT_MINUTES=10

# SQLite (MVP)
# No requiere configuración adicional - archivo local: padel.db
```

---

## 📝 Notas y Decisiones

### Decisiones Técnicas Tomadas

| # | Decisión | Alternativa Considerada | Justificación |
|---|---|---|---|
| 1 | React sobre Angular | Angular, Vue | Mayor ecosistema y flexibilidad |
| 2 | **SQLite** sobre PostgreSQL | PostgreSQL, MySQL | SQLite es más simple, no requiere servidor, ideal para MVP sin usuarios |
| 3 | .NET sobre Node.js | Node.js, Python FastAPI | Mejor rendimiento para procesamiento pesado |
| 4 | Docker | Bare metal | Facilita deploy y desarrollo consistente |
| 5 | Sin autenticación en MVP | JWT desde el inicio | Reduce complejidad, se agrega en V2.0 |
| 6 | Almacenamiento SQLite | AWS S3 | Archivo local (padel.db), migración a PostgreSQL/S3 en V2.0 |
| 7 | Procesamiento síncrono | Colas con RabbitMQ/Redis | Más simple, suficiente para MVP |
| 8 | YOLO v8 via Python subprocess | ONNX Runtime en .NET | Más simple, acceso al ecosistema Python/ML |
| 9 | Nginx + Kestrel | Solo Kestrel, IIS | Nginx maneja SSL, archivos estáticos y proxy |
| 10 | Debian Slim para backend | Alpine | Compatibilidad con FFmpeg y OpenCV (glibc) |
| 11 | BDD + TDD híbrido | Solo TDD | User Stories se traducen directamente a Gherkin |
| 12 | SpecFlow | BDDfy, Machine.Specifications | Mejor integración con .NET y Visual Studio |
| 13 | xUnit | NUnit, MSTest | Mejor rendimiento, estándar moderno en .NET |
| 14 | FluentAssertions | Shouldly, Assert | Sintaxis más legible y expresiva |
| 15 | **API First** | Code First | Permite desarrollo paralelo, documentación automática |
| 16 | **Conventional Commits** | Commits libres | Historial ordenado, generación de changelogs |
| 17 | **Git Flow** | Trunk-based | Flujo estructurado para releases |
| 18 | **ADRs** | Decisiones no documentadas | Trazabilidad de decisiones arquitectónicas |
| 19 | **API Versioning por URL** | Header versioning | Simplicidad, más legible |

### Riesgos Identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Procesamiento intensivo en CPU | Alta | Alto | Frame sampling (1/5), timeout de 10 min |
| Baja precisión en detección de pelota | Alta | Medio | Aceptar limitación en MVP, TrackNet en V2.0 |
| Videos con ángulos no estándar | Media | Alto | Documentar requisitos de grabación |
| Costos de almacenamiento | Media | Medio | Local en MVP, S3 en V2.0 |
| Escalabilidad del procesamiento | Baja (MVP) | Alto | Arquitectura modular para migrar a async |

### Limitaciones Conocidas del MVP

- **Detección de pelota**: Funciona bien para pelotas de color amarillo/verde con buena iluminación. Falla con oclusiones o movimientos muy rápidos.
- **Ángulo de cámara**: Se asume grabación cenital o semi-cenital estándar. Ángulos laterales reducen la precisión.
- **Iluminación**: El análisis requiere buena iluminación. Canchas bajo techo con mala luz pueden afectar la detección.
- **Tamaño máximo**: 500MB limita videos de alta resolución/larga duración.
- **Procesamiento síncrono**: El usuario espera en la misma request. Para videos largos, el timeout del browser puede ser un problema.

---

## 📝 Estrategia de Logging y Monitoreo

### Herramientas Backend

| Propósito | Herramienta | Notas |
|-----------|-------------|-------|
| Logging estructurado | Serilog | Ya incluido en .NET, alto rendimiento |
| Formato de logs | JSON | Estructurado, fácil de parsear |
| Rotación de logs | File sink | Archivos diarios, retención 7 días |
| Monitoreo | Minimal | Logs + health checks en MVP |

### Herramientas Frontend

| Propósito | Herramienta | Notas |
|-----------|-------------|-------|
| Error tracking | Custom (API endpoint) | Enviar errores a nuestro backend |
| Monitoreo | WebVitals | Métricas de performance |
| Desarrollo | Console + browser DevTools | Para debugging local |

### Niveles de Log (Backend)

| Nivel | Cuándo usar |
|-------|-------------|
| **Debug** | Información de desarrollo, variables, flow |
| **Information** | Eventos normales (upload, process started, completed) |
| **Warning** | Situaciones anómalas pero manejables |
| **Error** | Excepciones, fallos de procesamiento |

### Logs del MVP (Backend)

| Evento | Nivel | Datos a registrar |
|--------|-------|-------------------|
| Video subido | Info | filename, size, duration |
| Procesamiento iniciado | Info | videoId, startTime |
| Procesamiento completado | Info | videoId, duration, result |
| Error de procesamiento | Error | videoId, exception, stack trace |
| Health check | Debug | endpoint, response time |

### Frontend Error Tracking (Custom)

```typescript
// Enviar errores a nuestro propio backend
const logClientError = async (error: Error, context?: object) => {
  await fetch('/api/logs/client-error', {
    method: 'POST',
    body: JSON.stringify({
      message: error.message,
      stack: error.stack,
      url: window.location.href,
      timestamp: new Date().toISOString()
    })
  });
};

// Capturar errores globales
window.onerror = (msg, url, line) => logClientError(new Error(msg));
window.onunhandledrejection = (e) => logClientError(e.reason);
```

---

## 🧪 Estrategia de Testing

### Enfoque Híbrido: BDD + TDD

Se adopta un enfoque híbrido que combina Behavior-Driven Development (BDD) para features de usuario y Test-Driven Development (TDD) para lógica de negocio.

| Tipo | Herramienta | Uso |
|------|-------------|-----|
| **BDD** | SpecFlow | Features de usuario (User Stories) |
| **TDD** | xUnit + FluentAssertions | Lógica de negocio |
| **Mocking** | Moq | Dependencias en tests unitarios |

### Justificación de la Decisión

| # | Factor | BDD | TDD |
|---|--------|-----|-----|
| 1 | User Stories existentes | ✅ Se traducen directamente a Gherkin | ❌ Requiere adaptación |
| 2 | Documentación viva | ✅ SpecFlow genera docs automáticas | ❌ No genera docs |
| 3 | Stakeholders no técnicos | ✅ Gherkin es legible | ❌ Requiere conocimiento técnico |
| 4 | Velocidad de desarrollo | ⚠️ Setup inicial más lento | ✅ Más rápido de iniciar |
| 5 | Lógica de negocio compleja | ❌ Overkill para unit tests | ✅ Ideal para validaciones |

### Integración con el Plan de Implementación

| Fase | Actividades de Testing |
|------|----------------------|
| **Fase 1** | Setup de SpecFlow, xUnit, FluentAssertions |
| **Fase 2** | Tests BDD para US-1 (Subida de Videos), Tests TDD para VideoValidationService |
| **Fase 3** | Tests TDD para StatisticsCalculator, HeatmapGenerator |
| **Fase 4** | Tests BDD para US-2, US-3, US-4, US-5 |
| **Fase 5** | Tests de integración, CI/CD pipeline |

### Cobertura de Tests por User Story

| User Story | Escenarios BDD | Tests TDD | Prioridad |
|------------|----------------|-----------|-----------|
| US-1: Subir Video | 5 | 8 | Alta |
| US-2: Ver Estadísticas | 5 | 6 | Alta |
| US-3: Descargar PDF | 3 | 2 | Media |
| US-4: Historial | 3 | 4 | Media |
| US-5: Monitorear | 4 | 3 | Alta |

### Estimación de Tiempo para Testing

| Actividad | Días | Dependencia |
|-----------|------|-------------|
| Setup SpecFlow + xUnit | 1 | Fase 1 |
| Tests BDD US-1 | 1 | Fase 2 |
| Tests TDD Validadores | 1 | Fase 2 |
| Tests TDD Procesamiento | 2 | Fase 3 |
| Tests BDD US-2 a US-5 | 2 | Fase 4 |
| Tests Integración + CI/CD | 1 | Fase 5 |
| **Total** | **8 días** | |

### Decisión Documentada

| # | Decisión | Alternativa Considerada | Justificación |
|---|---|---|---|
| 11 | BDD + TDD híbrido | Solo TDD | User Stories se traducen directamente a Gherkin |
| 12 | SpecFlow | BDDfy, Machine.Specifications | Mejor integración con .NET y Visual Studio |
| 13 | xUnit | NUnit, MSTest | Mejor rendimiento, estándar moderno en .NET |
| 14 | FluentAssertions | Shouldly, Assert | Sintaxis más legible y expresiva |

---

## 🔬 Riesgos Técnicos y Spikes de Validación

Los spikes son experimentos técnicos cortos (1-2 días) para validar riesgos antes de invertir semanas en implementación.

### Resumen de Riesgos

| Riesgo | Nivel | ¿Bloquea MVP? | Spike |
|---|---|---|---|
| YOLO con baja precisión en contexto pádel | 🟡 Medio | Parcialmente | Spike 1 |
| Mapeo incorrecto pixel → cancha real | 🟡 Medio | No (degraded) | Spike 3 |
| Integración .NET → Python frágil en Docker | 🟡 Medio | **Sí** | Spike 4 |
| Rendimiento síncrono muy lento | 🔴 Alto | **Sí** | Diferido |

---

### Spike 1 — Validación de YOLO en Videos de Pádel
⏱️ Estimado: 1-2 días

**Riesgo que valida**: YOLO v8 está entrenado con COCO dataset (personas en contextos generales). El pádel tiene particularidades:
- Cámara cenital cambia la forma del cuerpo respecto al training data
- Jugadores con ropa de colores similares al fondo de cancha
- Vidrios de la cancha crean reflejos que confunden el detector
- En partidos con árbitro y espectadores visibles, pueden aparecer falsos positivos

**Objetivo**: Confirmar que YOLO v8 detecta correctamente los 4 jugadores en videos reales de pádel con cámara cenital.

**Pasos**:
1. Tomar 3-5 clips de 30 segundos de partidos reales (distintas canchas, iluminaciones, ángulos)
2. Correr YOLO v8 nano con un script Python minimal
3. Revisar visualmente los bounding boxes generados
4. Medir: ¿detecta 4 jugadores?, ¿hay falsos positivos?, ¿se pierde alguno?

**Script mínimo**:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('partido_padel.mp4', classes=[0], save=True)
# Revisar el video resultante en spikes/runs/detect/
```

**Criterio de éxito**: Detección correcta de los 4 jugadores en > 85% de los frames en condiciones normales de grabación.

**Plan B si falla**:
- Probar YOLO v8 medium o large (más preciso, más lento)
- Evaluar fine-tuning con un dataset pequeño de pádel (~500 imágenes etiquetadas)
- Filtrar detecciones por zona de la cancha (ignorar espectadores fuera de los límites)

---

### Spike 3 — Transformación de Coordenadas Pixel → Cancha
⏱️ Estimado: 2 días

**Riesgo que valida**: Para calcular distancias en metros y generar heatmaps precisos, necesitamos transformar coordenadas de pixel a metros reales de la cancha. Esto requiere detectar las líneas de la cancha, lo cual puede ser difícil con:
- Sombras sobre las líneas
- Cámaras no perfectamente centradas
- Distorsión de lente
- Canchas con iluminación no uniforme

**Objetivo**: Verificar si podemos detectar las 4 esquinas de la cancha y aplicar una transformación homográfica confiable.

**Pasos**:
1. Intentar detectar las líneas de la cancha con OpenCV `HoughLinesP`
2. Identificar las 4 esquinas de la cancha (intersecciones de líneas)
3. Aplicar `cv2.getPerspectiveTransform` para rectificar la vista
4. Transformar posiciones de jugadores a coordenadas de cancha (10m x 20m)
5. Validar que las distancias calculadas sean realistas

**Script de referencia**:
```python
import cv2
import numpy as np

# Puntos detectados en imagen (ejemplo)
src_points = np.float32([[px1,py1],[px2,py2],[px3,py3],[px4,py4]])
# Cancha de pádel: 10m x 20m (normalizado a pixels)
dst_points = np.float32([[0,0],[640,0],[0,360],[640,360]])

M = cv2.getPerspectiveTransform(src_points, dst_points)
# Aplicar transformación a posiciones de jugadores
player_court = cv2.perspectiveTransform(player_pixels, M)
```

**Criterio de éxito**: Las 4 esquinas de la cancha detectadas con error < 10 pixels en videos de distintas cámaras y condiciones.

**Plan B si falla**:
- Para MVP: usar coordenadas normalizadas (0.0 - 1.0) en lugar de metros reales
- Pedir al usuario que marque manualmente las 4 esquinas al subir el video
- Asumir un factor de conversión fijo (pixels/metro) como aproximación

---

### Spike 4 — Integración .NET → Python subprocess en Docker
⏱️ Estimado: 1 día

**Riesgo que valida**: La integración via subprocess tiene varios riesgos prácticos:
- YOLO carga el modelo completo en RAM (~200MB). Con múltiples requests simultáneos puede haber OOM
- Si Python falla, el error puede quedar silenciado y el request queda colgado
- Path management de scripts y modelos dentro del contenedor
- Race conditions con archivos de video en `/uploads`

**Objetivo**: Confirmar que la integración funciona en Docker con manejo correcto de errores y requests concurrentes.

**Pasos**:
1. Crear endpoint .NET mínimo que llame a un script Python que cargue YOLO
2. Testear con un video real → verificar resultado correcto
3. Simular error Python (script que falla con excepción) → verificar captura en .NET
4. Lanzar 3 requests simultáneos → verificar sin OOM ni deadlocks
5. Medir uso de memoria con YOLO cargado N veces simultáneamente

**Criterio de éxito**: Manejo correcto de errores + 3 requests simultáneos sin OOM ni deadlocks en una máquina con 4GB RAM.

**Plan B si falla**:
- Implementar un proceso Python daemon que mantiene YOLO cargado (1 instancia compartida)
- Implementar una cola simple con `SemaphoreSlim` en .NET para limitar a 1 request de procesamiento a la vez
- Considerar FastAPI como servicio interno desde el inicio del MVP

---

### Spike Diferido — Benchmark de Rendimiento
⏱️ Estimado: 1 día | Diferido para post-implementación inicial

**Cuando ejecutar**: Una vez completados Spike 1 y Spike 4, y antes de comenzar la Fase 3 (procesamiento de video).

**Objetivo**: Medir el tiempo real de procesamiento de un video de 5 minutos en CPU (sin GPU) y decidir si el modelo síncrono es viable o necesitamos cambiar a asíncrono.

---

## 🎨 Fase 4: Visualización y UI — Plan Detallado

### 🎯 Objetivo

Implementar la capa de visualización y UI del Analizador de Pádel, incluyendo dashboard principal, heatmap con D3.js, visualizaciones estadísticas con Chart.js, reproductor de video, y sistema de descarga de reportes PDF.

### 📊 Estado Actual

**Frontend (App.tsx):**
- ✅ Subida de videos básica
- ✅ Lista de videos con estado
- ✅ Botón para iniciar análisis
- ❌ Dashboard principal
- ❌ Heatmap con D3.js
- ❌ Visualizaciones con Chart.js
- ❌ Reproductor de video
- ❌ Descarga de reportes PDF

**Backend (Program.cs):**
- ✅ Endpoints para videos (CRUD)
- ✅ Endpoints para análisis (iniciar, obtener, stats, heatmap, report)
- ❌ Endpoint stream de video
- ❌ Endpoint frames específicos
- ❌ Endpoint timeline de eventos
- ❌ Endpoint estadísticas dashboard

### 🔌 Endpoints Backend Necesarios

#### Endpoints Existentes (✅ Ya implementados):
- `GET /api/videos` - Listar videos
- `GET /api/videos/{id}` - Obtener video por ID
- `GET /api/analyses/{id}` - Obtener análisis
- `GET /api/analyses/{id}/stats` - Estadísticas del análisis
- `GET /api/analyses/{id}/heatmap` - Datos del heatmap
- `GET /api/analyses/{id}/report` - Descargar PDF

#### Endpoints Nuevos a Implementar:

| Endpoint | Método | Descripción | Prioridad |
|----------|--------|-------------|-----------|
| `/api/videos/{id}/stream` | GET | Stream de video para reproductor (soporta range requests) | Alta |
| `/api/analyses/{id}/frames/{frameNumber}` | GET | Frame específico (PNG/JPEG) | Media |
| `/api/analyses/{id}/timeline` | GET | Timeline de eventos (detecciones, cambios posición) | Alta |
| `/api/dashboard/stats` | GET | Estadísticas generales del dashboard | Alta |
| `/api/analyses/{id}/players/{playerIndex}/stats` | GET | Stats por jugador (1-4) | Media |
| `/api/analyses/{id}/heatmap/player/{playerIndex}` | GET | Heatmap por jugador | Media |

### 📦 Librerías y Dependencias

#### Frontend (según TECHNICAL.md y ADR-004):

```json
{
  "dependencies": {
    "react": "^18.x",
    "react-dom": "^18.x",
    "react-router-dom": "^6.x",
    "axios": "^1.x",
    "@mui/material": "^5.x",
    "@mui/icons-material": "^5.x",
    "@emotion/react": "^11.x",
    "@emotion/styled": "^11.x",
    "chart.js": "^4.x",
    "react-chartjs-2": "^5.x",
    "d3": "^7.x",
    "video.js": "^8.x",
    "react-player": "^2.x",
    "zustand": "^4.x",
    "html2canvas": "^1.x",
    "jspdf": "^2.x"
  },
  "devDependencies": {
    "@types/d3": "^7.x",
    "@types/react": "^18.x",
    "@types/react-dom": "^18.x",
    "typescript": "^5.x",
    "vite": "^5.x",
    "@vitejs/plugin-react": "^4.x"
  }
}
```

#### Backend (adicional para Fase 4):

```xml
<PackageReference Include="QuestPDF" Version="2024.x" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.x" />
```

### 🎯 Alineación con ADRs

| ADR | Decisión | Aplicación en Fase 4 |
|-----|----------|---------------------|
| **ADR-001** | Stack Tecnológico Backend | ASP.NET Core 10 con Minimal APIs, Entity Framework Core, SQLite |
| **ADR-003** | API First | Endpoints diseñados siguiendo OpenAPI 3.0, cliente TypeScript generado, documentación con Scalar |
| **ADR-004** | Estructura del Proyecto | `frontend/src/components/`, `frontend/src/pages/`, `frontend/src/services/`, `frontend/src/hooks/`, `frontend/src/store/` |
| **TECHNICAL.md** | Stack Frontend | React 18 + TypeScript, Vite, Material-UI, Video.js, Chart.js + D3.js, Axios, React Router, Zustand |

### 📝 Plan de Implementación por Sprints

#### Sprint 1: Estructura Base y Dashboard (3 días)

**Objetivo**: Crear estructura de componentes y dashboard principal

**Tareas**:
1. Crear estructura de carpetas según ADR-004
2. Instalar dependencias (MUI, Chart.js, D3.js, Video.js, Zustand)
3. Crear layout principal con navegación
4. Implementar dashboard con estadísticas básicas
5. Crear endpoint `GET /api/dashboard/stats` en backend

**Archivos a crear**:
- `frontend/src/components/Layout.tsx`
- `frontend/src/components/Navigation.tsx`
- `frontend/src/pages/Dashboard.tsx`
- `frontend/src/store/dashboardStore.ts`
- `frontend/src/services/api/dashboardService.ts`

**Backend**:
- Agregar endpoint `GET /api/dashboard/stats` en Program.cs

#### Sprint 2: Reproductor de Video (3 días)

**Objetivo**: Implementar reproductor de video con soporte para seek

**Tareas**:
1. Implementar endpoint stream de video en backend
2. Crear componente VideoPlayer con Video.js
3. Integrar reproductor con lista de videos
4. Agregar controles de reproducción

**Archivos a crear**:
- `frontend/src/components/VideoPlayer.tsx`
- `frontend/src/pages/VideoView.tsx`
- `frontend/src/services/api/videoStreamService.ts`

**Backend**:
- Agregar endpoint `GET /api/videos/{id}/stream` en Program.cs
- Implementar range requests para seek

#### Sprint 3: Heatmap con D3.js (3 días)

**Objetivo**: Implementar visualización de heatmap con D3.js

**Tareas**:
1. Crear componente Heatmap con D3.js
2. Implementar endpoint heatmap por jugador
3. Integrar heatmap con reproductor de video
4. Agregar controles de filtrado por jugador

**Archivos a crear**:
- `frontend/src/components/Heatmap.tsx`
- `frontend/src/components/HeatmapControls.tsx`
- `frontend/src/services/api/heatmapService.ts`

**Backend**:
- Agregar endpoint `GET /api/analyses/{id}/heatmap/player/{playerIndex}` en Program.cs

#### Sprint 4: Visualizaciones Estadísticas (3 días)

**Objetivo**: Implementar gráficos estadísticos con Chart.js

**Tareas**:
1. Crear componente StatisticsCharts con Chart.js
2. Implementar endpoint stats por jugador
3. Crear gráficos: detecciones, frames, tasa éxito
4. Integrar con dashboard

**Archivos a crear**:
- `frontend/src/components/StatisticsCharts.tsx`
- `frontend/src/components/DetectionChart.tsx`
- `frontend/src/components/FrameChart.tsx`
- `frontend/src/services/api/statsService.ts`

**Backend**:
- Agregar endpoint `GET /api/analyses/{id}/players/{playerIndex}/stats` en Program.cs

#### Sprint 5: Timeline y Frames (2 días)

**Objetivo**: Implementar timeline de eventos y visualización de frames

**Tareas**:
1. Crear componente Timeline con marcadores
2. Implementar endpoint timeline de eventos
3. Crear componente FrameViewer para frames específicos
4. Integrar timeline con reproductor

**Archivos a crear**:
- `frontend/src/components/Timeline.tsx`
- `frontend/src/components/FrameViewer.tsx`
- `frontend/src/services/api/timelineService.ts`

**Backend**:
- Agregar endpoint `GET /api/analyses/{id}/timeline` en Program.cs
- Agregar endpoint `GET /api/analyses/{id}/frames/{frameNumber}` en Program.cs

#### Sprint 6: Reportes PDF y Testing (3 días)

**Objetivo**: Implementar descarga de reportes PDF y testing

**Tareas**:
1. Crear componente ReportDownloader
2. Implementar generación de PDF con QuestPDF
3. Crear tests BDD para US-2 (Ver Estadísticas)
4. Crear tests TDD para servicios de visualización

**Archivos a crear**:
- `frontend/src/components/ReportDownloader.tsx`
- `frontend/src/services/api/reportService.ts`
- `backend/src/AnalizadorPadel.Api/Services/ReportService.cs`
- `tests/BDD/Features/US-2-VerEstadisticas.feature`
- `tests/TDD/Services/HeatmapServiceTests.cs`
- `tests/TDD/Services/StatsServiceTests.cs`

**Backend**:
- Agregar servicio `ReportService.cs` con QuestPDF
- Implementar generación de PDF con estadísticas y heatmap

### 🧪 Testing

#### Tests BDD (SpecFlow):
- US-2: Ver Estadísticas
  - Escenario 1: Ver estadísticas de análisis completado
  - Escenario 2: Ver heatmap de posiciones
  - Escenario 3: Filtrar por jugador
  - Escenario 4: Descargar reporte PDF
  - Escenario 5: Ver timeline de eventos

#### Tests TDD (xUnit + FluentAssertions):
- HeatmapServiceTests
  - Test: GetHeatmapAsync retorna datos correctos
  - Test: GetHeatmapByPlayerAsync filtra correctamente
  - Test: HandleAnalysisNotFound retorna null

- StatsServiceTests
  - Test: GetStatsAsync retorna estadísticas correctas
  - Test: GetPlayerStatsAsync filtra por jugador
  - Test: GetDashboardStatsAsync retorna datos dashboard

- VideoStreamServiceTests
  - Test: StreamVideoAsync retorna stream correcto
  - Test: HandleRangeRequests funciona correctamente
  - Test: HandleVideoNotFound retorna error

### 📅 Timeline

| Sprint | Duración | Entregables |
|--------|----------|-------------|
| Sprint 1 | 3 días | Dashboard, estructura base |
| Sprint 2 | 3 días | Reproductor de video |
| Sprint 3 | 3 días | Heatmap con D3.js |
| Sprint 4 | 3 días | Visualizaciones Chart.js |
| Sprint 5 | 2 días | Timeline y frames |
| Sprint 6 | 3 días | Reportes PDF y testing |
| **Total** | **17 días** | Fase 4 completa |

### 🎯 Criterios de Éxito

1. ✅ Dashboard muestra estadísticas en tiempo real
2. ✅ Reproductor de video funciona con seek
3. ✅ Heatmap muestra posiciones de jugadores
4. ✅ Gráficos estadísticos renderizan correctamente
5. ✅ Timeline muestra eventos del análisis
6. ✅ Reporte PDF se genera y descarga
7. ✅ Tests BDD pasan para US-2
8. ✅ Tests TDD pasan para servicios de visualización

### ⚠️ Riesgos y Mitigación

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| D3.js complejo para heatmap | Media | Medio | Usar librería simple-heat como alternativa |
| Video.js problemas de compatibilidad | Baja | Alto | Usar react-player como fallback |
| PDF generación lenta | Media | Medio | Generar en background, mostrar progreso |
| Chart.js rendimiento con muchos datos | Baja | Medio | Implementar sampling de datos |

---

*Última actualización: 15 de Marzo de 2026*
