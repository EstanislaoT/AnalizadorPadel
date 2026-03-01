# 🔧 Especificación Técnica — Analizador de Pádel

## Stack Tecnológico

### Backend (.NET)
- **Framework**: ASP.NET Core 10 (LTS)
- **Arquitectura**: Web API con controladores REST (modular)
- **Base de Datos**: SQLite (MVP) / PostgreSQL (V2.0+)
- **ORM**: Entity Framework Core
- **Autenticación**: No requerida en MVP (Post-MVP: JWT Tokens)
- **Procesamiento de Video**:
  - FFmpeg (extracción de frames y manipulación)
  - YOLO v8 vía Python subprocess (detección de jugadores)
  - OpenCV HSV + HoughCircles (detección de pelota - MVP)
  - Emgu CV (cálculo de posiciones y heatmaps en .NET)
  - Kalman Filter (suavizado de trayectorias)
  - Procesamiento síncrono (MVP)

### Frontend (React + TypeScript)
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand o Redux Toolkit
- **UI Components**: Material-UI (MUI)
- **Video Player**: Video.js
- **Charts/Visualization**: Chart.js + D3.js
- **HTTP Client**: Axios
- **Routing**: React Router

### Infraestructura MVP
- **Contenedores**: Docker + Docker Compose
- **Web Server**: Nginx (reverse proxy + archivos estáticos)
- **App Server Backend**: Kestrel (integrado en .NET)
- **App Server Frontend (dev)**: Vite Dev Server
- **Base de Datos**: SQLite (archivo local)
- **Almacenamiento**: Sistema de archivos local (`/uploads`)
- **Logging**: Serilog

### Infraestructura Futura (Post-MVP)
- **Base de Datos**: PostgreSQL (migración desde SQLite)
- **Almacenamiento**: AWS S3 o Azure Blob Storage
- **CDN**: CloudFlare
- **Monitoring**: Application Insights
- **Caché**: Redis
- **Autenticación**: JWT Tokens

### API First
- **Especificación**: OpenAPI 3.0 (Swagger)
- **Generación de Código**: NSwag (cliente TypeScript)
- **Documentación**: Scalar.AspNetCore
- **Mocking**: Prism (desarrollo frontend independiente)

---

## 🔌 Especificación OpenAPI

### Workflow API First

```
1. Diseñar API en OpenAPI (openapi.yaml)
         ↓
2. Generar servidor stub (NSwag)
         ↓
3. Mock server para frontend (Prism)
         ↓
4. Implementar lógica backend
         ↓
5. Generar cliente TypeScript (NSwag)
         ↓
6. Frontend consume API tipada
```

### Endpoints del MVP

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/videos` | Subir nuevo video |
| GET | `/api/videos` | Listar videos |
| GET | `/api/videos/{id}` | Obtener video por ID |
| DELETE | `/api/videos/{id}` | Eliminar video |
| POST | `/api/videos/{id}/analyse` | Iniciar análisis |
| GET | `/api/analyses/{id}` | Obtener análisis |
| GET | `/api/analyses/{id}/stats` | Estadísticas del análisis |
| GET | `/api/analyses/{id}/heatmap` | Datos del heatmap |
| GET | `/api/analyses/{id}/report` | Descargar PDF |

### Especificación OpenAPI (openapi.yaml)

```yaml
openapi: 3.0.3
info:
  title: Analizador de Pádel API
  description: API para análisis de partidos de pádel mediante procesamiento de video
  version: 1.0.0
  contact:
    name: AnalizadorPadel Team

servers:
  - url: http://localhost:5000/api
    description: Desarrollo
  - url: http://localhost/api
    description: Producción (Docker)

tags:
  - name: Videos
    description: Gestión de videos
  - name: Analyses
    description: Análisis de partidos

paths:
  /videos:
    post:
      tags: [Videos]
      summary: Subir nuevo video
      description: Sube un video de partido para su posterior análisis
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]
              properties:
                file:
                  type: string
                  format: binary
                  description: Archivo de video (MP4, AVI, MOV)
      responses:
        '201':
          description: Video subido exitosamente
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VideoResponse'
        '400':
          description: Error de validación
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '413':
          description: Archivo demasiado grande
    get:
      tags: [Videos]
      summary: Listar videos
      description: Obtiene la lista de videos subidos
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
        - name: status
          in: query
          schema:
            type: string
            enum: [uploaded, processing, completed, failed]
      responses:
        '200':
          description: Lista de videos
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VideoListResponse'

  /videos/{id}:
    get:
      tags: [Videos]
      summary: Obtener video por ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Video encontrado
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VideoResponse'
        '404':
          description: Video no encontrado
    delete:
      tags: [Videos]
      summary: Eliminar video
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '204':
          description: Video eliminado
        '404':
          description: Video no encontrado

  /videos/{id}/analyse:
    post:
      tags: [Videos]
      summary: Iniciar análisis del video
      description: Procesa el video y genera estadísticas
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Análisis completado
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResponse'
        '202':
          description: Análisis iniciado (modo asíncrono futuro)
        '404':
          description: Video no encontrado
        '408':
          description: Timeout en procesamiento

  /analyses/{id}:
    get:
      tags: [Analyses]
      summary: Obtener análisis por ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Análisis encontrado
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResponse'
        '404':
          description: Análisis no encontrado

  /analyses/{id}/stats:
    get:
      tags: [Analyses]
      summary: Obtener estadísticas del análisis
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
        - name: playerIndex
          in: query
          description: Índice del jugador (1-4)
          schema:
            type: integer
            minimum: 1
            maximum: 4
      responses:
        '200':
          description: Estadísticas del análisis
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StatisticsResponse'

  /analyses/{id}/heatmap:
    get:
      tags: [Analyses]
      summary: Obtener datos del heatmap
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
        - name: playerIndex
          in: query
          schema:
            type: integer
      responses:
        '200':
          description: Datos del heatmap
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HeatmapResponse'

  /analyses/{id}/report:
    get:
      tags: [Analyses]
      summary: Descargar reporte PDF
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Reporte PDF
          content:
            application/pdf:
              schema:
                type: string
                format: binary
        '404':
          description: Análisis no encontrado

components:
  schemas:
    VideoResponse:
      type: object
      required: [id, originalFileName, storagePath, fileSize, duration, format, status, uploadedAt]
      properties:
        id:
          type: string
          format: uuid
        originalFileName:
          type: string
          example: "partido_padel_2026-02-18.mp4"
        storagePath:
          type: string
        fileSize:
          type: integer
          description: Tamaño en bytes
        duration:
          type: number
          description: Duración en segundos
        format:
          type: string
          enum: [mp4, avi, mov]
        status:
          type: string
          enum: [uploaded, processing, completed, failed]
        uploadedAt:
          type: string
          format: date-time
        processedAt:
          type: string
          format: date-time
          nullable: true

    VideoListResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/VideoResponse'
        pagination:
          $ref: '#/components/schemas/Pagination'

    AnalysisResponse:
      type: object
      required: [id, videoId, totalTime, totalPoints]
      properties:
        id:
          type: string
          format: uuid
        videoId:
          type: string
          format: uuid
        totalTime:
          type: number
          description: Tiempo total en segundos
        totalPoints:
          type: integer
        playerStats:
          type: array
          items:
            $ref: '#/components/schemas/PlayerStats'
        createdAt:
          type: string
          format: date-time

    PlayerStats:
      type: object
      properties:
        playerIndex:
          type: integer
        distanceMeters:
          type: number
        avgSpeed:
          type: number
        maxSpeed:
          type: number
        timeAtNet:
          type: number
          description: Porcentaje del tiempo en la red
        timeAtBack:
          type: number
        timeAtSides:
          type: number

    StatisticsResponse:
      type: object
      properties:
        analysisId:
          type: string
          format: uuid
        playerIndex:
          type: integer
        statistics:
          type: array
          items:
            $ref: '#/components/schemas/Statistic'

    Statistic:
      type: object
      properties:
        statType:
          type: string
          enum: [distance, speed, position]
        value:
          type: number
        metadata:
          type: object
          additionalProperties: true

    HeatmapResponse:
      type: object
      properties:
        analysisId:
          type: string
          format: uuid
        playerIndex:
          type: integer
        points:
          type: array
          items:
            $ref: '#/components/schemas/HeatmapPoint'

    HeatmapPoint:
      type: object
      properties:
        x:
          type: number
          minimum: 0
          maximum: 10
          description: Coordenada X en metros
        y:
          type: number
          minimum: 0
          maximum: 20
          description: Coordenada Y en metros
        intensity:
          type: number
          minimum: 0
          maximum: 1

    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer
        totalPages:
          type: integer

    ErrorResponse:
      type: object
      required: [statusCode, errorCode, message, timestamp]
      properties:
        statusCode:
          type: integer
        errorCode:
          type: string
        message:
          type: string
        detail:
          type: string
        timestamp:
          type: string
          format: date-time
        requestId:
          type: string
```

### Configuración de Swashbuckle

```csharp
// Program.cs
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "Analizador de Pádel API",
        Version = "v1",
        Description = "API para análisis de partidos de pádel"
    });
    
    var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
    var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);
    c.IncludeXmlComments(xmlPath);
});

app.UseSwagger();
app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "API v1"));
```

### Configuración de NSwag para Cliente TypeScript

```json
// nswag.json
{
  "runtime": "Net100",
  "documentGenerator": {
    "aspNetCoreToOpenApi": {
      "project": "AnalizadorPadel.csproj",
      "output": "openapi.json"
    }
  },
  "codeGenerators": {
    "openApiToTypeScriptClient": {
      "input": "openapi.json",
      "output": "../frontend/src/services/api/generated/client.ts",
      "template": "Axios",
      "promiseType": "async-await",
      "generateClientInterfaces": true,
      "typeStyle": "Interface"
    }
  }
}
```

### Mock Server con Prism

```bash
# Instalar Prism
npm install -g @stoplight/prism-cli

# Iniciar mock server desde openapi.yaml
prism mock openapi.yaml --port 4010

# Frontend puede usar http://localhost:4010/api durante desarrollo
```

---

## Sistemas Operativos

| Entorno | Componente | Sistema Operativo |
|---|---|---|
| **Desarrollo** | Máquina local | macOS / Windows / Linux |
| **Producción** | Nginx | Linux (Alpine 3.x) |
| **Producción** | Backend (.NET) | Linux (Debian Slim) |
| **Producción** | Frontend (build) | Linux (Node Alpine) |
| **Producción** | PostgreSQL | Linux (Alpine 3.x) |

> ℹ️ Todos los componentes de producción corren en contenedores Docker Linux, independientemente del OS del desarrollador.

> ⚠️ FFmpeg y OpenCV se instalan en el **Dockerfile del backend** (Linux), no en la máquina local del desarrollador.

**¿Por qué Debian Slim para el backend?**
Las librerías nativas de FFmpeg y OpenCV/Emgu CV requieren `glibc` (Debian/Ubuntu). Alpine usa `musl libc`, lo que genera incompatibilidades con estas dependencias. Microsoft también usa Debian como base oficial para sus imágenes .NET.

---

## 🏗️ Arquitectura del Sistema

### Diagrama MVP

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
│ /usr/share/nginx │  │  (.NET Core :5000)│
└──────────────────┘  └────────┬─────────┘
                               │
               ┌───────────────┴──────────────┐
               │                              │
               ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
    │    PostgreSQL    │          │  Local Storage   │
    │    Database      │          │   (/uploads)     │
    └──────────────────┘          └──────────────────┘
```

**Rutas Nginx:**
- `/` → React SPA (archivos estáticos)
- `/api/` → Proxy → Kestrel :5000
- `/uploads/` → Archivos de video locales

### Diagrama Futuro

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │  Background     │
│   (React)       │◄──►│   (.NET Core)   │◄──►│  Processing     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CDN/Static    │    │   PostgreSQL    │    │   File Storage  │
│   (CloudFlare)  │    │   Database      │    │   (AWS S3)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Flujos de Procesamiento

### Flujo MVP (Síncrono)

1. **Subida de Video**: Usuario sube → Frontend valida → Backend guarda en `/uploads` + registro en BD
2. **Procesamiento**: Backend procesa en el mismo request → FFmpeg + YOLO + OpenCV → resultados en BD
3. **Visualización**: Frontend recibe respuesta directa → renderiza estadísticas y heatmaps

### Flujo Futuro (Asíncrono)

1. **Subida de Video**: Usuario sube (autenticado) → video va a S3 → Backend crea registro en BD
2. **Procesamiento**: Background service descarga de S3 → procesa → guarda en BD → notifica via WebSocket
3. **Visualización**: Frontend consulta API → renderiza resultados

---

## 🗄️ Diseño de Base de Datos

### Esquema MVP (Sin Autenticación)

```sql
-- Videos
CREATE TABLE Videos (
    Id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    OriginalFileName VARCHAR(255) NOT NULL,
    StoragePath VARCHAR(500) NOT NULL,
    FileSize BIGINT NOT NULL,
    Duration DECIMAL(10,2) NOT NULL,
    Format VARCHAR(10) NOT NULL,
    Status VARCHAR(20) DEFAULT 'uploaded', -- uploaded, processing, completed, failed
    UploadedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ProcessedAt TIMESTAMP NULL
);

-- Análisis
CREATE TABLE Analyses (
    Id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    VideoId UUID NOT NULL REFERENCES Videos(Id),
    TotalTime DECIMAL(10,2) NOT NULL,
    TotalPoints INTEGER NOT NULL,
    PlayerStats JSONB NOT NULL,
    BallTracking JSONB NOT NULL,
    CourtPositions JSONB NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Estadísticas Detalladas
CREATE TABLE Statistics (
    Id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    AnalysisId UUID NOT NULL REFERENCES Analyses(Id),
    StatType VARCHAR(50) NOT NULL, -- distance, speed, position, etc.
    PlayerIndex INTEGER NOT NULL,  -- 1 or 2
    Value DECIMAL(10,2) NOT NULL,
    Metadata JSONB NULL
);

-- Heatmap Data
CREATE TABLE HeatmapData (
    Id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    AnalysisId UUID NOT NULL REFERENCES Analyses(Id),
    PlayerIndex INTEGER NOT NULL,
    XCoordinate DECIMAL(5,2) NOT NULL,
    YCoordinate DECIMAL(5,2) NOT NULL,
    Intensity DECIMAL(3,2) NOT NULL,
    Timestamp DECIMAL(10,2) NOT NULL
);
```

### Esquema Futuro (Con Autenticación)

```sql
-- Usuarios
CREATE TABLE Users (
    Id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    Email VARCHAR(255) UNIQUE NOT NULL,
    PasswordHash VARCHAR(255) NOT NULL,
    Name VARCHAR(100) NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agregar FK a Videos
ALTER TABLE Videos ADD COLUMN UserId UUID REFERENCES Users(Id);
```

---

## 🎥 Módulo de Procesamiento de Video

### Pipeline Completo

```
Video (.mp4)
    │
    ▼
FFmpeg (extracción de frames cada N segundos)
    │
    ├──► Python + YOLO v8 → Posiciones [x, y] de jugadores por frame
    │
    ├──► OpenCV HSV + HoughCircles → Posición [x, y] de pelota por frame
    │
    ▼
Kalman Filter (suavizado de trayectorias)
    │
    ▼
Emgu CV en .NET (cálculo de estadísticas)
    ├── Distancia recorrida
    ├── Velocidad promedio / máxima
    ├── Posiciones en cancha (heatmap)
    └── Tiempo en zonas (red, fondo, laterales)
    │
    ▼
PostgreSQL (persistencia de resultados)
```

### Comparativa: Detección de Jugadores

| Herramienta | Precisión | Velocidad | Soporte .NET | Decisión |
|---|---|---|---|---|
| **YOLO v8** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Vía Python | ✅ **Elegida** |
| OpenCV MOG2 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Nativo (Emgu) | ❌ Detecta movimiento, no personas |
| MediaPipe | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Vía Python | ❌ Mejor para pose estimation |
| Detectron2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Solo Python | ❌ Overkill para MVP |
| ML.NET | ⭐⭐⭐ | ⭐⭐⭐ | Nativo | ❌ Sin modelos pre-entrenados listos |

### Comparativa: Detección de Pelota

| Herramienta | Precisión Pádel | Complejidad | Decisión |
|---|---|---|---|
| **OpenCV HSV + HoughCircles** | ⭐⭐ | Baja | ✅ **Elegida MVP** |
| TrackNet | ⭐⭐⭐⭐⭐ | Alta | 🔜 Post-MVP |
| YOLO v8 (general) | ⭐⭐⭐ | Media | ❌ Pierde pelota en alta velocidad |
| OpenCV Optical Flow | ⭐⭐⭐ | Media | ❌ Solo tracking, no detección |
| Kalman Filter solo | ⭐⭐⭐⭐ | Media | ❌ Predictor, no detector |

> ⚠️ La pelota de pádel es pequeña y rápida. A 30fps puede aparecer como blur. HSV por color funciona para casos básicos. TrackNet se incorporará en V2.0.

### Comparativa: Integración Python ↔ .NET

| Estrategia | Pros | Contras | Decisión |
|---|---|---|---|
| **Python subprocess** | Simple, sin infra adicional | Proceso bloqueante | ✅ **Elegida MVP** |
| FastAPI microservicio | Escalable, REST API | Requiere servicio extra | 🔜 Post-MVP |
| ONNX Runtime (.NET) | Sin Python | Setup complejo | ❌ Descartada |
| gRPC | Muy rápido, tipado fuerte | Alta configuración | 🔜 V3.0 |

### Parámetros de Procesamiento

| Parámetro | Valor MVP | Justificación |
|---|---|---|
| Frame sampling | 1 cada 5 frames (6fps efectivos) | Balance precisión/velocidad |
| Modelo YOLO | yolov8n (nano, 6MB) | Rápido en CPU |
| Tamaño frame análisis | 640x360 | Reducido para velocidad |
| Timeout procesamiento | 10 min | Límite razonable para partido |

### Código de Integración (Ejemplo)

```csharp
// SyncVideoProcessingService.cs - .NET llama a Python
public async Task<DetectionResult> DetectPlayersAsync(string videoPath)
{
    var process = new Process {
        StartInfo = new ProcessStartInfo {
            FileName = "python3",
            Arguments = $"scripts/detect_players.py \"{videoPath}\"",
            RedirectStandardOutput = true,
            UseShellExecute = false
        }
    };
    process.Start();
    var output = await process.StandardOutput.ReadToEndAsync();
    await process.WaitForExitAsync();
    return JsonSerializer.Deserialize<DetectionResult>(output);
}
```

```python
# scripts/detect_players.py - YOLO v8
from ultralytics import YOLO
import cv2, json, sys

model = YOLO('models/yolov8n.pt')
cap = cv2.VideoCapture(sys.argv[1])
detections, frame_count = [], 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if frame_count % 5 == 0:  # 1 de cada 5 frames
        results = model(frame, classes=[0])  # class 0 = person
        players = [{'x': (b.xyxy[0][0]+b.xyxy[0][2])/2,
                    'y': (b.xyxy[0][1]+b.xyxy[0][3])/2,
                    'confidence': float(b.conf[0])}
                   for b in results[0].boxes]
        detections.append({'frame': frame_count,
                           'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC),
                           'players': players})
    frame_count += 1

cap.release()
print(json.dumps({'detections': detections}))
```

---

## 📁 Estructura del Proyecto

```
AnalizadorPadel/
├── backend/
│   ├── src/
│   │   ├── Controllers/
│   │   │   ├── VideosController.cs
│   │   │   └── AnalysesController.cs
│   │   ├── Services/
│   │   │   ├── Interfaces/
│   │   │   │   ├── IVideoStorageService.cs
│   │   │   │   ├── IVideoProcessingService.cs
│   │   │   │   └── IAnalysisService.cs
│   │   │   ├── Implementations/
│   │   │   │   ├── LocalVideoStorageService.cs
│   │   │   │   ├── SyncVideoProcessingService.cs
│   │   │   │   └── AnalysisService.cs
│   │   │   └── DTOs/
│   │   │       ├── VideoUploadDTO.cs
│   │   │       └── AnalysisResultDTO.cs
│   │   ├── Models/
│   │   │   ├── Entities/
│   │   │   │   ├── Video.cs
│   │   │   │   └── Analysis.cs
│   │   │   └── Enums/
│   │   │       └── VideoStatus.cs
│   │   ├── Data/
│   │   │   ├── ApplicationDbContext.cs
│   │   │   ├── Repositories/
│   │   │   └── Migrations/
│   │   ├── Middleware/
│   │   │   └── ExceptionHandlingMiddleware.cs
│   │   └── Configuration/
│   │       └── AppSettings.cs
│   ├── tests/
│   │   ├── Unit/
│   │   └── Integration/
│   ├── scripts/
│   │   ├── detect_players.py
│   │   ├── detect_ball.py
│   │   └── requirements.txt
│   ├── models/
│   │   └── yolov8n.pt
│   ├── uploads/
│   ├── Dockerfile
│   └── AnalizadorPadel.csproj
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Button/
│   │   │   │   ├── Input/
│   │   │   │   └── Loading/
│   │   │   ├── video/
│   │   │   │   ├── VideoUploader/
│   │   │   │   ├── VideoPlayer/
│   │   │   │   └── VideoPreview/
│   │   │   ├── analysis/
│   │   │   │   ├── StatsDisplay/
│   │   │   │   ├── Heatmap/
│   │   │   │   └── ReportDownload/
│   │   │   └── layout/
│   │   │       ├── Header/
│   │   │       ├── Sidebar/
│   │   │       └── Footer/
│   │   ├── pages/
│   │   │   ├── Dashboard/
│   │   │   ├── Upload/
│   │   │   └── Analysis/
│   │   ├── services/
│   │   │   ├── api/
│   │   │   │   ├── videoService.ts
│   │   │   │   └── analysisService.ts
│   │   │   └── storage/
│   │   │       └── localStorageService.ts
│   │   ├── hooks/
│   │   │   ├── useVideoUpload.ts
│   │   │   └── useAnalysis.ts
│   │   ├── types/
│   │   │   ├── Video.ts
│   │   │   └── Analysis.ts
│   │   ├── utils/
│   │   │   ├── constants.ts
│   │   │   ├── helpers.ts
│   │   │   └── validators.ts
│   │   ├── styles/
│   │   │   ├── globals.css
│   │   │   └── theme.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── Dockerfile
├── nginx/
│   └── nginx.conf
├── docker-compose.yml
├── docs/
│   ├── PRODUCT.md
│   ├── TECHNICAL.md
│   └── PLANNING.md
├── README.md
└── .gitignore
```

---

## 🔧 Principios de Diseño Modular

### Backend — Inyección de Dependencias

```csharp
// Program.cs
builder.Services.AddScoped<IVideoStorageService, LocalVideoStorageService>();
builder.Services.AddScoped<IVideoProcessingService, SyncVideoProcessingService>();
builder.Services.AddScoped<IAnalysisService, AnalysisService>();
```

### Frontend — Interfaces Desacopladas

```typescript
interface IVideoService {
  uploadVideo(file: File): Promise<VideoUploadResponse>;
  getAnalysis(videoId: string): Promise<AnalysisResult>;
}

interface IStorageService {
  save(key: string, data: any): void;
  get(key: string): any;
}
```

### Preparación para Escalar

| Componente Actual | Reemplazo Futuro |
|---|---|
| `LocalVideoStorageService` | `S3VideoStorageService` |
| `SyncVideoProcessingService` | `AsyncVideoProcessingService` |
| Python subprocess | FastAPI microservicio |
| OpenCV HSV (pelota) | TrackNet |
| Sin autenticación | JWT Middleware |

---

## 🚨 Manejo de Errores

### Códigos de Error

```csharp
public enum ErrorCode
{
    // Errores de Usuario (4xx)
    InvalidFileFormat = 4001,
    FileTooLarge = 4002,
    VideoTooShort = 4003,
    VideoCorrupted = 4004,
    ProcessingTimeout = 4005,
    
    // Errores de Servidor (5xx)
    StorageUnavailable = 5001,
    DatabaseConnectionFailed = 5002,
    PythonScriptFailed = 5003,
    ModelNotFound = 5004,
    FFmpegError = 5005,
    UnknownError = 5000
}
```

### Estructura de Respuesta de Error

```csharp
public class ApiErrorResponse
{
    public int StatusCode { get; set; }
    public string ErrorCode { get; set; }
    public string Message { get; set; }
    public string Detail { get; set; }
    public DateTime Timestamp { get; set; }
    public string RequestId { get; set; }
}
```

### Ejemplo de Respuesta

```json
{
    "statusCode": 400,
    "errorCode": "FileTooLarge",
    "message": "El archivo excede el tamaño máximo permitido",
    "detail": "Tamaño del archivo: 750MB. Máximo permitido: 500MB",
    "timestamp": "2026-02-18T19:30:00Z",
    "requestId": "abc123-def456"
}
```

### Estrategias por Tipo de Error

| Tipo de Error | Estrategia | Respuesta al Usuario |
|--------------|------------|---------------------|
| Archivo inválido | Validación temprana | Mensaje claro con formato esperado |
| Tamaño excedido | Validación temprana | Mensaje con tamaño máximo |
| Processing timeout | Timeout configurado | "El video tardó demasiado, intenta uno más corto" |
| Fallo de YOLO | Retry x1 luego fallback | "No se pudieron detectar jugadores" |
| Fallo de BD | Retry x3 con backoff | "Error interno, intenta más tarde" |
| Storage lleno | Check previo | "Espacio insuficiente, contacta al admin" |

### Middleware de Excepciones

```csharp
public class ExceptionHandlingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<ExceptionHandlingMiddleware> _logger;

    public ExceptionHandlingMiddleware(RequestDelegate next, ILogger<ExceptionHandlingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await _next(context);
        }
        catch (VideoProcessingException ex)
        {
            _logger.LogError(ex, "Error procesando video");
            context.Response.StatusCode = 500;
            await context.Response.WriteAsJsonAsync(new ApiErrorResponse
            {
                StatusCode = 500,
                ErrorCode = nameof(ErrorCode.PythonScriptFailed),
                Message = "Error al procesar el video",
                Detail = ex.Message,
                Timestamp = DateTime.UtcNow,
                RequestId = context.TraceIdentifier
            });
        }
        catch (FileNotFoundException ex)
        {
            _logger.LogError(ex, "Archivo no encontrado");
            context.Response.StatusCode = 404;
            await context.Response.WriteAsJsonAsync(new ApiErrorResponse
            {
                StatusCode = 404,
                ErrorCode = nameof(ErrorCode.VideoCorrupted),
                Message = "El archivo de video no fue encontrado",
                Detail = ex.Message,
                Timestamp = DateTime.UtcNow,
                RequestId = context.TraceIdentifier
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error no controlado");
            context.Response.StatusCode = 500;
            await context.Response.WriteAsJsonAsync(new ApiErrorResponse
            {
                StatusCode = 500,
                ErrorCode = nameof(ErrorCode.UnknownError),
                Message = "Ha ocurrido un error inesperado",
                Detail = "Contacte al administrador",
                Timestamp = DateTime.UtcNow,
                RequestId = context.TraceIdentifier
            });
        }
    }
}
```

### Políticas de Retry

```csharp
public class ProcessingRetryPolicy
{
    private readonly ILogger<ProcessingRetryPolicy> _logger;

    public ProcessingRetryPolicy(ILogger<ProcessingRetryPolicy> logger)
    {
        _logger = logger;
    }

    public IAsyncPolicy<ProcessResult> Policy => Policy
        .HandleResult<ProcessResult>(r => r.Failed)
        .WaitAndRetryAsync(
            retryCount: 3,
            sleepDurationProvider: retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)),
            onRetry: (outcome, timespan, retryAttempt, context) =>
            {
                _logger.LogWarning("Retry {RetryAttempt} después de {Delay}s", 
                    retryAttempt, timespan.TotalSeconds);
            });
}
```

### Validaciones Preventivas

```csharp
public class VideoValidationService
{
    private static readonly string[] AllowedExtensions = { ".mp4", ".avi", ".mov" };
    private static readonly string[] AllowedMimeTypes = { "video/mp4", "video/x-msvideo", "video/quicktime" };
    private const int MaxFileSizeMB = 500;
    private const int MinDurationSeconds = 60;

    public ValidationResult Validate(IFormFile file)
    {
        // Validar extensión
        var extension = Path.GetExtension(file.FileName).ToLower();
        if (!AllowedExtensions.Contains(extension))
            return ValidationResult.Fail("Formato no soportado. Use MP4, AVI o MOV.");

        // Validar MIME type (magic number)
        if (!AllowedMimeTypes.Contains(file.ContentType))
            return ValidationResult.Fail("El archivo no es un video válido.");

        // Validar tamaño
        if (file.Length > MaxFileSizeMB * 1024 * 1024)
            return ValidationResult.Fail($"El archivo excede {MaxFileSizeMB}MB.");

        return ValidationResult.Success();
    }
}
```

---

## Configuración de Servidores

### Nginx (`nginx/nginx.conf`)

```nginx
server {
    listen 80;

    # Frontend - React SPA
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Backend - .NET API
    location /api/ {
        proxy_pass http://backend:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection keep-alive;
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        client_max_body_size 500M;
    }

    # Videos - Archivos locales
    location /uploads/ {
        alias /var/uploads/;
        add_header Accept-Ranges bytes;
    }
}
```

### Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf
      - frontend_build:/usr/share/nginx/html
      - uploads:/var/uploads
    depends_on:
      - backend
      - frontend

  backend:
    build: ./backend
    expose:
      - "5000"
    volumes:
      - uploads:/app/uploads
    environment:
      - ConnectionStrings__DefaultConnection=Host=postgres;Database=padeldb;Username=postgres;Password=postgres
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    volumes:
      - frontend_build:/app/dist

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=padeldb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
  uploads:
  frontend_build:
```

---

## 🧪 Estrategia de Testing

### Enfoque Híbrido: BDD + TDD

| Tipo | Herramienta | Uso |
|------|-------------|-----|
| **BDD** | SpecFlow | Features de usuario (User Stories) |
| **TDD** | xUnit + FluentAssertions | Lógica de negocio |
| **Mocking** | Moq | Dependencias en tests unitarios |

### Justificación

- Las User Stories definidas en PRODUCT.md se traducen directamente a escenarios Gherkin
- SpecFlow genera documentación viva del sistema
- xUnit es el estándar de .NET con mejor rendimiento
- FluentAssertions mejora legibilidad de tests

### Herramientas

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| xUnit | 2.x | Framework de testing principal |
| SpecFlow | 3.x | BDD con Gherkin |
| FluentAssertions | 6.x | Assertions expresivas |
| Moq | 4.x | Mock objects |

### Estructura de Tests

```
backend/tests/
├── Unit/                    # TDD - Tests unitarios puros
│   ├── Validators/          # VideoValidationService tests
│   │   └── VideoValidationServiceTests.cs
│   ├── Services/            # Lógica de negocio
│   │   ├── StatisticsCalculatorTests.cs
│   │   └── HeatmapGeneratorTests.cs
│   └── Helpers/             # Utilidades
│       └── FileHelperTests.cs
├── Integration/             # Tests de integración API
│   └── Controllers/
│       ├── VideosControllerTests.cs
│       └── AnalysesControllerTests.cs
└── BDD/                     # SpecFlow
    ├── Features/            # Archivos .feature
    │   ├── VideoUpload.feature
    │   ├── Statistics.feature
    │   └── Processing.feature
    ├── Steps/               # Step definitions
    │   ├── VideoUploadSteps.cs
    │   └── CommonSteps.cs
    └── Hooks/               # Setup/Teardown
        └── TestHooks.cs
```

### Ejemplo: Feature BDD (Gherkin)

```gherkin
Feature: Subida de Videos
    Como jugador de pádel
    Quiero subir un video de mi partido
    Para obtener un análisis automático de mi juego

    @smoke
    Scenario: Video válido se procesa exitosamente
        Given que estoy en la página de subida
        When arrastro un video válido de 100MB
        Then veo la barra de progreso completar al 100%
        And puedo hacer clic en "Procesar"

    @validation
    Scenario: Video muy grande muestra error
        Given que estoy en la página de subida
        When arrastro un video de 750MB
        Then veo el mensaje "El archivo excede 500MB"
```

### Ejemplo: Step Definitions (SpecFlow)

```csharp
[Binding]
public class VideoUploadSteps
{
    private readonly UploadContext _context;
    
    public VideoUploadSteps(UploadContext context)
    {
        _context = context;
    }

    [Given(@"que estoy en la página de subida")]
    public void GivenQueEstoyEnPaginaSubida()
    {
        _context.Page = new UploadPage();
    }

    [When(@"arrastro un video válido de (.*)MB")]
    public void WhenArrojoVideoValido(int sizeMB)
    {
        _context.Result = _context.Uploader.Upload($"test_{sizeMB}mb.mp4");
    }

    [Then(@"veo la barra de progreso completar al (.*)%")]
    public void ThenVeoBarraProgreso(int percentage)
    {
        _context.Result.Progress.Should().Be(percentage);
    }
}
```

### Ejemplo: Test TDD (xUnit + FluentAssertions)

```csharp
public class VideoValidationServiceTests
{
    private readonly VideoValidationService _sut;

    public VideoValidationServiceTests()
    {
        _sut = new VideoValidationService();
    }

    [Fact]
    public void Validate_CuandoArchivoEsMP4_DebeRetornarSuccess()
    {
        // Arrange
        var file = CreateMockFile("video.mp4", "video/mp4", 100 * 1024 * 1024);

        // Act
        var result = _sut.Validate(file);

        // Assert
        result.IsValid.Should().BeTrue();
    }

    [Theory]
    [InlineData("video.exe", "application/exe")]
    [InlineData("video.txt", "text/plain")]
    [InlineData("video.pdf", "application/pdf")]
    public void Validate_CuandoFormatoInvalido_DebeRetornarError(string fileName, string mimeType)
    {
        // Arrange
        var file = CreateMockFile(fileName, mimeType, 1024);

        // Act
        var result = _sut.Validate(file);

        // Assert
        result.IsValid.Should().BeFalse();
        result.ErrorMessage.Should().Contain("Formato no soportado");
    }

    [Fact]
    public void Validate_CuandoArchivoExcede500MB_DebeRetornarError()
    {
        // Arrange
        var file = CreateMockFile("video.mp4", "video/mp4", 600 * 1024 * 1024);

        // Act
        var result = _sut.Validate(file);

        // Assert
        result.IsValid.Should().BeFalse();
        result.ErrorMessage.Should().Contain("excede");
    }
}
```

### Cobertura de Tests por User Story

| User Story | Escenarios BDD | Tests TDD | Prioridad |
|------------|----------------|-----------|-----------|
| US-1: Subir Video | 5 | 8 | Alta |
| US-2: Ver Estadísticas | 5 | 6 | Alta |
| US-3: Descargar PDF | 3 | 2 | Media |
| US-4: Historial | 3 | 4 | Media |
| US-5: Monitorear | 4 | 3 | Alta |

### Configuración de CI/CD

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup .NET
        uses: actions/setup-dotnet@v3
        with:
          dotnet-version: 10.0.x
      - name: Restore dependencies
        run: dotnet restore backend/AnalizadorPadel.sln
      - name: Build
        run: dotnet build backend/AnalizadorPadel.sln --no-restore
      - name: Test
        run: dotnet test backend/AnalizadorPadel.sln --no-build --verbosity normal
```

---

*Última actualización: 18 de Febrero 2026*
