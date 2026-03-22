# 🔧 Especificación Técnica — Analizador de Pádel

## Stack Tecnológico

### Backend (.NET)
- **Framework**: ASP.NET Core 10 (LTS)
- **Arquitectura**: Web API con Minimal APIs (REST)
- **Base de Datos**: SQLite con Entity Framework Core
- **ORM**: Entity Framework Core 10
- **Autenticación**: No requerida en MVP (Post-MVP: JWT Tokens)
- **Documentación API**: Scalar.AspNetCore (OpenAPI 3.0)
- **Logging**: Serilog (Console + File)
- **Procesamiento de Video**:
  - FFmpeg (extracción de frames y manipulación)
  - YOLO v8 vía Python subprocess (detección de jugadores)
  - OpenCV (procesamiento de imágenes)
  - Procesamiento síncrono (MVP)

### Frontend (React + TypeScript)
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand
- **UI Components**: Material-UI (MUI) v5
- **Video Player**: React Player
- **Charts/Visualization**: Chart.js + D3.js
- **HTTP Client**: Axios
- **Routing**: React Router v6
- **Testing**: Vitest + React Testing Library + MSW (Mock Service Worker)
- **API Client**: Generado con openapi-typescript

### Infraestructura MVP
- **Contenedores**: Docker + Docker Compose
- **Web Server**: Nginx (reverse proxy + archivos estáticos)
- **App Server Backend**: Kestrel (integrado en .NET)
- **App Server Frontend (dev)**: Vite Dev Server
- **Base de Datos**: SQLite (archivo local)
- **Almacenamiento**: Sistema de archivos local (`/uploads`, `/outputs`)
- **Logging**: Serilog con rotación diaria

### Infraestructura Futura (Post-MVP)
- **Base de Datos**: PostgreSQL (migración desde SQLite)
- **Almacenamiento**: AWS S3 o Azure Blob Storage
- **CDN**: CloudFlare
- **Monitoring**: Application Insights
- **Caché**: Redis
- **Autenticación**: JWT Tokens
- **Procesamiento**: Background workers asíncronos

### API First
- **Especificación**: OpenAPI 3.0
- **Generación de Código**: openapi-typescript (cliente TypeScript)
- **Documentación**: Scalar.AspNetCore

---

## 🔌 Especificación OpenAPI

### Workflow API First

```
1. Diseñar API en OpenAPI (openapi.json)
         ↓
2. Implementar lógica backend con Minimal APIs
         ↓
3. Generar cliente TypeScript (openapi-typescript)
         ↓
4. Frontend consume API tipada
```

### Endpoints Implementados

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/videos` | Subir nuevo video |
| GET | `/api/videos` | Listar videos |
| GET | `/api/videos/{id}` | Obtener video por ID |
| DELETE | `/api/videos/{id}` | Eliminar video |
| POST | `/api/videos/{id}/analyse` | Iniciar análisis del video |
| GET | `/api/analyses` | Listar análisis |
| GET | `/api/analyses/{id}` | Obtener análisis por ID |
| GET | `/api/dashboard/stats` | Estadísticas del dashboard |
| GET | `/api/health` | Health check |

### Configuración de OpenAPI con Scalar

```csharp
// Program.cs
builder.Services.AddOpenApi();

var app = builder.Build();

app.MapOpenApi();
app.MapScalarApiReference();
```

### Generación de Cliente TypeScript

```json
// frontend/package.json
{
  "scripts": {
    "codegen": "openapi-typescript ./src/services/api/openapi.json --output ./src/services/api/generated/types.ts"
  }
}
```

---

## Sistemas Operativos

| Entorno | Componente | Sistema Operativo |
|---|---|---|
| **Desarrollo** | Máquina local | macOS / Windows / Linux |
| **Producción** | Nginx | Linux (Alpine 3.x) |
| **Producción** | Backend (.NET) | Linux (Debian Slim) |
| **Producción** | Frontend (build) | Linux (Node Alpine) |

> ℹ️ Todos los componentes de producción corren en contenedores Docker Linux, independientemente del OS del desarrollador.

> ⚠️ FFmpeg y Python/YOLO se instalan en el **Dockerfile del backend** (Linux), no en la máquina local del desarrollador.

**¿Por qué Debian Slim para el backend?**
Las librerías nativas de FFmpeg y Python requieren `glibc` (Debian/Ubuntu). Alpine usa `musl libc`, lo que genera incompatibilidades. Microsoft también usa Debian como base oficial para sus imágenes .NET.

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
│ /usr/share/nginx │  │  (.NET :5000)    │
└──────────────────┘  └────────┬─────────┘
                               │
               ┌───────────────┴──────────────┐
               │                              │
               ▼                              ▼
    ┌──────────────────┐          ┌──────────────────┐
│    SQLite          │          │  Local Storage   │
│    Database        │          │   (/uploads)     │
└──────────────────┘          └──────────────────┘
```

**Rutas Nginx:**
- `/` → React SPA (archivos estáticos)
- `/api/` → Proxy → Kestrel :5000

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
2. **Procesamiento**: Backend procesa en el mismo request → FFmpeg + YOLO → resultados en BD
3. **Visualización**: Frontend recibe respuesta directa → renderiza estadísticas

### Flujo Futuro (Asíncrono)

1. **Subida de Video**: Usuario sube (autenticado) → video va a S3 → Backend crea registro en BD
2. **Procesamiento**: Background service descarga de S3 → procesa → guarda en BD → notifica via WebSocket
3. **Visualización**: Frontend consulta API → renderiza resultados

---

## 🗄️ Diseño de Base de Datos

### Esquema MVP (SQLite)

```csharp
// Entities
public class VideoEntity
{
    public Guid Id { get; set; }
    public string OriginalFileName { get; set; } = string.Empty;
    public string StoragePath { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public double Duration { get; set; }
    public string Format { get; set; } = string.Empty;
    public string Status { get; set; } = "uploaded";
    public DateTime UploadedAt { get; set; }
    public DateTime? ProcessedAt { get; set; }
    public AnalysisEntity? Analysis { get; set; }
}

public class AnalysisEntity
{
    public Guid Id { get; set; }
    public Guid VideoId { get; set; }
    public VideoEntity Video { get; set; } = null!;
    public double TotalTime { get; set; }
    public int TotalPoints { get; set; }
    public string PlayerStatsJson { get; set; } = "[]";
    public DateTime CreatedAt { get; set; }
}
```

### Configuración EF Core

```csharp
// PadelDbContext.cs
public class PadelDbContext : DbContext
{
    public PadelDbContext(DbContextOptions<PadelDbContext> options) : base(options) { }

    public DbSet<VideoEntity> Videos { get; set; }
    public DbSet<AnalysisEntity> Analyses { get; set; }
}
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
    ▼
Procesamiento de datos
    ├── Distancia recorrida
    ├── Velocidad promedio / máxima
    ├── Posiciones en cancha
    └── Tiempo en zonas
    │
    ▼
SQLite (persistencia de resultados)
```

### Comparativa: Detección de Jugadores

| Herramienta | Precisión | Velocidad | Soporte .NET | Decisión |
|---|---|---|---|---|
| **YOLO v8** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Vía Python | ✅ **Elegida** |
| OpenCV MOG2 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Nativo (Emgu) | ❌ Detecta movimiento, no personas |
| MediaPipe | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Vía Python | ❌ Mejor para pose estimation |
| Detectron2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Solo Python | ❌ Overkill para MVP |
| ML.NET | ⭐⭐⭐ | ⭐⭐⭐ | Nativo | ❌ Sin modelos pre-entrenados listos |

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

### Código de Integración

```csharp
// VideoAnalysisServices.cs - .NET llama a Python
public async Task<DetectionResult> ProcessVideoAsync(string videoPath)
{
    var process = new Process {
        StartInfo = new ProcessStartInfo {
            FileName = "python3",
            Arguments = $"process_video.py \"{videoPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
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
# python-scripts/process_video.py - YOLO v8
from ultralytics import YOLO
import cv2
import json
import sys

model = YOLO('ml-models/yolov8n.pt')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:  # 1 de cada 5 frames
            results = model(frame, classes=[0])  # class 0 = person
            players = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                players.append({
                    'x': float((x1 + x2) / 2),
                    'y': float((y1 + y2) / 2),
                    'confidence': float(box.conf[0])
                })
            detections.append({
                'frame': frame_count,
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC),
                'players': players
            })

        frame_count += 1

    cap.release()
    return {'detections': detections, 'total_frames': frame_count}

if __name__ == "__main__":
    result = process_video(sys.argv[1])
    print(json.dumps(result))
```

---

## 📁 Estructura del Proyecto

```
AnalizadorPadel/
├── backend/
│   ├── src/
│   │   └── AnalizadorPadel.Api/
│   │       ├── Program.cs              # Minimal APIs, configuración
│   │       ├── AnalizadorPadel.Api.csproj
│   │       ├── appsettings.json
│   │       ├── openapi.json            # Especificación OpenAPI
│   │       ├── Data/
│   │       │   └── PadelDbContext.cs   # EF Core DbContext
│   │       ├── Models/
│   │       │   ├── Entities/           # VideoEntity, AnalysisEntity
│   │       │   └── DTOs/               # ApiDtos.cs
│   │       └── Services/
│   │           └── VideoAnalysisServices.cs
│   └── tests/
│       └── AnalizadorPadel.Api.Tests/
│           ├── Unit/                   # Tests unitarios
│           │   └── Services/
│           ├── Integration/            # Tests de integración
│           └── BDD/                    # SpecFlow
│               ├── Features/
│               └── StepDefinitions/
├── frontend/
│   ├── src/
│   │   ├── components/                 # Componentes React
│   │   │   ├── Layout.tsx
│   │   │   └── VideoPlayer/
│   │   ├── pages/                      # Páginas
│   │   │   ├── Dashboard/
│   │   │   ├── Videos.tsx
│   │   │   └── Analyses.tsx
│   │   ├── services/
│   │   │   └── api/
│   │   │       ├── openapi.json
│   │   │       └── generated/          # Tipos TypeScript generados
│   │   ├── store/                      # Zustand stores
│   │   └── test/                       # Configuración de tests
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── python-scripts/
│   ├── process_video.py                # Script principal de procesamiento
│   ├── requirements.txt
│   └── tests/                          # Tests de Python
├── e2e/                                # Tests E2E con Playwright
│   ├── tests/
│   │   └── video-upload.spec.ts
│   ├── playwright.config.ts
│   └── package.json
├── infrastructure/
│   ├── Dockerfile.api                  # Backend + Python
│   ├── Dockerfile.frontend             # Nginx + React build
│   └── nginx.conf                      # Configuración Nginx
├── docker-compose.yml                  # Orquestación de contenedores
├── ml-models/                          # Modelos ML (YOLO)
├── spikes/                             # Investigaciones y spikes
├── docs/                               # Documentación
│   ├── ADR/                            # Architecture Decision Records
│   ├── PRODUCT.md
│   ├── TECHNICAL.md
│   └── PLANNING.md
├── scripts/                            # Scripts de utilidad
└── README.md
```

---

## 🔧 Principios de Diseño Modular

### Backend — Inyección de Dependencias

```csharp
// Program.cs
builder.Services.AddScoped<VideoService>();
builder.Services.AddScoped<AnalysisService>();
```

### Frontend — Interfaces Desacopladas

```typescript
// Servicios tipados con OpenAPI
import type { paths } from './services/api/generated/types';

export type VideoDto = paths['/api/videos']['get']['responses']['200']['content']['application/json']['data'][number];
export type AnalysisDto = paths['/api/analyses/{id}']['get']['responses']['200']['content']['application/json'];
```

### Preparación para Escalar

| Componente Actual | Reemplazo Futuro |
|---|---|
| `VideoService` (sync) | `AsyncVideoProcessingService` |
| SQLite | PostgreSQL |
| Python subprocess | FastAPI microservicio |
| Local storage | AWS S3 / Azure Blob |
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
    VideoNotFound = 4006,
    AnalysisNotFound = 4007,

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
    public string? Detail { get; set; }
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
    "timestamp": "2026-03-21T14:30:00Z",
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
// Configurado en Program.cs
app.UseExceptionHandler(errorApp =>
{
    errorApp.Run(async context =>
    {
        context.Response.StatusCode = 500;
        context.Response.ContentType = "application/json";
        var exception = context.Features.Get<IExceptionHandlerFeature>()?.Error;
        await context.Response.WriteAsJsonAsync(new
        {
            error = "An internal error occurred",
            requestId = context.TraceIdentifier
        });
    });
});
```

### Validaciones Preventivas

```csharp
public static class VideoValidation
{
    private static readonly string[] AllowedExtensions = { ".mp4", ".avi", ".mov", ".mkv" };
    private const int MaxFileSizeMB = 500;

    public static ValidationResult Validate(IFormFile file)
    {
        var extension = Path.GetExtension(file.FileName).ToLower();
        if (!AllowedExtensions.Contains(extension))
            return ValidationResult.Fail($"Formato no soportado. Use: {string.Join(", ", AllowedExtensions)}");

        if (file.Length > MaxFileSizeMB * 1024 * 1024)
            return ValidationResult.Fail($"El archivo excede {MaxFileSizeMB}MB");

        return ValidationResult.Success();
    }
}
```

---

## Configuración de Servidores

### Nginx (`infrastructure/nginx.conf`)

```nginx
server {
    listen 80;
    server_name localhost;

    # Frontend - React SPA
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Backend - .NET API
    location /api/ {
        proxy_pass http://api:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection keep-alive;
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        client_max_body_size 500M;
    }
}
```

### Docker Compose (`docker-compose.yml`)

```yaml
services:
  api:
    build:
      context: .
      dockerfile: infrastructure/Dockerfile.api
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
      - ConnectionStrings__DefaultConnection=Data Source=/app/data/padel.db
    volumes:
      - padel-data:/app/data
      - padel-uploads:/app/uploads
    ports:
      - "5001:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: infrastructure/Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      api:
        condition: service_healthy

volumes:
  padel-data:
  padel-uploads:
  padel-outputs:
  padel-logs:
```

---

## 🧪 Estrategia de Testing

### Enfoque Híbrido: BDD + TDD + E2E

| Tipo | Herramienta | Uso |
|------|-------------|-----|
| **BDD** | SpecFlow | Features de usuario (User Stories) |
| **TDD** | xUnit + FluentAssertions | Lógica de negocio |
| **Integration** | WebApplicationFactory | Tests de integración API |
| **Frontend** | Vitest + React Testing Library | Componentes React |
| **E2E** | Playwright | Flujos completos de usuario |
| **Mocking** | Moq / MSW | Dependencias en tests |

### Herramientas

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| xUnit | 2.9.x | Framework de testing principal (.NET) |
| SpecFlow | 3.9.x | BDD con Gherkin |
| FluentAssertions | 8.x | Assertions expresivas |
| Moq | 4.20.x | Mock objects |
| Vitest | 4.x | Framework de testing (Frontend) |
| React Testing Library | 16.x | Testing de componentes React |
| MSW | 2.x | Mock Service Worker (API mocking) |
| Playwright | 1.x | Tests E2E |

### Estructura de Tests

```
backend/tests/AnalizadorPadel.Api.Tests/
├── Unit/                          # TDD - Tests unitarios
│   └── Services/
│       ├── VideoServiceTests.cs
│       └── AnalysisServiceTests.cs
├── Integration/                   # Tests de integración
│   ├── VideoEndpointsTests.cs
│   └── AnalysisEndpointsTests.cs
├── BDD/                           # SpecFlow
│   ├── Features/
│   │   ├── US-1-SubirVideo.feature
│   │   └── US-2-VerEstadisticas.feature
│   └── StepDefinitions/
│       ├── VideoSteps.cs
│       └── AnalysisSteps.cs
└── Infrastructure/
    ├── CustomWebApplicationFactory.cs
    └── TestBase.cs

frontend/src/
├── components/
│   ├── Layout.test.tsx
│   └── VideoPlayer/
│       └── VideoPlayer.test.tsx
├── pages/Dashboard/
│   └── Dashboard.test.tsx
└── test/
    ├── setup.ts
    └── mocks/
        ├── handlers.ts
        └── server.ts

e2e/
├── tests/
│   └── video-upload.spec.ts
└── playwright.config.ts
```

### Ejemplo: Feature BDD (Gherkin)

```gherkin
Feature: Subida de Videos (US-1)
    Como jugador de pádel
    Quiero subir un video de mi partido
    Para obtener un análisis automático de mi juego

    @smoke
    Scenario: Video válido se sube exitosamente
        Given que tengo un video válido de pádel
        When subo el video al sistema
        Then el video se guarda correctamente
        And recibo una confirmación con el ID del video

    @validation
    Scenario: Video muy grande muestra error
        Given que tengo un video de 750MB
        When intento subir el video
        Then recibo un error indicando que el archivo excede el límite
```

### Ejemplo: Step Definitions (SpecFlow)

```csharp
[Binding]
public class VideoSteps
{
    private readonly HttpClient _client;
    private HttpResponseMessage _response;

    public VideoSteps(CustomWebApplicationFactory factory)
    {
        _client = factory.CreateClient();
    }

    [Given("que tengo un video válido de pádel")]
    public void GivenQueTengoVideoValido()
    {
        // Setup
    }

    [When("subo el video al sistema")]
    public async Task WhenSuboElVideo()
    {
        var content = new MultipartFormDataContent();
        content.Add(new StreamContent(File.OpenRead("test.mp4")), "file", "test.mp4");
        _response = await _client.PostAsync("/api/videos", content);
    }

    [Then("el video se guarda correctamente")]
    public void ThenElVideoSeGuarda()
    {
        _response.StatusCode.Should().Be(HttpStatusCode.Created);
    }
}
```

### Ejemplo: Test TDD (xUnit + FluentAssertions)

```csharp
public class VideoServiceTests
{
    private readonly VideoService _sut;
    private readonly Mock<PadelDbContext> _dbContextMock;

    public VideoServiceTests()
    {
        _dbContextMock = new Mock<PadelDbContext>();
        _sut = new VideoService(_dbContextMock.Object);
    }

    [Fact]
    public async Task GetVideoById_WithExistingId_ShouldReturnVideo()
    {
        // Arrange
        var videoId = Guid.NewGuid();
        var expectedVideo = new VideoEntity { Id = videoId, OriginalFileName = "test.mp4" };
        _dbContextMock.Setup(x => x.Videos.FindAsync(videoId))
            .ReturnsAsync(expectedVideo);

        // Act
        var result = await _sut.GetVideoByIdAsync(videoId);

        // Assert
        result.Should().NotBeNull();
        result!.Id.Should().Be(videoId);
        result.OriginalFileName.Should().Be("test.mp4");
    }
}
```

### Ejemplo: Test Frontend (Vitest)

```typescript
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Layout } from './Layout';

describe('Layout', () => {
    it('should render navigation links', () => {
        render(<Layout />);

        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Videos')).toBeInTheDocument();
        expect(screen.getByText('Analyses')).toBeInTheDocument();
    });
});
```

### Ejemplo: Test E2E (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test.describe('Video Upload E2E', () => {
    test('user can upload a video', async ({ page }) => {
        await page.goto('/videos');
        await page.getByText('Upload Video').click();

        const fileInput = page.locator('input[type="file"]');
        await fileInput.setInputFiles('test-data/sample.mp4');

        await page.getByText('Submit').click();

        await expect(page.getByText('Video uploaded successfully')).toBeVisible();
    });
});
```

### Cobertura de Tests por User Story

| User Story | Escenarios BDD | Tests TDD | Tests E2E | Prioridad |
|------------|----------------|-----------|-----------|-----------|
| US-1: Subir Video | 5 | 8 | 2 | Alta |
| US-2: Ver Estadísticas | 5 | 6 | 2 | Alta |
| US-3: Descargar PDF | 3 | 2 | 1 | Media |
| US-4: Historial | 3 | 4 | 1 | Media |
| US-5: Monitorear | 4 | 3 | 1 | Alta |

### Comandos de Ejecución

```bash
# Backend tests
dotnet test backend/AnalizadorPadel.sln

# Frontend tests
cd frontend && npm test

# E2E tests
cd e2e && npx playwright test

# All tests
./scripts/validate.sh
```

---

## 📚 Documentación Adicional

- **[PRODUCT.md](PRODUCT.md)** - Especificación de producto y User Stories
- **[PLANNING.md](PLANNING.md)** - Planificación y roadmap del proyecto
- **[ADR/README.md](ADR/README.md)** - Architecture Decision Records
  - ADR-001: Stack tecnológico backend
  - ADR-002: Estrategia de testing
  - ADR-003: API First
  - ADR-004: Estructura del proyecto
  - ADR-005: Reorganización de carpetas

---

*Última actualización: 21 de Marzo 2026*
