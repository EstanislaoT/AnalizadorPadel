# Plan de Validación Automatizada - AnalizadorPadel

## Resumen Ejecutivo
Implementar una suite completa de tests automatizados para validar el funcionamiento del MVP, cubriendo backend (API REST), frontend (componentes React) y procesamiento Python. Incluir tests de integración, unitarios y un script de validación end-to-end.

---

## Alcance

### Incluido
- Tests de integración para endpoints API (.NET)
- Tests de componentes para UI (React + Vitest)
- Tests unitarios para procesamiento Python
- Script de validación E2E con flujo completo
- Configuración de cobertura de código
- Tests BDD con SpecFlow

### Excluido
- Tests de UI end-to-end con Playwright/Cypress (futura fase)
- Performance testing
- Stress testing

---

## Estructura de Tareas

### Fase 1: Setup Infraestructura de Testing

#### Tarea 1.1: Crear proyecto de tests .NET
**Archivos a modificar/crear:**
- Crear `backend/tests/AnalizadorPadel.Tests/AnalizadorPadel.Tests.csproj`
- Crear `backend/tests/AnalizadorPadel.Tests/Integration/`
- Modificar `backend/AnalizadorPadel.sln` (agregar proyecto)

**Paquetes NuGet a instalar:**
```xml
<PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.13.0" />
<PackageReference Include="xunit" Version="2.9.3" />
<PackageReference Include="xunit.runner.visualstudio" Version="3.0.2" />
<PackageReference Include="FluentAssertions" Version="8.1.1" />
<PackageReference Include="Microsoft.AspNetCore.Mvc.Testing" Version="8.0.0" />
<PackageReference Include="Microsoft.EntityFrameworkCore.InMemory" Version="10.0.0" />
<PackageReference Include="SpecFlow.xUnit" Version="3.9.74" />
<PackageReference Include="SpecFlow.Plus.LivingDocPlugin" Version="3.9.57" />
```

**Comandos:**
```bash
cd backend
dotnet new xunit -n AnalizadorPadel.Tests -o tests/AnalizadorPadel.Tests
dotnet sln add tests/AnalizadorPadel.Tests/AnalizadorPadel.Tests.csproj
cd tests/AnalizadorPadel.Tests
dotnet add package FluentAssertions
dotnet add package Microsoft.AspNetCore.Mvc.Testing
dotnet add package Microsoft.EntityFrameworkCore.InMemory
dotnet add package SpecFlow.xUnit
```

#### Tarea 1.2: Configurar WebApplicationFactory
**Archivo:** `backend/tests/AnalizadorPadel.Tests/Integration/CustomWebApplicationFactory.cs`

**Requisitos:**
- Configurar SQLite en memoria para tests
- Configurar Serilog para modo test (logs mínimos)
- Configurar HttpClient con base URL
- Crear helper para seed de datos

#### Tarea 1.3: Configurar Vitest en Frontend
**Archivos a modificar/crear:**
- Crear `frontend/vitest.config.ts`
- Crear `frontend/src/test/setup.ts`
- Modificar `frontend/package.json` (agregar scripts)

**Paquetes npm a instalar:**
```bash
cd frontend
npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event msw jsdom @vitest/coverage-v8
```

**Scripts a agregar en package.json:**
```json
{
  "test": "vitest",
  "test:ui": "vitest --ui",
  "coverage": "vitest run --coverage"
}
```

#### Tarea 1.4: Configurar Pytest para Python
**Archivos a modificar/crear:**
- Crear `python-scripts/tests/__init__.py`
- Crear `python-scripts/tests/conftest.py`
- Crear `python-scripts/pytest.ini`

**Comandos:**
```bash
cd python-scripts
pip install pytest pytest-asyncio pytest-cov
```

**Configuración pytest.ini:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=scripts --cov-report=term-missing
```

---

### Fase 2: Tests de Integración Backend

#### Tarea 2.1: Tests de Videos API
**Archivo:** `backend/tests/AnalizadorPadel.Tests/Integration/VideosControllerTests.cs`

**Tests a implementar:**
| Test | Endpoint | Expectativa |
|------|----------|-------------|
| `PostVideo_ValidFile_Returns201` | POST /api/videos | Status 201, retorna videoId |
| `PostVideo_InvalidExtension_Returns400` | POST /api/videos | Status 400, mensaje error |
| `PostVideo_EmptyFile_Returns400` | POST /api/videos | Status 400 |
| `GetVideos_ReturnsList` | GET /api/videos | Status 200, array no vacío |
| `GetVideoById_Exists_Returns200` | GET /api/videos/{id} | Status 200, metadata correcta |
| `GetVideoById_NotFound_Returns404` | GET /api/videos/{id} | Status 404 |
| `DeleteVideo_Exists_Returns204` | DELETE /api/videos/{id} | Status 204 |
| `GetVideoStream_WithRange_Returns206` | GET /api/videos/{id}/stream | Status 206, headers Range |

**Video de prueba:** Usar `test-videos/PadelPro3.mp4` (5.9MB)

#### Tarea 2.2: Tests de Análisis API
**Archivo:** `backend/tests/AnalizadorPadel.Tests/Integration/AnalysisControllerTests.cs`

**Tests a implementar:**
| Test | Endpoint | Expectativa |
|------|----------|-------------|
| `PostAnalysis_ValidVideo_Returns202` | POST /api/videos/{id}/analyse | Status 202, retorna analysisId |
| `PostAnalysis_VideoNotFound_Returns404` | POST /api/videos/{id}/analyse | Status 404 |
| `PostAnalysis_AlreadyProcessing_Returns409` | POST /api/videos/{id}/analyse | Status 409 |
| `GetAnalysis_Exists_Returns200` | GET /api/analyses/{id} | Status 200, estado correcto |
| `GetAnalysisStats_ReturnsStructure` | GET /api/analyses/{id}/stats | Status 200, schema válido |
| `GetAnalysisHeatmap_ReturnsData` | GET /api/analyses/{id}/heatmap | Status 200, array de puntos |

#### Tarea 2.3: Tests de Dashboard API
**Archivo:** `backend/tests/AnalizadorPadel.Tests/Integration/DashboardControllerTests.cs`

**Tests:**
- `GetDashboardStats_ReturnsAggregatedData`
- `GetHealth_Returns200AndHealthy`

---

### Fase 3: Tests de Componentes Frontend

#### Tarea 3.1: Configurar MSW (Mock Service Worker)
**Archivos:**
- Crear `frontend/src/test/mocks/handlers.ts`
- Crear `frontend/src/test/mocks/server.ts`
- Modificar `frontend/src/test/setup.ts`

**Mocks a implementar:**
- GET /api/videos → lista de videos mock
- GET /api/videos/{id} → video individual
- GET /api/analyses/{id} → análisis mock
- GET /api/dashboard/stats → estadísticas mock

#### Tarea 3.2: Tests de VideoPlayer
**Archivo:** `frontend/src/components/VideoPlayer/VideoPlayer.test.tsx`

**Tests:**
- `renders video element with correct src`
- `displays loading state initially`
- `displays error message on load failure`
- `calls onError when video fails to load`

#### Tarea 3.3: Tests de Layout y Navegación
**Archivo:** `frontend/src/components/Layout/Layout.test.tsx`

**Tests:**
- `renders navigation links`
- `navigates to dashboard on logo click`
- `highlights active route`

#### Tarea 3.4: Tests de Dashboard
**Archivo:** `frontend/src/pages/Dashboard/Dashboard.test.tsx`

**Tests:**
- `fetches and displays statistics on mount`
- `renders recent videos list`
- `renders recent analyses list`
- `displays loading state while fetching`

---

### Fase 4: Tests Python

#### Tarea 4.1: Tests de Procesamiento de Video
**Archivo:** `python-scripts/tests/test_process_video.py`

**Tests:**
- `test_detect_players_returns_detections`
- `test_process_video_creates_output_file`
- `test_process_video_handles_invalid_file`
- `test_calculate_statistics_returns_dict`

**Fixture:** Crear video de prueba pequeño (1 segundo, 640x480)

---

### Fase 5: Tests BDD con SpecFlow

#### Tarea 5.1: Feature Video Upload
**Archivo:** `backend/tests/AnalizadorPadel.Tests/BDD/Features/VideoUpload.feature`

```gherkin
Feature: Video Upload
  As a user
  I want to upload padel match videos
  So that I can analyze them later

  Scenario: Successfully upload a valid video
    Given I have a valid MP4 video file
    When I submit the video to the upload endpoint
    Then I should receive a 201 Created response
    And the response should contain a video ID
    And the video should be stored in the system

  Scenario: Upload fails with invalid file type
    Given I have a text file disguised as MP4
    When I submit the file to the upload endpoint
    Then I should receive a 400 Bad Request response
    And the error message should indicate invalid file type
```

**Step Definitions:** `backend/tests/AnalizadorPadel.Tests/BDD/Steps/VideoUploadSteps.cs`

#### Tarea 5.2: Feature Video Analysis
**Archivo:** `backend/tests/AnalizadorPadel.Tests/BDD/Features/VideoAnalysis.feature`

```gherkin
Feature: Video Analysis
  As a user
  I want to analyze uploaded videos
  So that I can get statistics about the match

  Scenario: Start analysis for existing video
    Given I have an uploaded video
    When I request analysis for the video
    Then I should receive a 202 Accepted response
    And the analysis status should be "Processing"

  Scenario: Get analysis results after completion
    Given I have a completed analysis
    When I request the analysis results
    Then I should receive the player positions
    And I should receive match statistics
```

---

### Fase 6: Script de Validación E2E

#### Tarea 6.1: Crear Script de Validación
**Archivo:** `scripts/validate.sh`

**Flujo del script:**
```bash
#!/bin/bash
set -e

echo "=== AnalizadorPadel Validation Script ==="

# 1. Verificar prerequisitos
echo "Checking prerequisites..."
dotnet --version
node --version
python3 --version

# 2. Iniciar backend en modo test
echo "Starting backend..."
cd backend
dotnet run --configuration Test &
BACKEND_PID=$!
sleep 5

# 3. Ejecutar tests backend
echo "Running backend tests..."
dotnet test tests/AnalizadorPadel.Tests --verbosity normal

# 4. Ejecutar tests frontend
echo "Running frontend tests..."
cd ../frontend
npm test -- --run

# 5. Ejecutar tests Python
echo "Running Python tests..."
cd ../python-scripts
pytest

# 6. Prueba E2E manual con curl
echo "Running E2E validation..."
./scripts/e2e-test.sh

# 7. Cleanup
kill $BACKEND_PID

echo "=== Validation Complete ==="
```

#### Tarea 6.2: Crear Prueba E2E con curl
**Archivo:** `scripts/e2e-test.sh`

**Pasos:**
1. Subir video: `curl -X POST -F "file=@test-videos/PadelPro3.mp4" http://localhost:5000/api/videos`
2. Iniciar análisis: `curl -X POST http://localhost:5000/api/videos/{id}/analyse`
3. Polling de estado: `curl http://localhost:5000/api/analyses/{id}` (repetir hasta completado)
4. Verificar dashboard: `curl http://localhost:5000/api/dashboard/stats`
5. Verificar streaming: `curl -H "Range: bytes=0-1023" http://localhost:5000/api/videos/{id}/stream`

---

## Dependencias entre Tareas

```
Fase 1.1 ─┬──────────────────────────────────────┐
          ├─> Fase 2.1, 2.2, 2.3 ─┬──────────────┤
Fase 1.2 ─┘                       ├─> Fase 5.1, 5.2
                                  │
Fase 1.3 ─> Fase 3.1 ─> 3.2, 3.3, 3.4
                                  │
Fase 1.4 ─> Fase 4.1              │
                                  │
Fase 6.1, 6.2 <───────────────────┴─ (depende de todo)
```

---

## Criterios de Éxito

1. ✅ Todos los tests de integración backend pasan (>90% cobertura de endpoints)
2. ✅ Todos los tests de componentes frontend pasan
3. ✅ Todos los tests Python pasan
4. ✅ Al menos 2 features BDD completadas con SpecFlow
5. ✅ Script de validación E2E ejecuta sin errores
6. ✅ Flujo completo (upload → process → view) funciona con video de prueba
7. ✅ Documentación de testing actualizada en docs/TESTING.md

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Tests de integración lentos | Media | Medio | Usar SQLite en memoria, paralelizar tests |
| Procesamiento de video tarda mucho en tests | Alta | Alto | Usar video de 1 segundo para tests, timeout de 30s |
| Dependencias de frontend cambian | Baja | Medio | Fijar versiones en package.json |
| SpecFlow tiene curva de aprendizaje | Media | Bajo | Documentar ejemplos claros, empezar con tests simples |

---

## Estimación de Esfuerzo

| Fase | Tiempo Estimado |
|------|-----------------|
| Fase 1: Setup | 2-3 horas |
| Fase 2: Tests Backend | 4-5 horas |
| Fase 3: Tests Frontend | 3-4 horas |
| Fase 4: Tests Python | 2 horas |
| Fase 5: Tests BDD | 3-4 horas |
| Fase 6: Script E2E | 2 horas |
| **Total** | **16-20 horas** |

---

## Comandos de Ejecución

### Ejecutar todos los tests:
```bash
./scripts/validate.sh
```

### Ejecutar tests individuales:
```bash
# Backend
cd backend
dotnet test tests/AnalizadorPadel.Tests

# Frontend
cd frontend
npm test

# Python
cd python-scripts
pytest

# Solo BDD
dotnet test --filter "FullyQualifiedName~BDD"
```

### Ver cobertura:
```bash
# Backend
dotnet test --collect:"XPlat Code Coverage"

# Frontend
npm run coverage

# Python
pytest --cov=scripts --cov-report=html
```

---

## Checklist de Implementación

- [ ] Fase 1.1: Proyecto de tests .NET creado
- [ ] Fase 1.2: WebApplicationFactory configurado
- [ ] Fase 1.3: Vitest configurado en frontend
- [ ] Fase 1.4: Pytest configurado para Python
- [ ] Fase 2.1: Tests de Videos API
- [ ] Fase 2.2: Tests de Análisis API
- [ ] Fase 2.3: Tests de Dashboard API
- [ ] Fase 3.1: MSW configurado
- [ ] Fase 3.2: Tests de VideoPlayer
- [ ] Fase 3.3: Tests de Layout
- [ ] Fase 3.4: Tests de Dashboard
- [ ] Fase 4.1: Tests Python de procesamiento
- [ ] Fase 5.1: Feature VideoUpload con SpecFlow
- [ ] Fase 5.2: Feature VideoAnalysis con SpecFlow
- [ ] Fase 6.1: Script validate.sh
- [ ] Fase 6.2: Script e2e-test.sh
- [ ] Documentación actualizada

---

## Notas Adicionales

- Usar `test-videos/PadelPro3.mp4` (5.9MB) como video de prueba principal
- Para tests que requieran video procesado, considerar crear mock del servicio Python
- Los tests BDD deben ser ejecutables en CI/CD (sin interfaz gráfica)
- Agregar badge de cobertura en README.md al finalizar
