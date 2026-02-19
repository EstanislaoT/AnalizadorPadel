# 📦 Producto — Analizador de Pádel

## Overview

Aplicación web para análisis de partidos de pádel mediante procesamiento de video. Los usuarios suben videos de sus partidos y obtienen estadísticas detalladas, análisis de movimiento y recomendaciones de mejora.

**Público objetivo**: Jugadores de pádel amateur y semi-profesional que quieran mejorar su juego con datos objetivos.

---

## 🎯 Funcionalidades del MVP

### 1. Subida de Videos
- [ ] Interfaz drag & drop para subir videos
- [ ] Soporte para formatos: MP4, AVI, MOV
- [ ] Límite de tamaño: 500MB
- [ ] Barra de progreso de subida
- [ ] Vista previa del video antes de procesar
- [ ] Validación de formato y duración mínima (1 minuto)

**Criterios de Aceptación:**

| ID | Criterio | Condición de Éxito |
|----|----------|-------------------|
| AC-1.1 | Validación de formato | Solo acepta archivos MP4, AVI, MOV |
| AC-1.2 | Validación de tamaño | Rechaza archivos > 500MB con mensaje claro |
| AC-1.3 | Validación de duración | Rechaza videos < 1 minuto con mensaje explicativo |
| AC-1.4 | Barra de progreso | Muestra progreso exacto (%) durante la subida |
| AC-1.5 | Vista previa | Permite reproducir video antes de confirmar procesamiento |
| AC-1.6 | Feedback de error | Muestra mensaje específico si la subida falla |
| AC-1.7 | Drag & drop | Permite arrastrar archivos o usar botón de selección |

### 2. Procesamiento de Video
- [ ] Detección de jugadores en la cancha
- [ ] Seguimiento de la pelota
- [ ] Análisis de movimiento básico
- [ ] Extracción de fotogramas clave
- [ ] Procesamiento con feedback en tiempo real

**Criterios de Aceptación:**

| ID | Criterio | Condición de Éxito |
|----|----------|-------------------|
| AC-2.1 | Detección de jugadores | Identifica ≥ 3 de 4 jugadores en ≥ 85% de frames |
| AC-2.2 | Seguimiento de pelota | Detecta pelota en ≥ 70% de frames (condiciones óptimas) |
| AC-2.3 | Feedback en tiempo real | Muestra estado: "Procesando: 25%" |
| AC-2.4 | Timeout | Cancela procesamiento si supera 10 minutos |
| AC-2.5 | Extracción de frames | Genera fotogramas clave cada punto |

### 3. Análisis y Estadísticas
- [ ] Tiempo total del partido
- [ ] Número de puntos jugados
- [ ] Heatmap de movimiento en la cancha
- [ ] Estadísticas de posición (red, fondo, laterales)
- [ ] Velocidad de desplazamiento
- [ ] Distancia total recorrida

**Criterios de Aceptación:**

| ID | Criterio | Condición de Éxito |
|----|----------|-------------------|
| AC-3.1 | Tiempo total | Muestra duración exacta del partido (error < 5%) |
| AC-3.2 | Conteo de puntos | Cuenta puntos con precisión ≥ 90% |
| AC-3.3 | Heatmap | Genera visualización de movimiento por jugador |
| AC-3.4 | Posiciones | Muestra % tiempo en red/fondo/laterales |
| AC-3.5 | Velocidad | Muestra velocidad promedio y máxima |
| AC-3.6 | Distancia | Muestra distancia total recorrida en metros |

### 4. Interfaz de Usuario
- [ ] Dashboard principal con videos recientes
- [ ] Historial de análisis previos
- [ ] Visualización interactiva de resultados
- [ ] Descarga de reportes (PDF)

**Criterios de Aceptación:**

| ID | Criterio | Condición de Éxito |
|----|----------|-------------------|
| AC-4.1 | Dashboard | Muestra últimos 10 videos subidos |
| AC-4.2 | Historial | Permite ver análisis anteriores |
| AC-4.3 | Visualización | Gráficos interactivos con tooltips |
| AC-4.4 | Reporte PDF | Descarga PDF con estadísticas completas |

---

## 🔜 Features Secundarias (Post-MVP)

- [ ] Sistema de autenticación de usuarios
- [ ] Perfil de usuario
- [ ] Detección de tipo de golpes (derecha, revés, volea, smash)
- [ ] Análisis de técnica
- [ ] Comparación entre partidos
- [ ] Recomendaciones de mejora personalizadas
- [ ] Modo entrenamiento
- [ ] Integración con wearables

---

## 📊 Métricas de Éxito

### Técnicas
| Métrica | Objetivo |
|---|---|
| Tiempo de procesamiento | < 5 min por cada 10 min de video |
| Precisión detección jugadores | > 90% |
| Tiempo de respuesta API | < 200ms |
| Uptime | 99.5% |

### de Usuario
| Métrica | Objetivo |
|---|---|
| Tiempo de subida de video | < 2 minutos |
| Tiempo hasta primeros resultados | < 10 minutos |

---

## 👤 User Stories

### US-1: Subir un Video de Partido

**Como** jugador de pádel  
**Quiero** subir un video de mi partido  
**Para** obtener un análisis automático de mi juego  

**Criterios de Aceptación:**
- [ ] Puedo arrastrar un archivo MP4 al área de subida
- [ ] Veo una barra de progreso durante la subida
- [ ] Recibo un mensaje si el formato no es válido
- [ ] Recibo un mensaje si el archivo es muy grande (>500MB)
- [ ] Puedo ver una vista previa del video antes de procesarlo

**Escenario de Prueba:**
```
GIVEN que estoy en la página de subida
WHEN arrastro un video válido de 100MB
THEN veo la barra de progreso completar al 100%
AND puedo hacer clic en "Procesar"
```

---

### US-2: Ver Estadísticas del Partido

**Como** jugador de pádel  
**Quiero** ver las estadísticas de mi partido  
**Para** conocer mi rendimiento y áreas de mejora  

**Criterios de Aceptación:**
- [ ] Puedo ver el tiempo total del partido
- [ ] Puedo ver el número de puntos jugados
- [ ] Puedo ver un heatmap de mi movimiento
- [ ] Puedo ver mi velocidad promedio y máxima
- [ ] Puedo ver la distancia total que recorrí

**Escenario de Prueba:**
```
GIVEN que tengo un análisis completado
WHEN accedo a la página de resultados
THEN veo todas las estadísticas del partido
AND puedo interactuar con los gráficos
```

---

### US-3: Descargar Reporte PDF

**Como** jugador de pádel  
**Quiero** descargar un reporte PDF  
**Para** compartir el análisis con mi compañero o entrenador  

**Criterios de Aceptación:**
- [ ] Hay un botón de "Descargar PDF" visible
- [ ] El PDF se genera en menos de 30 segundos
- [ ] El PDF incluye todas las estadísticas principales
- [ ] El PDF incluye el heatmap visual

**Escenario de Prueba:**
```
GIVEN que tengo un análisis completado
WHEN hago clic en "Descargar PDF"
THEN se descarga un archivo PDF con el reporte
AND puedo abrirlo en cualquier visor de PDF
```

---

### US-4: Revisar Historial de Análisis

**Como** jugador de pádel  
**Quiero** ver mis análisis anteriores  
**Para** comparar mi progreso entre partidos  

**Criterios de Aceptación:**
- [ ] Veo una lista de mis últimos 10 análisis en el dashboard
- [ ] Cada item muestra: fecha, duración del video, estado
- [ ] Puedo hacer clic en un análisis anterior para ver los resultados
- [ ] Los análisis se ordenan por fecha (más reciente primero)

**Escenario de Prueba:**
```
GIVEN que tengo múltiples análisis realizados
WHEN accedo al dashboard
THEN veo la lista de análisis ordenados por fecha
AND puedo acceder a cualquier análisis anterior
```

---

### US-5: Monitorear Procesamiento

**Como** jugador de pádel  
**Quiero** ver el progreso del procesamiento  
**Para** saber cuánto falta para ver los resultados  

**Criterios de Aceptación:**
- [ ] Veo un indicador de estado: "Subiendo", "Procesando", "Completado"
- [ ] Durante procesamiento veo porcentaje de progreso
- [ ] Si el procesamiento falla, veo un mensaje de error claro
- [ ] Puedo cancelar el procesamiento antes de que complete

**Escenario de Prueba:**
```
GIVEN que inicié el procesamiento de un video
WHEN está en progreso
THEN veo "Procesando: 45%" actualizado en tiempo real
AND cuando termina veo "Completado"
```

---

## 🔄 Roadmap

### V1.0 — MVP (Actual)
- Subida y almacenamiento local de videos
- Procesamiento síncrono básico
- Detección de jugadores con YOLO v8
- Estadísticas básicas y heatmaps
- Sin autenticación

### V2.0 — Post-MVP
- Autenticación de usuarios (JWT)
- Procesamiento asíncrono
- Almacenamiento en S3
- Detección de pelota mejorada (TrackNet)
- Análisis de técnica de golpeo
- Comparación entre partidos
- Recomendaciones de mejora

### V3.0 — Futuro
- IA para recomendaciones personalizadas
- Coach virtual
- Torneos virtuales
- Comunidad y rankings
- Integración con wearables
- CDN para distribución de contenido

---

*Última actualización: 18 de Febrero 2026*
