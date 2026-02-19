# ADR-001: Stack Tecnológico Backend

## Estado

Aceptado

## Contexto

Se necesita definir el stack tecnológico para el backend del Analizador de Pádel. El sistema debe:

- Procesar videos de partidos de pádel
- Ejecutar modelos de ML (YOLO) para detección de jugadores
- Manejar almacenamiento de archivos grandes (videos)
- Proporcionar una API REST para el frontend
- Ser desplegable en contenedores Docker

### Restricciones

- El procesamiento de video es intensivo en CPU
- Se necesita integración con Python para modelos de ML
- El MVP debe ser simple pero escalable

## Decisión

Se adopta **ASP.NET Core 8** como framework principal del backend.

### Componentes del Stack

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Framework | ASP.NET Core 8 | LTS, alto rendimiento, ecosistema maduro |
| Base de Datos | PostgreSQL | Soporte JSONB, extensibilidad, open source |
| ORM | Entity Framework Core | Integración nativa con .NET |
| Procesamiento Video | FFmpeg + YOLO v8 (Python) | Mejor ecosistema para ML/Computer Vision |
| Contenedores | Docker + Debian Slim | Compatibilidad con librerías nativas |

## Consecuencias

### Positivas
- Alto rendimiento para procesamiento intensivo
- Tipado fuerte reduce errores en runtime
- Excelente soporte para APIs REST
- Integración con ecosistema Azure si se requiere en futuro
- Long Term Support hasta 2026

### Negativas
- Curva de aprendizaje para desarrolladores sin experiencia en .NET
- Integración con Python requiere subprocess
- Mayor uso de memoria que Node.js

### Neutrales
- Requiere equipo con conocimientos en múltiples lenguajes (C# y Python)

## Alternativas Consideradas

| Alternativa | Pros | Contras | ¿Por qué se descartó? |
|-------------|------|---------|----------------------|
| **Node.js + Express** | Mismo lenguaje que frontend, async nativo | Menor rendimiento en CPU-bound tasks | No ideal para procesamiento pesado |
| **Python FastAPI** | Nativo para ML, código unificado | Menor rendimiento general, ecosistema web menos maduro | No ideal como API server principal |
| **Go** | Muy rápido, compilado | Ecosistema ML más limitado | Mayor complejidad de desarrollo |

## Referencias

- [ASP.NET Core 8 Documentation](https://docs.microsoft.com/aspnet/core)
- [TECHNICAL.md - Stack Tecnológico](../TECHNICAL.md)

---

*Fecha: 2026-02-18*
*Autores: Equipo de Arquitectura*