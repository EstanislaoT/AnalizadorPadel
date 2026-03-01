# ADR-003: API First

## Estado

Aceptado (actualizado: 2026-03-01)

## Contexto

Se necesita definir el enfoque de desarrollo de la API para el proyecto que:

- Permita desarrollo paralelo entre frontend y backend
- Asegure consistencia en el contrato de la API
- Genere documentación actualizada automáticamente
- Facilite el mocking para desarrollo independiente

### Restricciones

- El frontend (React) y backend (.NET) son equipos potencialmente separados
- Se requiere documentación de API para terceros
- Los cambios en la API deben ser versionados y documentados

## Decisión

Se adopta el enfoque **API First** con las siguientes herramientas:

| Herramienta | Propósito |
|-------------|-----------|
| **OpenAPI 3.0** | Especificación de API |
| **Scalar** | UI de documentación (reemplaza Swagger) |
| **openapi-typescript-codegen** | Generación de cliente TypeScript |
| **Prism** | Mock server para desarrollo frontend |

> **Nota técnica:** Se usa **Minimal APIs** de .NET 10 en lugar de Controllers MVC. Esto permite API más ligera sin boilerplate de controladores.

### Workflow recomendado

#### API-First puro

```
1. Diseñar API en OpenAPI (openapi.yaml)
2. Generar cliente TypeScript (openapi-typescript-codegen)
3. Mock server para frontend (Prism)
4. Escribir Minimal APIs que cumplan el spec
5. Frontend consume API tipada
```

#### Code-First pragmático (para equipos pequeños)

```
1. Escribir Minimal APIs directamente
2. Generar spec automáticamente (Microsoft.AspNetCore.OpenApi)
3. Generar cliente TypeScript (openapi-typescript-codegen)
4. Documentación siempre actualizada via Scalar
```

### Minimal APIs vs Controllers

| Aspecto | Minimal APIs | Controllers MVC |
|---------|--------------|-----------------|
| Boilerplate | Bajo | Alto |
| Attributes | Mínimo | Muchos |
| Generación auto | OpenAPI.NET | NSwag |
| Mejor para | Microservices, APISimple | Aplicacionesgrandes |

## Consecuencias

### Positivas
- Frontend puede desarrollar con mocks sin esperar al backend
- Contrato de API definido antes de implementación (API-First)
- Documentación Scalar automática y siempre actualizada
- Cliente TypeScript generado automáticamente con tipos
- Menos fricción entre equipos frontend y backend

### Negativas
- Tiempo adicional en la Fase 1 para diseñar la especificación (API-First)
- Requiere mantener openapi.yaml sincronizado si se usa API-First puro
- Curva de aprendizaje para OpenAPI spec

### Neutrales
- Introduce un paso adicional en el flujo de desarrollo

## Alternativas Consideradas

| Alternativa | Pros | Contras | ¿Por qué se descarta? |
|-------------|------|---------|----------------------|
| **Code First tradicional** | Más rápido al inicio | Documentación desactualizada, frontend espera backend | No permite desarrollo paralelo |
| **NSwag** | Genera servidores y clientes | Overkill para Minimal APIs, genera Controllers | No se adapta bien a .NET 10 |
| **GraphQL** | Flexible, una sola endpoint | Overkill para MVP, más complejo | No se alinea con requisitos simples |
| **gRPC** | Muy eficiente, tipado fuerte | No browser-friendly, más complejo | No ideal para frontend React |

## Stack recomendado

```
Backend: .NET 10 + Minimal APIs
Spec: OpenAPI 3.0
UI: Scalar
Cliente TS: openapi-typescript-codegen
Mock: Prism
```

## Referencias

- [OpenAPI Specification](https://swagger.io/specification/)
- [TECHNICAL.md - Especificación OpenAPI](../TECHNICAL.md)
- [Scalar for .NET](https://aka.ms/scalar)

---

*Fecha: 2026-02-18*
*Actualizado: 2026-03-01*
*Autores: Equipo de Arquitectura*
