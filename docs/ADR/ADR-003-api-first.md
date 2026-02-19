# ADR-003: API First

## Estado

Aceptado

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
| OpenAPI 3.0 | Especificación de API |
| Swashbuckle | Generación de Swagger en .NET |
| NSwag | Generación de cliente TypeScript |
| Prism | Mock server para desarrollo frontend |

### Workflow

```
1. Diseñar API en OpenAPI (openapi.yaml)
2. Generar servidor stub (NSwag)
3. Mock server para frontend (Prism)
4. Implementar lógica backend
5. Generar cliente TypeScript (NSwag)
6. Frontend consume API tipada
```

## Consecuencias

### Positivas
- Frontend puede desarrollar con mocks sin esperar al backend
- Contrato de API definido antes de implementación
- Documentación Swagger automática y siempre actualizada
- Cliente TypeScript generado automáticamente con tipos
- Menos fricción entre equipos frontend y backend

### Negativas
- Tiempo adicional en la Fase 1 para diseñar la especificación
- Requiere mantener openapi.yaml sincronizado
- Curva de aprendizaje para OpenAPI spec

### Neutrales
- Introduce un paso adicional en el flujo de desarrollo

## Alternativas Consideradas

| Alternativa | Pros | Contras | ¿Por qué se descartó? |
|-------------|------|---------|----------------------|
| **Code First** | Más rápido al inicio | Documentación desactualizada, frontend espera backend | No permite desarrollo paralelo |
| **GraphQL** | Flexible, una sola endpoint | Overkill para MVP, más complejo | No se alinea con requisitos simples |
| **gRPC** | Muy eficiente, tipado fuerte | No browser-friendly, más complejo | No ideal para frontend React |

## Referencias

- [OpenAPI Specification](https://swagger.io/specification/)
- [TECHNICAL.md - Especificación OpenAPI](../TECHNICAL.md)

---

*Fecha: 2026-02-18*
*Autores: Equipo de Arquitectura*