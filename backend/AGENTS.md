# Backend - Consideraciones Operativas

## ✅ Contexto Backend

- La API está implementada en ASP.NET Core con Minimal APIs y EF Core sobre SQLite.
- El proyecto principal vive en `backend/src/AnalizadorPadel.Api`.
- La suite principal vive en `backend/tests/AnalizadorPadel.Api.Tests`.

## ✅ Contratos y Persistencia

- Los endpoints principales cubren videos, análisis, dashboard y health check bajo `/api`.
- El backend persiste metadatos en SQLite y usa filesystem local para archivos de trabajo y salidas del procesamiento.
- El procesamiento de video delega en scripts Python con YOLO y actualiza luego el estado del análisis en la base de datos.

## ✅ Validación Backend

- Después de cambios del backend, validar al menos con `dotnet test backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj`.
- Para validar integración real con frontend o navegador, no alcanza con tests aislados; comprobar también el flujo HTTP observable.
- Antes de concluir que un puerto, binding o configuración está bloqueado, intentar el arranque real del servicio en el entorno actual.

## ✅ Operación Local

- Verificar siempre la combinación de `puerto backend`, CORS, rutas runtime y configuración consumida por otros componentes.
- Mantener separados código fuente y estado mutable de ejecución.
