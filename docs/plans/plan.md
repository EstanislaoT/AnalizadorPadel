# Plan de Validacion Automatizada - Estado Actual

## Resumen

El backend ya no usa un proyecto dual de tests. La estrategia vigente se concentra en:

- `backend/tests/AnalizadorPadel.Api.Tests`

Ese proyecto contiene:
- tests unitarios
- tests de integracion
- tests BDD

## Estado Actual

### Backend

Proyecto vigente:
- `backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj`

Estado:
- estable
- `87/87` tests pasando

Comando principal:
```bash
dotnet test backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj
```

### Proyecto retirado

El proyecto `backend/tests/AnalizadorPadel.Tests` fue eliminado por redundancia y no debe volver a usarse como referencia de planificacion ni de ejecucion.

## Siguientes lineas de trabajo sugeridas

1. Mantener toda nueva cobertura backend dentro de `AnalizadorPadel.Api.Tests`.
2. Si se amplian features BDD, reusar la infraestructura actual de `Infrastructure/` y `StepDefinitions/`.
3. Si se documentan comandos de testing en otros archivos, usar siempre la ruta del proyecto vigente.

## Comandos utiles

```bash
# Suite completa
dotnet test backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj

# Unit
dotnet test backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj --filter FullyQualifiedName~Unit

# Integration
dotnet test backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj --filter FullyQualifiedName~Integration

# BDD
dotnet test backend/tests/AnalizadorPadel.Api.Tests/AnalizadorPadel.Api.Tests.csproj --filter FullyQualifiedName~BDD
```
