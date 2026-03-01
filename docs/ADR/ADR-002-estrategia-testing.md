# ADR-002: Estrategia de Testing BDD+TDD

## Estado

Aceptado

## Contexto

Se necesita definir una estrategia de testing para el proyecto que:

- Valide las funcionalidades desde la perspectiva del usuario
- Asegure la calidad del código de negocio
- Permita desarrollo paralelo entre equipos
- Genere documentación viva del sistema

### Restricciones

- Ya existen User Stories definidas en PRODUCT.md
- El equipo tiene experiencia con xUnit
- Se requiere documentación actualizada automáticamente

## Decisión

Se adopta un **enfoque híbrido BDD + TDD + E2E**:

- **BDD con SpecFlow** para features de usuario (User Stories)
- **TDD con xUnit + FluentAssertions** para lógica de negocio
- **Playwright** para tests E2E automatizados

### Herramientas

| Herramienta | Propósito |
|-------------|-----------|
| SpecFlow | BDD con Gherkin |
| xUnit | Framework de testing unitario e integración |
| FluentAssertions | Assertions expresivas |
| Moq | Mock objects |
| Playwright | Tests E2E automatizados |

### Stack de Testing para MVP

| Tipo | Cuándo ejecutarlo | Cobertura |
|------|-------------------|----------|
| **Unit** (xUnit) | Cada commit | Lógica de negocio |
| **Integración** (xUnit) | Cada commit | Endpoints API |
| **BDD** (SpecFlow) | Cada commit | Features de usuario |
| **E2E** (Playwright) | CI/CD (cada PR) | Flujos completos UI |

## Consecuencias

### Positivas
- User Stories se traducen directamente a escenarios Gherkin
- SpecFlow genera documentación viva
- Tests más legibles para stakeholders no técnicos
- Separación clara entre tests de usuario y tests unitarios
- Tests E2E con Playwright permiten automatización completa
- Cobertura integral: Unit + Integración + BDD + E2E

### Negativas
- Setup inicial más complejo que solo TDD
- Requiere aprender sintaxis Gherkin
- Mayor tiempo de desarrollo en etapas iniciales
- Playwright requiere configuración adicional

### Neutrales
- Aumenta la cantidad de archivos en el proyecto
- Necesita CI/CD configurado para E2E

## Alternativas Consideradas

| Alternativa | Pros | Contras | ¿Por qué se descartó? |
|-------------|------|---------|----------------------|
| **Solo TDD** | Más simple, rápido inicio | No genera docs, no alineado con User Stories | No aprovecha User Stories existentes |
| **Solo BDD** | Documentación automática | Overkill para unit tests, más lento | No ideal para lógica de negocio |
| **NUnit + BDDfy** | NUnit es familiar | BDDfy menos integrado con .NET | SpecFlow tiene mejor tooling |

## Referencias

- [SpecFlow Documentation](https://docs.specflow.org/)
- [PLANNING.md - Estrategia de Testing](../PLANNING.md)

---

*Fecha: 2026-02-18*
*Autores: Equipo de Arquitectura*