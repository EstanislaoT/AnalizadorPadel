# 📋 Architecture Decision Records (ADRs)

Este directorio contiene los registros de decisiones arquitectónicas del proyecto Analizador de Pádel.

---

## ¿Qué es un ADR?

Un Architecture Decision Record (ADR) es un documento que captura una decisión arquitectónica importante junto con su contexto y consecuencias.

### Estructura de un ADR

```markdown
# Título de la Decisión

## Estado
[Propuesto | Aceptado | Deprecado | Supersede]

## Contexto
¿Cuál es el problema que se está resolviendo?

## Decisión
¿Cuál es la decisión tomada?

## Consecuencias
¿Qué efectos tiene esta decisión?

## Alternativas Consideradas
¿Qué otras opciones se evaluaron?
```

---

## Índice de ADRs

| Número | Título | Estado | Fecha |
|--------|--------|--------|-------|
| ADR-001 | Stack Tecnológico Backend | Aceptado | 2026-02-18 |
| ADR-002 | Estrategia de Testing BDD+TDD | Aceptado | 2026-02-18 |
| ADR-003 | API First | Aceptado | 2026-02-18 |
| ADR-004 | Estructura del Proyecto | Propuesto | 2026-02-28 |
| ADR-005 | Reorganización Carpeta Runs | Aceptado | 2026-02-28 |

---

## Cómo crear un nuevo ADR

1. Copiar `template.md` a `ADR-XXX-titulo-descriptivo.md`
2. Completar las secciones
3. Agregar al índice en este README
4. Commit con mensaje: `docs(adr): agregar ADR-XXX`

---

*Última actualización: 18 de Febrero 2026*