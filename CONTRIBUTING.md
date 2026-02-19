# 🤝 Guía de Contribución

Este documento establece las convenciones y procesos para contribuir al proyecto Analizador de Pádel.

---

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Proceso de Desarrollo](#proceso-de-desarrollo)
- [Convenciones de Commits](#convenciones-de-commits)
- [Estrategia de Branches (Git Flow)](#estrategia-de-branches-git-flow)
- [Code Review Checklist](#code-review-checklist)
- [Proceso de Pull Request](#proceso-de-pull-request)

---

## Código de Conducta

- Ser respetuoso e inclusivo en todas las interacciones
- Aceptar críticas constructivas
- Enfocarse en lo que es mejor para el proyecto
- Mostrar empatía hacia otros miembros del equipo

---

## Proceso de Desarrollo

### Flujo de Trabajo

```
1. Crear issue → 2. Crear branch → 3. Desarrollar → 4. Crear PR → 5. Code Review → 6. Merge
```

### Responsabilidades

| Rol | Responsabilidad |
|-----|-----------------|
| **Developer** | Crear feature branch, implementar, tests, documentar |
| **Reviewer** | Revisar código, sugerir mejoras, aprobar/rechazar |
| **Maintainer** | Merge a main, releases, manejo de conflictos |

---

## Convenciones de Commits

### Formato

```
<tipo>(<scope>): <descripción>

[cuerpo opcional]

[footer(s) opcional]
```

### Tipos

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| `feat` | Nueva funcionalidad | `feat(videos): agregar endpoint de subida` |
| `fix` | Corrección de bug | `fix(processing): corregir timeout en YOLO` |
| `docs` | Documentación | `docs: actualizar README con nuevos endpoints` |
| `style` | Formato, no afecta lógica | `style: formatear código con prettier` |
| `refactor` | Refactorización | `refactor(services): extraer lógica de validación` |
| `test` | Agregar/modificar tests | `test(videos): agregar tests de integración` |
| `chore` | Tareas de mantenimiento | `chore: actualizar dependencias` |
| `perf` | Mejoras de rendimiento | `perf(processing): optimizar extracción de frames` |
| `ci` | Cambios en CI/CD | `ci: agregar job de testing` |

### Scopes

| Scope | Módulo |
|-------|--------|
| `videos` | Gestión de videos |
| `analyses` | Análisis de partidos |
| `processing` | Procesamiento de video |
| `ui` | Interfaz de usuario |
| `api` | API endpoints |
| `db` | Base de datos |
| `docker` | Configuración Docker |

### Reglas

1. **Idioma**: Los commits en español o inglés, pero consistentes
2. **Tamaño**: Máximo 72 caracteres en la descripción
3. **Tiempo verbal**: Usar imperativo ("agregar" no "agregado")
4. **Referencias**: Incluir issue number cuando aplique

### Ejemplos

```bash
# ✅ Buenos ejemplos
feat(videos): agregar validación de duración mínima
fix(processing): corregir memoria en procesamiento concurrente
docs(api): documentar endpoint de heatmap
test(analyses): agregar tests BDD para estadísticas

# ❌ Malos ejemplos
Fixed bug
update
WIP
asdfasdf
```

---

## Estrategia de Branches (Git Flow)

### Diagrama

```
main
│
├── develop
│   │
│   ├── feature/US-1-subida-videos
│   ├── feature/US-2-estadisticas
│   └── feature/US-3-pdf-report
│
├── release/v1.0.0
│
└── hotfix/critical-bug
```

### Branches Principales

| Branch | Descripción | Protegida |
|--------|-------------|-----------|
| `main` | Código en producción | ✅ Sí |
| `develop` | Código en desarrollo activo | ✅ Sí |

### Branches de Soporte

| Tipo | Formato | Ejemplo | Merge a |
|------|---------|---------|---------|
| **Feature** | `feature/<descripcion>` | `feature/US-1-subida-videos` | develop |
| **Bugfix** | `bugfix/<descripcion>` | `bugfix/fix-validation` | develop |
| **Release** | `release/v<version>` | `release/v1.0.0` | main + develop |
| **Hotfix** | `hotfix/<descripcion>` | `hotfix/critical-security` | main + develop |

### Reglas

1. **Features**: Siempre desde `develop` hacia `develop`
2. **Releases**: Desde `develop` hacia `main`
3. **Hotfixes**: Desde `main` hacia `main` + `develop`
4. **Nombres**: Usar kebab-case, incluir issue ID si aplica

### Comandos Comunes

```bash
# Crear feature branch
git checkout develop
git pull origin develop
git checkout -b feature/US-1-subida-videos

# Mantener branch actualizado
git fetch origin
git rebase origin/develop

# Finalizar feature
git checkout develop
git merge --no-ff feature/US-1-subida-videos
git push origin develop
git branch -d feature/US-1-subida-videos
```

---

## Code Review Checklist

### Checklist para Reviewers

#### ✅ Funcionalidad
- [ ] El código implementa la funcionalidad descrita en el issue/US
- [ ] Los criterios de aceptación están cubiertos
- [ ] Los edge cases están considerados

#### ✅ Calidad de Código
- [ ] Código legible y bien estructurado
- [ ] Nombres de variables/funciones descriptivos
- [ ] Sin código duplicado
- [ ] Comentarios donde son necesarios
- [ ] Sin código comentado o debug logs

#### ✅ Tests
- [ ] Tests unitarios para nueva funcionalidad
- [ ] Tests de integración cuando aplica
- [ ] Tests BDD para User Stories
- [ ] Todos los tests pasan
- [ ] Cobertura de código adecuada

#### ✅ Documentación
- [ ] OpenAPI/Swagger actualizado si hay cambios en API
- [ ] README actualizado si es necesario
- [ ] Comentarios XML en métodos públicos

#### ✅ Seguridad
- [ ] Sin credenciales hardcodeadas
- [ ] Validación de inputs
- [ ] Manejo de errores apropiado
- [ ] Sin vulnerabilidades conocidas

#### ✅ Performance
- [ ] Sin queries N+1
- [ ] Operaciones costosas optimizadas
- [ ] Memoria liberada correctamente

### Checklist para Autores

Antes de crear PR:

- [ ] Código formateado según estándares
- [ ] Sin warnings del compilador
- [ ] Tests pasando localmente
- [ ] Documentación actualizada
- [ ] Self-review completada

---

## Proceso de Pull Request

### Título de PR

```
<tipo>: <descripción corta>
```

Ejemplo: `feat: Implementar subida de videos con drag & drop`

### Template de PR

```markdown
## 📝 Descripción
Descripción clara del cambio realizado.

## 🔗 Issue Relacionado
Closes #<issue-number>

## 📋 Tipo de Cambio
- [ ] Feature (nueva funcionalidad)
- [ ] Bug fix (corrección)
- [ ] Refactor
- [ ] Documentación
- [ ] Test

## ✅ Checklist
- [ ] Código sigue convenciones
- [ ] Tests agregados/actualizados
- [ ] Documentación actualizada
- [ ] Sin conflictos con develop

## 📸 Screenshots (si aplica)
Capturas de pantalla de cambios en UI.

## 🧪 Cómo Probar
1. Pasos para probar el cambio
2. Comandos a ejecutar
3. Resultado esperado
```

### Proceso de Revisión

1. **Autor**: Crea PR con template completo
2. **Reviewer**: Revisa según checklist (máximo 24h)
3. **Feedback**: Comentarios en línea, aprobación o cambios solicitados
4. **Autor**: Aborda feedback, marca comentarios como resueltos
5. **Merge**: Reviewer con permisos hace merge

### Reglas de Merge

- ✅ Al menos 1 aprobación requerida
- ✅ Todos los tests deben pasar
- ✅ Sin conflictos
- ✅ Branch actualizada con develop
- ✅ Squash merge para features pequeños
- ✅ Merge commit para features grandes

---

## 🔗 Referencias

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Versioning](https://semver.org/)

---

*Última actualización: 18 de Febrero 2026*