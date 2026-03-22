# Frontend - Consideraciones Operativas

## ✅ Validación Frontend

- Para confirmar que el frontend habla correctamente con el backend, usar un smoke test real con navegador o E2E; servir `index.html` o compilar no garantiza integración funcional.
- En desarrollo local, validar siempre la combinación completa de `puerto frontend`, `puerto backend`, `proxy o VITE_API_URL` y `CORS`.
- Mantener la organización de `src` separando código por `features`, elementos compartidos en `shared` y soporte de pruebas en `test`.
- Después de cambios estructurales del frontend, validar al menos con `npm run build` y `npm run test -- --run`.
