#!/bin/bash

# Script de prueba E2E usando curl
# Realiza pruebas de integración contra el backend

set -e

BASE_URL="http://localhost:5000"
VIDEO_FILE="/Users/estanislao/Documents/Codigo/AnalizadorPadel/test-videos/PadelPro3.mp4"

echo "========================================="
echo "  E2E API Testing with curl"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper functions
pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}ℹ${NC} $1"; }

# Check if server is running
echo "Verificando que el servidor esté corriendo..."
if ! curl -s "$BASE_URL/api/health" > /dev/null; then
    fail "El servidor no está corriendo en $BASE_URL"
fi
pass "Servidor respondiendo"

# Test 1: Health endpoint
echo ""
info "Test 1: Health endpoint"
if curl -s "$BASE_URL/api/health" | grep -q "healthy\|Healthy"; then
    pass "Health check OK"
else
    fail "Health check failed"
fi

# Test 2: Get videos list
echo ""
info "Test 2: GET /api/videos"
RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/api/videos")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" = "200" ]; then
    pass "GET /api/videos retorna 200"
else
    fail "GET /api/videos retorna $HTTP_CODE"
fi

# Test 3: Get dashboard stats
echo ""
info "Test 3: GET /api/dashboard/stats"
RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/api/dashboard/stats")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" = "200" ]; then
    pass "GET /api/dashboard/stats retorna 200"
else
    fail "GET /api/dashboard/stats retorna $HTTP_CODE"
fi

# Test 4: Upload video (if file exists)
echo ""
info "Test 4: POST /api/videos (upload)"
if [ -f "$VIDEO_FILE" ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        -F "file=@$VIDEO_FILE" \
        "$BASE_URL/api/videos")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)
    
    if [ "$HTTP_CODE" = "201" ] || [ "$HTTP_CODE" = "200" ]; then
        pass "Upload retorna $HTTP_CODE"
        # Extract video ID if possible
        VIDEO_ID=$(echo "$BODY" | grep -o '"id":[0-9]*' | cut -d: -f2)
        if [ ! -z "$VIDEO_ID" ]; then
            info "Video ID creado: $VIDEO_ID"
        fi
    else
        fail "Upload retorna $HTTP_CODE"
    fi
else
    info "Archivo de video no encontrado, saltando test de upload"
fi

# Test 5: Get non-existent video
echo ""
info "Test 5: GET /api/videos/999999 (not found)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/videos/999999")
if [ "$HTTP_CODE" = "404" ]; then
    pass "GET video no existente retorna 404"
else
    fail "Esperado 404, retornado $HTTP_CODE"
fi

# Test 6: Get non-existent analysis
echo ""
info "Test 6: GET /api/analyses/999999 (not found)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/analyses/999999")
if [ "$HTTP_CODE" = "404" ]; then
    pass "GET análisis no existente retorna 404"
else
    fail "Esperado 404, retornado $HTTP_CODE"
fi

# Test 7: Invalid file upload
echo ""
info "Test 7: POST /api/videos con archivo inválido"
TEMP_FILE=$(mktemp)
echo "invalid content" > "$TEMP_FILE"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -F "file=@$TEMP_FILE" \
    "$BASE_URL/api/videos")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
rm "$TEMP_FILE"

if [ "$HTTP_CODE" = "400" ]; then
    pass "Upload de archivo inválido retorna 400"
else
    fail "Esperado 400, retornado $HTTP_CODE"
fi

# Test 8: Video streaming with Range header (if video exists)
if [ ! -z "$VIDEO_ID" ]; then
    echo ""
    info "Test 8: GET /api/videos/$VIDEO_ID/stream con Range"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Range: bytes=0-1023" \
        "$BASE_URL/api/videos/$VIDEO_ID/stream")
    if [ "$HTTP_CODE" = "206" ] || [ "$HTTP_CODE" = "200" ]; then
        pass "Streaming con Range retorna $HTTP_CODE"
    else
        fail "Streaming retorna $HTTP_CODE"
    fi
fi

echo ""
echo "========================================="
echo -e "${GREEN}✓ Todos los tests E2E pasaron!${NC}"
echo "========================================="
