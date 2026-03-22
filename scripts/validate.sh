#!/bin/bash

# AnalizadorPadel Validation Script
# This script runs all tests and generates a comprehensive report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Report directory
REPORT_DIR="$PROJECT_ROOT/validation-report"
mkdir -p "$REPORT_DIR"

# Timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_FILE="$REPORT_DIR/validation-report-$TIMESTAMP.md"

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to run tests and capture results
run_test_suite() {
    local name=$1
    local command=$2
    local log_file=$3

    print_header "Running: $name"

    if eval "$command" > "$log_file" 2>&1; then
        print_success "$name passed"
        ((TESTS_PASSED++))
        return 0
    else
        print_error "$name failed"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Initialize report
cat > "$REPORT_FILE" << EOF
# AnalizadorPadel Validation Report

**Date:** $(date)
**Commit:** $(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
**Branch:** $(git branch --show-current 2>/dev/null || echo "N/A")

## Summary

| Category | Status |
|----------|--------|
| Backend Unit Tests | TBD |
| Backend Integration Tests | TBD |
| Backend BDD Tests | TBD |
| Frontend Tests | TBD |
| Python Tests | TBD |
| E2E Tests | TBD |
| Build Verification | TBD |

## Detailed Results

EOF

echo "Starting AnalizadorPadel Validation..."
echo "Report will be saved to: $REPORT_FILE"

# ============================================
# PHASE 0: Prerequisites Check
# ============================================

print_header "PHASE 0: Checking Prerequisites"

if command -v dotnet &> /dev/null; then
    DOTNET_VERSION=$(dotnet --version)
    print_success "dotnet found: $DOTNET_VERSION"
else
    print_error "dotnet not found"
    exit 1
fi

if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "node found: $NODE_VERSION"
else
    print_error "node not found"
    exit 1
fi

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "python3 found: $PYTHON_VERSION"
else
    print_error "python3 not found"
    exit 1
fi

# ============================================
# PHASE 1: Backend Tests
# ============================================

print_header "PHASE 1: Backend Tests"

cd "$PROJECT_ROOT"

# Restore dependencies
echo "Restoring .NET dependencies..."
dotnet restore --verbosity quiet

# Build the solution
echo "Building solution..."
dotnet build --verbosity quiet

# Run all backend tests
BACKEND_LOG="$REPORT_DIR/backend-tests-$TIMESTAMP.log"
if run_test_suite "Backend Tests" "dotnet test --verbosity normal" "$BACKEND_LOG"; then
    BACKEND_STATUS="✅ PASSED"
else
    BACKEND_STATUS="❌ FAILED"
fi

# ============================================
# PHASE 2: Frontend Tests
# ============================================

print_header "PHASE 2: Frontend Tests"

cd "$PROJECT_ROOT/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Run frontend tests
FRONTEND_LOG="$REPORT_DIR/frontend-tests-$TIMESTAMP.log"
if run_test_suite "Frontend Tests" "npm run test -- --run" "$FRONTEND_LOG"; then
    FRONTEND_STATUS="✅ PASSED"
else
    FRONTEND_STATUS="❌ FAILED"
fi

# ============================================
# PHASE 3: Python Tests
# ============================================

print_header "PHASE 3: Python Tests"

cd "$PROJECT_ROOT/python-scripts"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

# Run Python tests
PYTHON_LOG="$REPORT_DIR/python-tests-$TIMESTAMP.log"
if run_test_suite "Python Tests" "pytest -v --tb=short" "$PYTHON_LOG"; then
    PYTHON_STATUS="✅ PASSED"
else
    PYTHON_STATUS="❌ FAILED"
fi

# Deactivate virtual environment
deactivate

# ============================================
# PHASE 4: Build Verification
# ============================================

print_header "PHASE 4: Build Verification"

# Backend build verification
cd "$PROJECT_ROOT/backend"
BUILD_BACKEND_LOG="$REPORT_DIR/build-backend-$TIMESTAMP.log"
if run_test_suite "Backend Build" "dotnet build --verbosity quiet" "$BUILD_BACKEND_LOG"; then
    BUILD_BACKEND_STATUS="✅ PASSED"
else
    BUILD_BACKEND_STATUS="❌ FAILED"
fi

# Frontend build verification
cd "$PROJECT_ROOT/frontend"
BUILD_FRONTEND_LOG="$REPORT_DIR/build-frontend-$TIMESTAMP.log"
if run_test_suite "Frontend Build" "npm run build" "$BUILD_FRONTEND_LOG"; then
    BUILD_FRONTEND_STATUS="✅ PASSED"
else
    BUILD_FRONTEND_STATUS="❌ FAILED"
fi

# ============================================
# PHASE 5: E2E Tests (optional)
# ============================================

print_header "PHASE 5: E2E Tests"

cd "$PROJECT_ROOT/e2e"

if [ -f "package.json" ] && [ -d "node_modules" ]; then
    # Run E2E tests
    E2E_LOG="$REPORT_DIR/e2e-tests-$TIMESTAMP.log"
    if run_test_suite "E2E Tests" "npm test" "$E2E_LOG"; then
        E2E_STATUS="✅ PASSED"
    else
        E2E_STATUS="❌ FAILED"
    fi
else
    print_warning "E2E tests not configured or dependencies not installed"
    E2E_STATUS="⚠️ SKIPPED"
    ((TESTS_SKIPPED++))
fi

# ============================================
# Generate Final Report
# ============================================

print_header "Generating Final Report"

# Update report with results
cat >> "$REPORT_FILE" << EOF

## Backend Tests

Status: $BACKEND_STATUS

\`\`\`
$(tail -100 "$BACKEND_LOG" 2>/dev/null || echo "Log not available")
\`\`\`

## Frontend Tests

Status: $FRONTEND_STATUS

\`\`\`
$(tail -100 "$FRONTEND_LOG" 2>/dev/null || echo "Log not available")
\`\`\`

## Python Tests

Status: $PYTHON_STATUS

\`\`\`
$(tail -100 "$PYTHON_LOG" 2>/dev/null || echo "Log not available")
\`\`\`

## Build Verification

### Backend Build
Status: $BUILD_BACKEND_STATUS

### Frontend Build
Status: $BUILD_FRONTEND_STATUS

## E2E Tests

Status: $E2E_STATUS

\`\`\`
$(tail -50 "$E2E_LOG" 2>/dev/null || echo "Log not available")
\`\`\`

## Final Summary

| Metric | Count |
|--------|-------|
| Test Suites Passed | $TESTS_PASSED |
| Test Suites Failed | $TESTS_FAILED |
| Test Suites Skipped | $TESTS_SKIPPED |

EOF

# Calculate overall result
if [ $TESTS_FAILED -eq 0 ]; then
    OVERALL_STATUS="✅ SUCCESS"
    EXIT_CODE=0
else
    OVERALL_STATUS="❌ FAILURE"
    EXIT_CODE=1
fi

cat >> "$REPORT_FILE" << EOF

## Overall Status: $OVERALL_STATUS

---
Generated by AnalizadorPadel Validation Script
EOF

# Print final summary
print_header "VALIDATION COMPLETE"
echo -e "Backend Tests:      $BACKEND_STATUS"
echo -e "Frontend Tests:     $FRONTEND_STATUS"
echo -e "Python Tests:       $PYTHON_STATUS"
echo -e "Backend Build:      $BUILD_BACKEND_STATUS"
echo -e "Frontend Build:     $BUILD_FRONTEND_STATUS"
echo -e "E2E Tests:          $E2E_STATUS"
echo ""
echo -e "Overall Status:     $OVERALL_STATUS"
echo ""
echo "Report saved to:    $REPORT_FILE"
echo ""

exit $EXIT_CODE
