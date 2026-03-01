#!/usr/bin/env python3
"""
Spike 4 - Prueba de Integración via Subprocess

Simula cómo .NET llamaría a Python via subprocess para procesar videos.
Valida:
1. Manejo de errores
2. Uso de memoria con YOLO
3. Comportamiento con múltiples procesos concurrentes
4. Path management

Este spike NO requiere .NET - simula la integración con Python puro.
"""

import subprocess
import sys
import os
import json
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def get_memory_usage_mb():
    """Obtiene el uso de memoria del proceso actual en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_python_script(script_path, args, timeout=60):
    """
    Simula cómo .NET ejecutaría un script Python via subprocess.
    
    Args:
        script_path: Ruta al script Python
        args: Lista de argumentos
        timeout: Timeout en segundos
    
    Returns:
        dict con resultado, stdout, stderr, returncode
    """
    cmd = [sys.executable, script_path] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'error': None
        }
    
    except subprocess.TimeoutExpired as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': f'Timeout after {timeout}s',
            'error': 'TIMEOUT'
        }
    
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'error': type(e).__name__
        }


def test_1_successful_execution():
    """Test 1: Ejecución exitosa de script Python."""
    print("\n" + "="*60)
    print("TEST 1: Ejecución Exitosa")
    print("="*60)
    
    # Crear script de prueba simple
    test_script = Path("spikes/spike4/test_success.py")
    test_script.write_text("""
import sys
import json

# Simular procesamiento
result = {
    'status': 'success',
    'message': 'Video procesado correctamente',
    'frames_processed': 100
}

print(json.dumps(result))
sys.exit(0)
""")
    
    result = run_python_script(str(test_script), [])
    
    print(f"✅ Success: {result['success']}")
    print(f"   Return code: {result['returncode']}")
    print(f"   Output: {result['stdout'].strip()}")
    
    # Limpiar
    test_script.unlink()
    
    return result['success']


def test_2_error_handling():
    """Test 2: Manejo de errores cuando Python falla."""
    print("\n" + "="*60)
    print("TEST 2: Manejo de Errores")
    print("="*60)
    
    # Crear script que falla
    test_script = Path("spikes/spike4/test_error.py")
    test_script.write_text("""
import sys

# Simular error
raise ValueError("Error simulado: Video corrupto")
sys.exit(1)
""")
    
    result = run_python_script(str(test_script), [])
    
    print(f"❌ Success: {result['success']} (esperado: False)")
    print(f"   Return code: {result['returncode']}")
    print(f"   Error capturado: {result['stderr'][:100]}...")
    
    # Limpiar
    test_script.unlink()
    
    return not result['success']  # Éxito si detectó el error


def test_3_yolo_memory_usage():
    """Test 3: Uso de memoria al cargar YOLO."""
    print("\n" + "="*60)
    print("TEST 3: Uso de Memoria con YOLO")
    print("="*60)
    
    # Crear script que carga YOLO
    test_script = Path("spikes/spike4/test_yolo.py")
    test_script.write_text("""
import sys
import os
import psutil
from ultralytics import YOLO

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Memoria inicial
mem_before = get_memory_mb()
print(f"Memoria antes de cargar YOLO: {mem_before:.1f} MB")

# Cargar modelo
model = YOLO('../models/yolov8m.pt')

# Memoria después
mem_after = get_memory_mb()
print(f"Memoria después de cargar YOLO: {mem_after:.1f} MB")
print(f"Incremento: {mem_after - mem_before:.1f} MB")

sys.exit(0)
""")
    
    mem_before = get_memory_usage_mb()
    print(f"Memoria proceso padre antes: {mem_before:.1f} MB")
    
    result = run_python_script(str(test_script), [], timeout=30)
    
    mem_after = get_memory_usage_mb()
    print(f"Memoria proceso padre después: {mem_after:.1f} MB")
    
    if result['success']:
        print(f"\n✅ YOLO cargado exitosamente")
        print(f"   Output:\n{result['stdout']}")
    else:
        print(f"\n❌ Error al cargar YOLO")
        print(f"   Error: {result['stderr']}")
    
    # Limpiar
    test_script.unlink()
    
    return result['success']


def test_4_concurrent_execution():
    """Test 4: Ejecución concurrente de múltiples procesos."""
    print("\n" + "="*60)
    print("TEST 4: Ejecución Concurrente (3 procesos)")
    print("="*60)
    
    # Crear script que simula procesamiento
    test_script = Path("spikes/spike4/test_concurrent.py")
    test_script.write_text("""
import sys
import time
import random

# Simular procesamiento
process_id = sys.argv[1] if len(sys.argv) > 1 else "0"
duration = random.uniform(0.5, 2.0)

print(f"Proceso {process_id}: Iniciando procesamiento ({duration:.1f}s)")
time.sleep(duration)
print(f"Proceso {process_id}: Completado")

sys.exit(0)
""")
    
    # Ejecutar 3 procesos concurrentes
    num_processes = 3
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            future = executor.submit(
                run_python_script,
                str(test_script),
                [str(i)]
            )
            futures.append((i, future))
        
        results = []
        for proc_id, future in futures:
            result = future.result()
            results.append(result)
            status = "✅" if result['success'] else "❌"
            print(f"{status} Proceso {proc_id}: {result['returncode']}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tiempo total: {elapsed:.2f}s")
    print(f"   Procesos exitosos: {sum(1 for r in results if r['success'])}/{num_processes}")
    
    # Limpiar
    test_script.unlink()
    
    return all(r['success'] for r in results)


def test_5_timeout_handling():
    """Test 5: Manejo de timeout."""
    print("\n" + "="*60)
    print("TEST 5: Manejo de Timeout")
    print("="*60)
    
    # Crear script que tarda mucho
    test_script = Path("spikes/spike4/test_timeout.py")
    test_script.write_text("""
import time

print("Iniciando proceso largo...")
time.sleep(10)  # Más que el timeout
print("Completado")
""")
    
    result = run_python_script(str(test_script), [], timeout=2)
    
    print(f"⏱️ Timeout detectado: {result['error'] == 'TIMEOUT'}")
    print(f"   Error: {result['stderr']}")
    
    # Limpiar
    test_script.unlink()
    
    return result['error'] == 'TIMEOUT'


def run_all_tests():
    """Ejecuta todos los tests del spike."""
    print("🔬 Spike 4 - Prueba de Integración via Subprocess")
    print("="*60)
    print("\nSimula cómo .NET llamaría a Python para procesar videos")
    print("Valida: errores, memoria, concurrencia, timeouts\n")
    
    # Crear directorio si no existe
    os.makedirs("spikes/spike4", exist_ok=True)
    
    results = {}
    
    # Ejecutar tests
    results['test_1_success'] = test_1_successful_execution()
    results['test_2_errors'] = test_2_error_handling()
    results['test_3_memory'] = test_3_yolo_memory_usage()
    results['test_4_concurrent'] = test_4_concurrent_execution()
    results['test_5_timeout'] = test_5_timeout_handling()
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests pasados")
    
    # Guardar resultados
    output = {
        'spike': 'Spike 4 - Subprocess Integration',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': results,
        'summary': {
            'total': total_tests,
            'passed': total_passed,
            'failed': total_tests - total_passed,
            'success_rate': f"{100 * total_passed / total_tests:.1f}%"
        }
    }
    
    output_path = "spikes/spike4/results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Resultados guardados en: {output_path}")
    
    # Conclusión
    print("\n" + "="*60)
    print("🎯 CONCLUSIÓN")
    print("="*60)
    
    if total_passed == total_tests:
        print("✅ Todos los tests pasaron")
        print("   La integración via subprocess es viable")
        print("   Recomendación: Proceder con implementación en .NET")
    else:
        print("⚠️ Algunos tests fallaron")
        print("   Revisar los resultados antes de implementar en .NET")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)