#!/usr/bin/env python3
"""
Script para corregir referencias duplicadas de 'spikes/spikes/' a 'spikes/' 
en los archivos Python del proyecto.
"""

import os
import re
from pathlib import Path

def fix_file(file_path: str) -> bool:
    """Corrige las referencias duplicadas en un archivo específico."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patrón para encontrar referencias duplicadas
        original_content = content
        
        # Corregir referencias duplicadas
        content = re.sub(r'"spikes/spikes/runs/', '"spikes/runs/', content)
        content = re.sub(r"'spikes/spikes/runs/", "'spikes/runs/", content)
        content = re.sub(r'spikes/spikes/runs/', 'spikes/runs/', content)
        
        # Si hubo cambios, guardar el archivo
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corregido: {file_path}")
            return True
        else:
            print(f"⏭️  Sin cambios: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        return False

def main():
    """Función principal."""
    print("🔧 Corrigiendo referencias duplicadas de 'spikes/spikes/' a 'spikes/'...")
    
    # Buscar todos los archivos Python en la carpeta spikes/
    spikes_dir = Path("spikes")
    python_files = list(spikes_dir.rglob("*.py"))
    
    fixed_count = 0
    total_count = len(python_files)
    
    for file_path in python_files:
        if fix_file(str(file_path)):
            fixed_count += 1
    
    print(f"\n📊 Resumen:")
    print(f"   Total archivos procesados: {total_count}")
    print(f"   Archivos corregidos: {fixed_count}")
    print(f"   Archivos sin cambios: {total_count - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ Corrección completada. {fixed_count} archivos modificados.")
    else:
        print(f"\n⏭️  No se encontraron referencias duplicadas.")

if __name__ == "__main__":
    main()