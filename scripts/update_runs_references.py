#!/usr/bin/env python3
"""
Script para actualizar todas las referencias de 'runs/' a 'spikes/runs/' 
en los archivos Python del proyecto.
"""

import os
import re
from pathlib import Path

def update_file(file_path: str) -> bool:
    """Actualiza las referencias en un archivo específico."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patrón para encontrar referencias a runs/
        original_content = content
        
        # Reemplazar referencias en strings literales
        content = re.sub(r'"runs/', '"spikes/runs/', content)
        content = re.sub(r"'runs/", "'spikes/runs/", content)
        
        # Reemplazar referencias en comentarios y documentación
        content = re.sub(r'runs/', 'spikes/runs/', content)
        
        # Si hubo cambios, guardar el archivo
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Actualizado: {file_path}")
            return True
        else:
            print(f"⏭️  Sin cambios: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        return False

def main():
    """Función principal."""
    print("🔄 Actualizando referencias de 'runs/' a 'spikes/runs/'...")
    
    # Buscar todos los archivos Python en la carpeta spikes/
    spikes_dir = Path("spikes")
    python_files = list(spikes_dir.rglob("*.py"))
    
    updated_count = 0
    total_count = len(python_files)
    
    for file_path in python_files:
        if update_file(str(file_path)):
            updated_count += 1
    
    print(f"\n📊 Resumen:")
    print(f"   Total archivos procesados: {total_count}")
    print(f"   Archivos actualizados: {updated_count}")
    print(f"   Archivos sin cambios: {total_count - updated_count}")
    
    if updated_count > 0:
        print(f"\n✅ Reorganización completada. {updated_count} archivos actualizados.")
    else:
        print(f"\n⏭️  No se encontraron referencias para actualizar.")

if __name__ == "__main__":
    main()