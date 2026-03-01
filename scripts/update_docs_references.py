#!/usr/bin/env python3
"""
Script para actualizar todas las referencias de 'runs/' a 'spikes/runs/' 
en los archivos Markdown del proyecto.
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
        
        # Reemplazar referencias en código blocks
        content = re.sub(r'`runs/', '`spikes/runs/', content)
        
        # Reemplazar referencias en texto plano
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
    print("🔄 Actualizando referencias de 'runs/' a 'spikes/runs/' en documentación...")
    
    # Buscar todos los archivos Markdown en la carpeta docs/
    docs_dir = Path("docs")
    markdown_files = list(docs_dir.rglob("*.md"))
    
    updated_count = 0
    total_count = len(markdown_files)
    
    for file_path in markdown_files:
        # Omitir el ADR-005 que ya está correcto
        if "ADR-005-reorganizacion-carpeta-runs.md" in str(file_path):
            print(f"⏭️  Omitiendo: {file_path} (ADR-005)")
            continue
            
        if update_file(str(file_path)):
            updated_count += 1
    
    print(f"\n📊 Resumen:")
    print(f"   Total archivos procesados: {total_count}")
    print(f"   Archivos actualizados: {updated_count}")
    print(f"   Archivos sin cambios: {total_count - updated_count}")
    
    if updated_count > 0:
        print(f"\n✅ Documentación actualizada. {updated_count} archivos modificados.")
    else:
        print(f"\n⏭️  No se encontraron referencias para actualizar en la documentación.")

if __name__ == "__main__":
    main()