# -*- coding: utf-8 -*-
"""
Garbage Collector para ManchasGuida.

Este módulo se encarga de la limpieza automática de archivos obsoletos o temporales
generados por la aplicación para evitar el crecimiento indefinido del uso de disco.

Archivos y directorios gestionados:
1. data/runtime/ (captures, metadata, snapshots)
2. config/logs/ (archivos de traza de rendimiento)
3. runs/ (salidas de YOLO) -> ¡NO SE BORRAN! (Gestionado externamente o críticos)

Configuración:
Las variables globales al inicio del archivo definen los tiempos de retención.
"""

import os
import shutil
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN DE RETENCIÓN (CONFIGURABLE POR EL USUARIO)
# ==============================================================================

# Días para retener carpetas de 'runs/' (YOLO ultralytics)
# Se borrarán las carpetas de entrenamiento/inferencia más antiguas que esto.
# RETENTION_DAYS_RUNS = 7  <-- DESACTIVADO POR PETICIÓN DE USUARIO

# Días para retener datos de tiempo de ejecución en 'data/runtime/'
# Incluye capturas (captures), metadatos (metadata) y snapshots temporales.
# Estos archivos se generan por cada sesión (instance_id).
RETENTION_DAYS_RUNTIME = 7

# Días para retener logs de rendimiento en 'config/logs/'
RETENTION_DAYS_LOGS = 7

# Intervalo de ejecución del recolector de basura (en minutos)
GARBAGE_COLLECTOR_INTERVAL_MINUTES = 60

# ==============================================================================
# LOGGER INTERNO
# ==============================================================================

LOGGER = logging.getLogger("GarbageCollector")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][GC] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

# ==============================================================================
# LÓGICA DE LIMPIEZA
# ==============================================================================

def get_base_dir() -> Path:
    """Obtiene el directorio base del proyecto (asumiendo que este script está en app/)."""
    # Este script está en app/garbage_collector.py -> base del proyecto es app/../
    return Path(__file__).parent.parent.resolve()

# def cleanup_runs_folder(base_dir: Path, retention_days: int) -> int:
#     """Limpia carpetas antiguas generadas por YOLO en runs/."""
#     # ... (ELIMINADO: Los runs son críticos y no deben borrarse)
#     return 0

def cleanup_runtime_data(base_dir: Path, retention_days: int) -> int:
    """Limpia carpetas de sesión antiguas en data/runtime/."""
    runtime_dir = base_dir / "data" / "runtime"
    if not runtime_dir.exists():
        return 0

    count = 0
    cutoff_time = time.time() - (retention_days * 86400)

    # data/runtime/ tiene subcarpetas por tipo: captures, metadata, snapshots, etc.
    # Dentro de ellas están las carpetas por instance_id
    
    # Lista de subcarpetas conocidas dentro de runtime
    subdirs = ["captures", "metadata", "snapshots"]

    for subdir_name in subdirs:
        subdir_path = runtime_dir / subdir_name
        if subdir_path.exists() and subdir_path.is_dir():
            for instance_path in subdir_path.iterdir():
                if instance_path.is_dir():
                    try:
                        mtime = instance_path.stat().st_mtime
                        if mtime < cutoff_time:
                            LOGGER.info(f"Eliminando datos de sesión obsoletos ({subdir_name}): {instance_path}")
                            shutil.rmtree(instance_path)
                            count += 1
                    except Exception as e:
                        LOGGER.error(f"Error limpiando {instance_path}: {e}")

    return count

def cleanup_logs(base_dir: Path, retention_days: int) -> int:
    """Limpia logs antiguos en config/logs/."""
    # Intentar ubicar config/logs relative a la estructura
    # Si este script está en app/, config/ está en ../config
    logs_dir = base_dir / "config" / "logs"
    
    if not logs_dir.exists():
        return 0

    count = 0
    cutoff_time = time.time() - (retention_days * 86400)

    for log_file in logs_dir.glob("*.log"):
        try:
            mtime = log_file.stat().st_mtime
            if mtime < cutoff_time:
                LOGGER.info(f"Eliminando log obsoleto: {log_file}")
                log_file.unlink()
                count += 1
        except Exception as e:
            LOGGER.error(f"Error borrando log {log_file}: {e}")
            
    return count

def run_garbage_collection() -> None:
    """Ejecuta todos los procesos de limpieza."""
    LOGGER.info("Iniciando Garbage Collector...")
    base_dir = get_base_dir()
    cwd = Path.cwd()
    
    try:
        # Limpiar runs/ -> DESACTIVADO
        # deleted_runs = cleanup_runs_folder(base_dir, RETENTION_DAYS_RUNS)
        # if cwd != base_dir:
        #     deleted_runs += cleanup_runs_folder(cwd, RETENTION_DAYS_RUNS)

        # if deleted_runs > 0:
        #     LOGGER.info(f"Limpieza de runs/: {deleted_runs} carpetas eliminadas (>{RETENTION_DAYS_RUNS} días).")
        
        deleted_runtime = cleanup_runtime_data(base_dir, RETENTION_DAYS_RUNTIME)
        if deleted_runtime > 0:
            LOGGER.info(f"Limpieza de data/runtime/: {deleted_runtime} carpetas de sesión eliminadas (>{RETENTION_DAYS_RUNTIME} días).")
        
        deleted_logs = cleanup_logs(base_dir, RETENTION_DAYS_LOGS)
        if deleted_logs > 0:
            LOGGER.info(f"Limpieza de config/logs/: {deleted_logs} archivos eliminados (>{RETENTION_DAYS_LOGS} días).")
            
    except Exception as e:
        LOGGER.error(f"Excepción general en Garbage Collector: {e}")
        
    LOGGER.info("Garbage Collector finalizado.")

# ==============================================================================
# GESTIÓN DE HILO Y PROGRAMACIÓN
# ==============================================================================

_GC_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()

def _gc_loop():
    """Bucle del hilo que ejecuta el GC periódicamente."""
    # Ejecutar inmediatamente al inicio (dar unos segundos de cortesía para no competir en arranque)
    if not _STOP_EVENT.wait(5.0):
        # Ejecución inicial
        run_garbage_collection()
    
    while not _STOP_EVENT.is_set():
        # Esperar intervalo
        if _STOP_EVENT.wait(GARBAGE_COLLECTOR_INTERVAL_MINUTES * 60):
            break
        # Ejecutar periódicamente
        run_garbage_collection()

def start_garbage_collector():
    """Inicia el Garbage Collector en un hilo en segundo plano."""
    global _GC_THREAD
    if _GC_THREAD is not None and _GC_THREAD.is_alive():
        LOGGER.warning("Garbage Collector ya está en ejecución.")
        return

    _STOP_EVENT.clear()
    _GC_THREAD = threading.Thread(target=_gc_loop, name="GarbageCollectorThread", daemon=True)
    _GC_THREAD.start()
    LOGGER.info(f"Garbage Collector iniciado (hilo secundario). Intervalo: {GARBAGE_COLLECTOR_INTERVAL_MINUTES} min.")

def stop_garbage_collector():
    """Detiene el hilo del Garbage Collector."""
    global _GC_THREAD
    if _GC_THREAD:
        _STOP_EVENT.set()
        _GC_THREAD.join(timeout=2.0)
        _GC_THREAD = None
        LOGGER.info("Garbage Collector detenido.")

if __name__ == "__main__":
    # Prueba manual
    print("Modo prueba manual de Garbage Collector")
    logging.basicConfig(level=logging.INFO)
    run_garbage_collection()
