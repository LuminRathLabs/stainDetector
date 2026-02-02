"""
Utilidades generales de la aplicación.
Este módulo centraliza funciones auxiliares reutilizables para limpieza de código.
"""

from PIL import Image, ImageTk

import os
import sys
import subprocess
import logging
import tkinter as tk

# Configuración básica de logging para este módulo
LOGGER = logging.getLogger("Utils")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][Utils] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

def get_resource_path(relative_path: str) -> str:
    """
    Obtiene la ruta absoluta de un recurso, compatible con PyInstaller y desarrollo.
    
    Args:
        relative_path (str): Ruta relativa desde la raíz del proyecto (o desde el ejecutable).
                             Ejemplo: "bin/mediamtx/mediamtx.exe"
    
    Returns:
        str: Ruta absoluta al recurso.
    """
    if getattr(sys, 'frozen', False):
        # Entorno empaquetado (PyInstaller)
        # Preferimos sys._MEIPASS (onefile extrae allí), fallback al dir del exe.
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        # Entorno desarrollo
        # Asumimos que utils.py está en app/, así que subimos un nivel para ir a la raíz del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.abspath(os.path.join(base_dir, relative_path))

def ensure_mediamtx_running() -> None:
    """
    Verifica si MediaMTX está ejecutándose y, si no lo está, lo inicia.
    Busca el ejecutable en bin/mediamtx/mediamtx.exe.
    """
    PROCESS_NAME = "mediamtx.exe"
    
    # 1. Comprobar si ya está corriendo
    try:
        # Usamos tasklist para buscar el proceso por nombre
        output = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {PROCESS_NAME}"],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            stderr=subprocess.STDOUT
        ).decode('oem', errors='ignore')
        
        if PROCESS_NAME.lower() in output.lower():
            LOGGER.info(f"{PROCESS_NAME} ya está ejecutándose.")
            return

    except Exception as e:
        LOGGER.warning(f"Error comprobando procesos ({e}). Se intentará iniciar de todos modos.")

    # 2. Localizar el ejecutable
    # NOTA: Ajusta la ruta relativa según tu estructura. 
    # Si utils.py está en app/, y bin está en la raíz, usamos "bin/mediamtx/mediamtx.exe"
    exe_path = get_resource_path(os.path.join("bin", "mediamtx", "mediamtx.exe"))
    
    if not os.path.exists(exe_path):
        LOGGER.error(f"No se encontró {PROCESS_NAME} en: {exe_path}")
        return

    # 3. Iniciar el proceso
    try:
        LOGGER.info(f"Iniciando {PROCESS_NAME} desde {exe_path}...")
        
        # Ejecutar en segundo plano, independiente del proceso padre
        if os.name == 'nt':
            # DETACHED_PROCESS = 0x00000008, CREATE_NEW_PROCESS_GROUP = 0x00000200
            # Esto evita que se cierre si cerramos la app principal
            creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            
            # Es importante definir el cwd (current working directory) donde está el exe
            # para que encuentre su mediamtx.yml
            cwd = os.path.dirname(exe_path)
            
            subprocess.Popen(
                [exe_path],
                cwd=cwd,
                creationflags=creation_flags,
                close_fds=True,
                shell=False
            )
        else:
            # Unix-like fallback (no suele pasar en este entorno Windows del usuario, pero por robustez)
            subprocess.Popen(
                [exe_path],
                cwd=os.path.dirname(exe_path),
                start_new_session=True
            )
            
        LOGGER.info("MediaMTX iniciado correctamente.")
        
    except Exception as e:
        LOGGER.error(f"Error crítico al intentar iniciar {PROCESS_NAME}: {e}")

def set_app_logo(root: tk.Tk) -> None:
    """
    Configura el logo de la aplicación para la ventana y la barra de tareas.
    Usa iconbitmap para Windows (mejor para barra de tareas) e iconphoto para el resto.
    """
    try:
        # Rutas absolutas a los recursos
        ico_path = get_resource_path(os.path.join("resources", "detectManchasLogo.ico"))
        png_path = get_resource_path(os.path.join("resources", "detectManchasLogo.png"))
        
        # 1. Identificar el proceso para Windows (agrupación y calidad de icono)
        if sys.platform == "win32":
            try:
                import ctypes
                # Esto ayuda a que Windows asocie el icono al proceso de Python de forma correcta
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MetalRollBand.ManchasGuida.App.v1")
            except Exception:
                pass

        # 2. Configurar el icono de la barra de tareas (más efectivo en Windows)
        if sys.platform == "win32" and os.path.exists(ico_path):
            try:
                # wm_iconbitmap(default=) establece el icono para todas las ventanas futuras (Toplevels)
                root.wm_iconbitmap(default=ico_path)
                LOGGER.info(f"Icono .ico configurado como predeterminado: {ico_path}")
            except Exception as e:
                LOGGER.debug(f"Error cargando .ico con wm_iconbitmap: {e}")
                # Fallback manual si falla el default
                try:
                    root.iconbitmap(ico_path)
                except Exception:
                    pass

        # 3. Configurar el icono de la ventana (necesario para la esquina superior izquierda y otros SO)
        if os.path.exists(png_path):
            try:
                # Usar una versión de alta resolución para que el sistema escale
                icon_img = Image.open(png_path)
                # 512x512 es suficiente para la mayoría de pantallas 4K
                icon_img = icon_img.resize((512, 512), Image.LANCZOS)
                photo = ImageTk.PhotoImage(icon_img)
                
                # iconphoto(True, ...) también intenta aplicarlo a ventanas futuras
                root.iconphoto(True, photo)
                
                # IMPORTANTE: Guardar una referencia fuerte. 
                # Si se guarda en root, pero root se oculta o hay varios, puede fallar.
                # Lo guardamos en una variable global o en un atributo del root.
                if not hasattr(tk, "_global_app_icon"):
                    tk._global_app_icon = photo # type: ignore[attr-defined]
                root._app_icon = photo # type: ignore[attr-defined]
                
                LOGGER.info(f"Logo .png cargado como respaldo/esquina: {png_path}")
            except Exception as e:
                LOGGER.warning(f"Error al cargar logo .png: {e}")
        
        if not os.path.exists(ico_path) and not os.path.exists(png_path):
            LOGGER.warning("No se encontró ningún archivo de logo en resources/.")
            
    except Exception as e:
        LOGGER.warning(f"No se pudo configurar el logo de la aplicación: {e}")
        import traceback
        LOGGER.debug(traceback.format_exc())
