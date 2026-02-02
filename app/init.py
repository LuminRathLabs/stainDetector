# -*- coding: utf-8 -*-
"""
init.py - Punto de entrada principal para el Software MetalRollBand - Manchas.
Versión V4 - Diseño Premium State-of-the-Art con Splash Screen.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import logging
import traceback
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("Launcher")

# Asegurar que el directorio 'app' esté en el path para importaciones relativas
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class SplashScreen:
    """Pantalla de carga con estética ultra-premium y detalles granulares."""
    def __init__(self, root):
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.0) # Empezar invisible para fade-in
        self.window.overrideredirect(True) # Sin bordes de sistema
        
        # Dimensiones optimizadas
        self.width, self.height = 640, 440
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        x = (sw - self.width) // 2
        y = (sh - self.height) // 2
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.window.configure(bg="#020617") # Slate 950 (Oscuro profundo)

        # Contenedor principal con borde de cristal (Glassmorphism sutil)
        self.container = tk.Frame(self.window, bg="#020617", 
                                  highlightthickness=1, 
                                  highlightbackground="#1e293b")
        self.container.pack(fill=tk.BOTH, expand=True)

        # Línea de acento superior (Sutil detalle de diseño)
        self.accent_line = tk.Frame(self.container, bg="#3b82f6", height=2)
        self.accent_line.pack(fill=tk.X)

        # Logo central (Más grande y con aire)
        try:
            from PIL import Image, ImageTk
            logo_path = os.path.join(current_dir, "..", "resources", "detectManchasLogo.png")
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                img = img.resize((170, 170), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                self.logo_label = tk.Label(self.container, image=self.logo_img, bg="#020617")
                self.logo_label.pack(pady=(45, 15))
            else:
                tk.Label(self.container, text="MetalRollBand", fg="#3b82f6", bg="#020617", 
                         font=("Segoe UI", 42, "bold")).pack(pady=40)
        except Exception:
            tk.Label(self.container, text="Manchas System", fg="white", bg="#020617", 
                     font=("Segoe UI", 28)).pack(pady=50)

        # Subtítulo Industrial
        tk.Label(self.container, text="SOFTWARE DE INSPECCIÓN • IA VISION", 
                 fg="#475569", bg="#020617", font=("Segoe UI Semibold", 9)).pack()

        # Mensaje de estado principal (Tipografía nítida)
        self.status_text = tk.StringVar(value="Inicializando Sistema...")
        self.status_label = tk.Label(self.container, textvariable=self.status_text, 
                                     fg="#f1f5f9", bg="#020617", 
                                     font=("Segoe UI", 13, "bold"))
        self.status_label.pack(pady=(45, 2))

        # Detalle granular (Sub-estado en color slate-400)
        self.detail_text = tk.StringVar(value="Preparando motores de detección...")
        self.detail_label = tk.Label(self.container, textvariable=self.detail_text, 
                                     fg="#94a3b8", bg="#020617", 
                                     font=("Segoe UI", 10, "italic"))
        self.detail_label.pack(pady=(0, 15))

        # Progress bar moderna con Canvas (Efecto "Glow")
        self.progress_width = 480
        # Pista de la barra (track)
        self.progress_canvas = tk.Canvas(self.container, width=self.progress_width, height=8, 
                                         bg="#0f172a", highlightthickness=0, bd=0)
        self.progress_canvas.pack(pady=5)
        
        # Barra de progreso
        self.bar = self.progress_canvas.create_rectangle(0, 0, 0, 8, fill="#2563eb", outline="")
        
        # Porcentaje estilizado
        self.percent_text = tk.StringVar(value="0%")
        self.percent_label = tk.Label(self.container, textvariable=self.percent_text, 
                                      fg="#3b82f6", bg="#020617", 
                                      font=("Segoe UI", 11, "bold"))
        self.percent_label.pack()

        # Footer sutil
        tk.Label(self.container, text="v1.2.0 • Proceso de Inspección Activo", 
                 fg="#1e293b", bg="#020617", font=("Segoe UI", 8)).pack(side=tk.BOTTOM, pady=10)

        # Iniciar fundido de entrada
        self.fade_in()

    def fade_in(self):
        alpha = self.window.attributes("-alpha")
        if alpha < 1.0:
            self.window.attributes("-alpha", alpha + 0.1)
            self.window.after(15, self.fade_in)

    def update_progress(self, val, msg, detail=""):
        """Actualiza progreso con suavidad."""
        self.status_text.set(msg)
        if detail:
            self.detail_text.set(detail)
        self.percent_text.set(f"{int(val)}%")
        
        # Animación de la barra
        target_w = (val / 100) * self.progress_width
        self.progress_canvas.coords(self.bar, 0, 0, target_w, 8)
        self.window.update()

    def close(self):
        """Cierre elegante con fundido de salida."""
        self.fade_out()

    def fade_out(self):
        alpha = self.window.attributes("-alpha")
        if alpha > 0.0:
            self.window.attributes("-alpha", alpha - 0.1)
            self.window.after(10, self.fade_out)
        else:
            self.window.destroy()

def load_application(splash, root):
    """Secuencia de carga optimizada con feedback granular."""
    try:
        # FASE 1: NÚCLEO (0-20%)
        splash.update_progress(5, "Configurando Entorno", "Optimizando kernel de ejecución...")
        time.sleep(0.3)
        splash.update_progress(15, "Configurando Entorno", "Cargando bibliotecas de sistema C++...")
        
        # FASE 2: COMUNICACIÓN (20-35%)
        splash.update_progress(25, "Enlazando Servicios", "Conectando con Servidor RTSP (MediaMTX)...")
        from utils import ensure_mediamtx_running
        ensure_mediamtx_running()
        splash.update_progress(35, "Enlazando Servicios", "Streaming Server activo y verificado.")
        
        # FASE 3: INTELIGENCIA ARTIFICIAL (35-80%) - Carga intensiva
        splash.update_progress(40, "Cargando Motor de IA", "Vinculando NumPy y tensores matemáticos...")
        import numpy as np
        
        splash.update_progress(45, "Cargando Motor de IA", "Preparando visión artificial OpenCV...")
        import cv2
        
        splash.update_progress(55, "Cargando Motor de IA", "Inicializando PyTorch & CUDA Cores...")
        import torch
        
        splash.update_progress(65, "Cargando Motor de IA", "Desplegando redes YOLO (Ultralytics)...")
        from ultralytics import YOLO
        
        splash.update_progress(75, "Cargando Motor de IA", "Cargando lógica de detección...")
        from detect_manchas_gui_rtsp import DetectorGUI, get_resource_path
        
        # FASE 4: INTERFAZ & CONFIG (80-95%)
        splash.update_progress(85, "Construyendo Interfaz", "Cargando perfiles, iconos y estilos...")
        root.title("Software MetalRollBand - Manchas")
        
        # Icono de ventana
        try:
            from PIL import Image, ImageTk
            icon_path = get_resource_path(os.path.join("..", "resources", "detectManchasLogo.png"), is_config=False)
            if os.path.exists(icon_path):
                icon_img = Image.open(icon_path).resize((512, 512), Image.LANCZOS)
                photo = ImageTk.PhotoImage(icon_img)
                root.iconphoto(True, photo)
                root._app_icon = photo
        except Exception: pass

        # Instanciar APP principal
        app = DetectorGUI(root)
        
        splash.update_progress(95, "Finalizando", "Sincronizando PLC y telemetría...")
        time.sleep(0.3)
        
        # CIERRE
        splash.update_progress(100, "¡Todo Listo!", "Ejecutando aplicación principal...")
        time.sleep(0.2)
        
        splash.close()
        
        # Preparar y mostrar ventana principal
        root.state('zoomed')
        root.deiconify()
        root.lift()
        root.focus_force()
        
        LOGGER.info(">>> Sistema iniciado exitosamente desde init.py")

    except Exception as e:
        LOGGER.error(f"Error crítico en el arranque: {e}\n{traceback.format_exc()}")
        if splash:
            splash.status_text.set("ERROR CRÍTICO")
            splash.detail_text.set(str(e)[:60])
            splash.window.configure(bg="#450a0a") # Color error oscuro
        time.sleep(4)
        sys.exit(1)

def main():
    """Arranque atomizado."""
    root = tk.Tk()
    root.withdraw() # Ocultar hasta que el splash termine
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    splash = SplashScreen(root)
    # Dar un pequeño margen para que el splash se dibuje antes de empezar la carga pesada
    root.after(400, lambda: load_application(splash, root))
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
