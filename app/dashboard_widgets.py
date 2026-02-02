# dashboard_widgets.py
"""
Widgets personalizables para el dashboard principal.
Soporta: Dato PLC, Acceso a Ajustes, Contador Detecciones, Estado RTSP.
"""
from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from detect_manchas_gui_rtsp import DetectorApp

LOGGER = logging.getLogger(__name__)

# Tipos de widgets disponibles
WIDGET_TYPE_PLC = "plc_value"
WIDGET_TYPE_SETTINGS = "settings_shortcut"
WIDGET_TYPE_DETECTIONS = "detection_count"
WIDGET_TYPE_RTSP_STATUS = "rtsp_status"
WIDGET_TYPE_SYSTEM = "system_resource"

WIDGET_TYPES = {
    WIDGET_TYPE_PLC: "Dato de PLC",
    WIDGET_TYPE_SETTINGS: "Acceso a Ajustes",
    WIDGET_TYPE_DETECTIONS: "Contador Detecciones",
    WIDGET_TYPE_RTSP_STATUS: "Estado RTSP",
    WIDGET_TYPE_SYSTEM: "Recursos Sistema",
}

# Pestañas disponibles para accesos directos
SETTINGS_TABS = [
    "General",
    "Detección",
    "Visualización",
    "Sectores",
    "Conexiones",
    "Rendimiento",
]

# Estilos visuales
COLOR_BG_CARD = "#ffffff"
COLOR_BORDER = "#e0e0e0"
COLOR_TEXT_PRIMARY = "#202124"
COLOR_TEXT_SECONDARY = "#5f6368"
COLOR_ACCENT = "#1a73e8"
COLOR_ACCENT_HOVER = "#1557b0"
FONT_LABEL = ("Segoe UI", 9)
FONT_VALUE = ("Segoe UI", 14, "bold")
FONT_BUTTON = ("Segoe UI", 12, "bold")


class DashboardWidget:
    """Clase base para widgets del dashboard."""
    
    def __init__(
        self,
        widget_id: str,
        widget_type: str,
        label: str,
        config: dict,
        interval_sec: float = 5.0,
    ):
        self.widget_id = widget_id
        self.widget_type = widget_type
        self.label = label
        self.config = config
        self.interval_sec = max(0.5, interval_sec)  # mínimo 0.5s
        
        # UI elements (se asignan al construir)
        self.frame: tk.Frame | None = None
        self.value_var: tk.StringVar | None = None
        self._update_job: str | None = None
        self._running = False
    
    def to_dict(self) -> dict:
        """Serializa el widget a diccionario para guardar."""
        return {
            "widget_id": self.widget_id,
            "type": self.widget_type,
            "label": self.label,
            "config": self.config,
            "interval_sec": self.interval_sec,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DashboardWidget":
        """Deserializa un widget desde diccionario."""
        return cls(
            widget_id=data.get("widget_id", ""),
            widget_type=data.get("type", ""),
            label=data.get("label", "Widget"),
            config=data.get("config", {}),
            interval_sec=data.get("interval_sec", 5.0),
        )


class DashboardWidgetManager:
    """Gestiona los widgets personalizables del dashboard."""
    
    def __init__(self, app: "DetectorApp", parent_frame: tk.Frame):
        self.app = app
        self.parent_frame = parent_frame
        self.widgets: list[DashboardWidget] = []
        self._widget_frames: dict[str, tk.Frame] = {}
        self._plc_connections: dict[str, Any] = {}  # cache de conexiones PLC
        self._update_threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        
        # Intentar obtener colores del tema de la app si existen
        self.bg_card = self.app.colors.get("bg_card", COLOR_BG_CARD) if hasattr(self.app, "colors") else COLOR_BG_CARD
        
        # Contenedor para los widgets
        self.container = tk.Frame(parent_frame, bg=self.bg_card)
        
    def build_ui(self) -> tk.Frame:
        """Construye el contenedor de widgets con botón de añadir."""
        # Frame principal que contiene widgets + botón añadir
        main_frame = tk.Frame(self.parent_frame, bg=self.bg_card)
        
        # Contenedor de widgets (horizontal)
        self.widgets_container = tk.Frame(main_frame, bg=self.bg_card)
        self.widgets_container.pack(side="left", fill="x", expand=True)
        
        # Botón añadir widget (Estilizado)
        self.add_button = tk.Button(
            main_frame,
            text="+",
            font=FONT_BUTTON,
            bg=COLOR_ACCENT,
            fg="white",
            activebackground=COLOR_ACCENT_HOVER,
            activeforeground="white",
            relief="flat",
            borderwidth=0,
            width=3,
            cursor="hand2",
            command=self._on_add_widget,
        )
        self.add_button.pack(side="left", padx=(12, 0), ipady=2)
        
        # Tracking para responsividad
        self._last_cols = 0
        self.widgets_container.bind("<Configure>", self._on_container_resize)
        
        # Cargar widgets guardados
        self._load_widgets_from_config()
        self._rebuild_widget_ui()
        
        return main_frame
    
    def _load_widgets_from_config(self):
        """Carga widgets desde la configuración."""
        try:
            widgets_data = self.app.config.get("dashboard_widgets", [])
            if isinstance(widgets_data, list):
                self.widgets = [DashboardWidget.from_dict(w) for w in widgets_data]
        except Exception as e:
            LOGGER.warning("Error cargando widgets del dashboard: %s", e)
            self.widgets = []
    
    def _save_widgets_to_config(self):
        """Guarda widgets en la configuración."""
        try:
            self.app.config["dashboard_widgets"] = [w.to_dict() for w in self.widgets]
            if hasattr(self.app, "_save_config"):
                self.app._save_config()
        except Exception as e:
            LOGGER.warning("Error guardando widgets del dashboard: %s", e)
    
    def _rebuild_widget_ui(self):
        """Reconstruye todos los widgets en la UI."""
        # Limpiar widgets existentes
        for frame in self._widget_frames.values():
            try:
                frame.destroy()
            except Exception:
                pass
        # Detener hilos antes de limpiar frames
        self._stop_all_updates()
        self._widget_frames.clear()
        
        # Reconstruir con grid responsivo
        self._regrid_widgets()
    
    def _on_container_resize(self, event):
        """Maneja el cambio de tamaño para reajustar el grid (wrap)."""
        self._regrid_widgets()

    def _regrid_widgets(self):
        """Calcula dinámicamente cuántas columnas caben y aplica el grid."""
        if not hasattr(self, "widgets_container") or not self.widgets_container.winfo_exists():
            return

        w = self.widgets_container.winfo_width()
        if w <= 1: 
            return # Aún no renderizado
            
        # Estimación de ancho por widget (card + padding)
        target_w = 160
        cols = max(1, w // target_w)
        
        # Solo re-grid si cambia el número de columnas o si estamos forzando (rebuild)
        # Para simplificar y que sea robusto, lo aplicamos siempre que haya widgets
        for i, widget in enumerate(self.widgets):
            frame = self._widget_frames.get(widget.widget_id)
            if not frame:
                frame = self._create_widget_frame(widget)
                self._widget_frames[widget.widget_id] = frame
                self._start_widget_updates(widget)
            
            row = i // cols
            col = i % cols
            frame.grid(row=row, column=col, padx=(0, 10), pady=(0, 10), sticky="nw")
            
        self._last_cols = cols
    
    def _create_widget_frame(self, widget: DashboardWidget) -> tk.Frame:
        """Crea el frame visual para un widget con estilo mejorado."""
        # Frame contenedor con borde sutil
        frame = tk.Frame(
            self.widgets_container,
            bg=COLOR_BG_CARD,
            bd=1,
            relief="solid",
            highlightbackground="#dadce0", # Borde gris Google Material
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        
        # Para hover effect
        def on_enter(e):
            frame.config(highlightbackground=COLOR_ACCENT, highlightthickness=1)
            
        def on_leave(e):
            frame.config(highlightbackground="#dadce0", highlightthickness=1)
            
        frame.bind("<Enter>", on_enter)
        frame.bind("<Leave>", on_leave)
        
        # Variable para el valor
        widget.value_var = tk.StringVar(value="--")
        
        # Label superior (nombre) - Texto secundario
        label = tk.Label(
            frame,
            text=widget.label.upper(), # Uppercase para estilo "etiqueta"
            font=("Segoe UI", 8, "bold"),
            bg=COLOR_BG_CARD,
            fg=COLOR_TEXT_SECONDARY,
            anchor="w"
        )
        label.pack(side="top", anchor="w", fill="x")
        
        # Valor principal - Texto primario grande
        value_label = tk.Label(
            frame,
            textvariable=widget.value_var,
            font=FONT_VALUE,
            bg=COLOR_BG_CARD,
            fg=COLOR_TEXT_PRIMARY,
            anchor="w"
        )
        value_label.pack(side="top", anchor="w", fill="x", pady=(2, 0))
        
        # Icono de tipo sutil en la esquina (opcional, por ahora solo texto)
        
        # Eventos
        for elem in (frame, label, value_label):
            elem.bind("<Button-3>", lambda e, w=widget: self._show_context_menu(e, w))
            
            if widget.widget_type == WIDGET_TYPE_SETTINGS:
                elem.bind("<Double-1>", lambda e, w=widget: self._on_widget_action(w))
                elem.config(cursor="hand2")
        
        widget.frame = frame
        return frame
    
    def _show_context_menu(self, event: tk.Event, widget: DashboardWidget):
        """Muestra menú contextual para un widget."""
        menu = tk.Menu(self.parent_frame, tearoff=0, font=("Segoe UI", 9))
        menu.add_command(label="✏️ Editar widget...", command=lambda: self._edit_widget(widget))
        menu.add_separator()
        menu.add_command(label="🗑️ Eliminar widget", command=lambda: self._delete_widget(widget))
        menu.add_separator()
        menu.add_command(label=f"⏱️ Intervalo: {widget.interval_sec}s", state="disabled")
        menu.tk_popup(event.x_root, event.y_root)
    
    def _on_add_widget(self):
        """Abre diálogo para añadir un nuevo widget."""
        dialog = AddWidgetDialog(self.app.root, self.app)
        if dialog.result:
            widget = DashboardWidget(
                widget_id=f"widget_{int(time.time()*1000)}",
                widget_type=dialog.result["type"],
                label=dialog.result["label"],
                config=dialog.result["config"],
                interval_sec=dialog.result.get("interval_sec", 5.0),
            )
            self.widgets.append(widget)
            self._save_widgets_to_config()
            self._rebuild_widget_ui()
    
    def _edit_widget(self, widget: DashboardWidget):
        """Edita un widget existente."""
        dialog = AddWidgetDialog(
            self.app.root,
            self.app,
            edit_mode=True,
            initial_data=widget.to_dict(),
        )
        if dialog.result:
            widget.label = dialog.result["label"]
            widget.config = dialog.result["config"]
            widget.interval_sec = dialog.result.get("interval_sec", 5.0)
            self._save_widgets_to_config()
            self._rebuild_widget_ui()
    
    def _delete_widget(self, widget: DashboardWidget):
        """Elimina un widget."""
        if messagebox.askyesno("Eliminar widget", f"¿Estás seguro de eliminar el widget '{widget.label}'?"):
            self.widgets = [w for w in self.widgets if w.widget_id != widget.widget_id]
            self._save_widgets_to_config()
            self._rebuild_widget_ui()
    
    def _on_widget_action(self, widget: DashboardWidget):
        """Ejecuta la acción principal del widget (ej: abrir ajustes)."""
        if widget.widget_type == WIDGET_TYPE_SETTINGS:
            tab_name = widget.config.get("tab", "General")
            if hasattr(self.app, "_open_settings_dialog"):
                self.app._open_settings_dialog(tab_name=tab_name)
    
    def _start_widget_updates(self, widget: DashboardWidget):
        """Inicia las actualizaciones periódicas para un widget con hilo daemon."""
        stop_event = threading.Event()
        self._stop_events[widget.widget_id] = stop_event
        
        def update_loop():
            while not stop_event.is_set():
                try:
                    value = self._fetch_widget_value(widget)
                    if widget.value_var and not stop_event.is_set():
                        try:
                            # Actualización segura en hilo principal
                            self.app.root.after(0, lambda v=value: widget.value_var.set(v))
                        except Exception:
                            pass
                except Exception as e:
                    LOGGER.debug("Error actualizando widget %s: %s", widget.widget_id, e)
                stop_event.wait(widget.interval_sec)
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        self._update_threads[widget.widget_id] = thread
    
    def _stop_all_updates(self):
        """Detiene todas las actualizaciones de widgets."""
        for stop_event in self._stop_events.values():
            stop_event.set()
        self._stop_events.clear()
        self._update_threads.clear()
    
    def _fetch_widget_value(self, widget: DashboardWidget) -> str:
        """Obtiene el valor actual de un widget según su tipo."""
        val = "--"
        if widget.widget_type == WIDGET_TYPE_PLC:
            val = self._fetch_plc_value(widget)
        elif widget.widget_type == WIDGET_TYPE_SETTINGS:
            val = self._fetch_settings_value(widget)
        elif widget.widget_type == WIDGET_TYPE_DETECTIONS:
            val = self._fetch_detection_count(widget)
        elif widget.widget_type == WIDGET_TYPE_RTSP_STATUS:
            val = self._fetch_rtsp_status(widget)
        elif widget.widget_type == WIDGET_TYPE_SYSTEM:
            val = self._fetch_system_value(widget)
        return val

    def _fetch_system_value(self, widget: DashboardWidget) -> str:
        """Obtiene métricas del sistema (CPU, RAM, Disco)."""
        resource = widget.config.get("resource", "cpu")
        try:
            import shutil
            import psutil  # type: ignore
        except ImportError:
            # Fallback simple para disco si no hay psutil
            if resource == "disk":
                try:
                    total, used, free = shutil.disk_usage("/")
                    percent = (used / total) * 100
                    return f"{percent:.1f}%"
                except:
                    pass
            return "N/A"

        try:
            if resource == "cpu":
                return f"{psutil.cpu_percent()}%"
            elif resource == "ram":
                return f"{psutil.virtual_memory().percent}%"
            elif resource == "disk":
                return f"{psutil.disk_usage('/').percent}%"
        except Exception:
            pass
        return "--"
    
    def _fetch_plc_value(self, widget: DashboardWidget) -> str:
        """Lee un valor del PLC con manejo robusto de errores."""
        try:
            from plc_bit_writer import PLC, parse_s7_address
            
            config = widget.config
            ip = config.get("ip", "")
            rack = int(config.get("rack", 0))
            slot = int(config.get("slot", 2))
            address = config.get("address", "")
            data_type = config.get("data_type", "UINT")
            
            if not ip or not address:
                return "🔧 Config"
            
            # Obtener o crear conexión PLC
            conn_key = f"{ip}:{rack}:{slot}"
            plc = self._plc_connections.get(conn_key)
            
            if plc is None or not plc.is_connected():
                plc = PLC()
                if not plc.connect(ip, rack, slot):
                    return "❌ Conexión"
                self._plc_connections[conn_key] = plc
            
            # Parsear dirección
            try:
                parsed = parse_s7_address(address)
            except ValueError:
                return "⚠️ Dirección"

            dbn = parsed["db_number"]
            offset = parsed["byte_offset"]
            
            val = None
            if data_type == "UINT":
                val = plc.read_db_word(dbn, offset, signed=False)
            elif data_type == "INT":
                val = plc.read_db_word(dbn, offset, signed=True)
            elif data_type == "REAL":
                v = plc.read_db_real(dbn, offset)
                return f"{v:.2f}"
            elif data_type == "BYTE":
                val = plc.read_db_byte(dbn, offset)
            elif data_type == "BOOL":
                bit = parsed.get("bit_offset", 0)
                b = plc.read_db_bit(dbn, offset, bit)
                return "ON" if b else "OFF"
            else:
                return "❓ Tipo"
            
            return str(val)
        except Exception as e:
            # LOGGER.debug("Error leyendo PLC: %s", e)
            return "❌ Error"
    
    def _fetch_settings_value(self, widget: DashboardWidget) -> str:
        """Retorna texto limpio para ajustes."""
        tab = widget.config.get("tab", "General")
        return f"⚙️ {tab}"
    
    def _fetch_detection_count(self, widget: DashboardWidget) -> str:
        """Obtiene el contador de detecciones."""
        try:
            mode = widget.config.get("mode", "total")  # total, by_class
            target_class = widget.config.get("class", "")
            
            # 1. Prioridad: Datos históricos recientes (suavizado)
            if hasattr(self.app, "_last_detection_counts"):
                counts = self.app._last_detection_counts
                if mode == "by_class" and target_class:
                    c = counts.get(target_class, 0)
                else:
                    c = sum(counts.values())
                return str(c)
            
            # 2. Fallback: Detecciones del frame actual
            if hasattr(self.app, "_current_frame_detections"):
                detections = self.app._current_frame_detections
                if mode == "by_class" and target_class:
                    c = len([d for d in detections if d.get("class") == target_class])
                else:
                    c = len(detections)
                return str(c)
            
            return "0"
        except Exception:
            return "--"
    
    def _fetch_rtsp_status(self, widget: DashboardWidget) -> str:
        """Obtiene el estado de la conexión RTSP con iconos."""
        try:
            target = widget.config.get("target", "input")  # input, output
            
            if target == "output":
                if hasattr(self.app, "_rtsp_out_connected") and self.app._rtsp_out_connected:
                    return "🟢 Conectado"
                if hasattr(self.app, "_ffmpeg_proc") and self.app._ffmpeg_proc:
                    return "🟡 Iniciando"
                return "🔴 Inactivo"
            else:  # input
                if hasattr(self.app, "running") and self.app.running:
                    if hasattr(self.app, "cap") and self.app.cap is not None:
                         if self.app.cap.isOpened():
                             # Opcional: mostrar FPS de entrada
                             fps = self.app.cap.get(5) # CAP_PROP_FPS
                             return f"🟢 {int(fps)} FPS"
                    return "🟡 Conectando"
                return "⚪ Detenido"
        except Exception:
            return "--"
    
    def cleanup(self):
        """Limpia recursos al cerrar."""
        self._stop_all_updates()
        for plc in self._plc_connections.values():
            try:
                plc.disconnect()
            except Exception:
                pass
        self._plc_connections.clear()


class AddWidgetDialog(simpledialog.Dialog):
    """Diálogo para añadir/editar un widget del dashboard."""
    
    def __init__(
        self,
        parent,
        app: "DetectorApp",
        edit_mode: bool = False,
        initial_data: dict | None = None,
    ):
        self.app = app
        self.edit_mode = edit_mode
        self.initial_data = initial_data or {}
        self.result: dict | None = None
        
        self.bg_color = "#f8f9fa" # Color de fondo suave para diálogo
        self._type_var = None
        self._label_var = None
        self._interval_var = None
        self._config_frame = None
        self._config_vars = {}
        
        title = "Editar Widget" if edit_mode else "Nuevo Widget"
        super().__init__(parent, title)

    def body(self, master):
        """Construye el cuerpo del diálogo."""
        # master.configure(bg=self.bg_color)
        master.columnconfigure(1, weight=1)
        row = 0
        
        pad_y = 6
        pad_x = 10
        
        # --- Tipo ---
        ttk.Label(master, text="Tipo de Widget:", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self._type_var = tk.StringVar(value=self.initial_data.get("type", WIDGET_TYPE_SETTINGS))
        type_combo = ttk.Combobox(
            master,
            textvariable=self._type_var,
            values=list(WIDGET_TYPES.values()),
            state="readonly" if not self.edit_mode else "disabled",
            width=28,
            font=("Segoe UI", 9)
        )
        type_combo.grid(row=row, column=1, sticky="ew", padx=pad_x, pady=pad_y)
        type_combo.bind("<<ComboboxSelected>>", self._on_type_change)
        
        initial_type = self.initial_data.get("type", WIDGET_TYPE_SETTINGS)
        if initial_type in WIDGET_TYPES:
            type_combo.set(WIDGET_TYPES[initial_type])
        
        row += 1
        
        # --- Etiqueta ---
        ttk.Label(master, text="Título (Etiqueta):", font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self._label_var = tk.StringVar(value=self.initial_data.get("label", "Mi Widget"))
        label_entry = ttk.Entry(master, textvariable=self._label_var, width=30, font=("Segoe UI", 9))
        label_entry.grid(row=row, column=1, sticky="ew", padx=pad_x, pady=pad_y)
        row += 1
        
        # --- Intervalo ---
        ttk.Label(master, text="Actualizar cada (s):", font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self._interval_var = tk.DoubleVar(value=self.initial_data.get("interval_sec", 1.0))
        interval_spin = ttk.Spinbox(
            master,
            from_=0.5,
            to=3600,
            increment=0.5,
            textvariable=self._interval_var,
            width=10,
            font=("Segoe UI", 9)
        )
        interval_spin.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        row += 1
        
        # Separador estético
        f = ttk.Frame(master)
        f.grid(row=row, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=pad_x)
        row += 1
        
        # --- Configuración Específica ---
        self._config_frame = ttk.LabelFrame(master, text=" Parámetros del Widget ", padding=10)
        self._config_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=pad_x, pady=5)
        self._config_frame.columnconfigure(1, weight=1)
        
        self._build_config_ui()
        
        return label_entry
    
    def _on_type_change(self, event=None):
        self._build_config_ui()
    
    def _get_selected_type(self) -> str:
        display = self._type_var.get()
        for key, val in WIDGET_TYPES.items():
            if val == display:
                return key
        return WIDGET_TYPE_SETTINGS
    
    def _build_config_ui(self):
        for child in self._config_frame.winfo_children():
            child.destroy()
        self._config_vars.clear()
        
        widget_type = self._get_selected_type()
        initial_config = self.initial_data.get("config", {})
        
        if widget_type == WIDGET_TYPE_PLC:
            self._build_plc_config(initial_config)
        elif widget_type == WIDGET_TYPE_SETTINGS:
            self._build_settings_config(initial_config)
        elif widget_type == WIDGET_TYPE_DETECTIONS:
            self._build_detections_config(initial_config)
        elif widget_type == WIDGET_TYPE_RTSP_STATUS:
            self._build_rtsp_config(initial_config)
        elif widget_type == WIDGET_TYPE_SYSTEM:
            self._build_system_config(initial_config)

    def _build_system_config(self, initial: dict):
        ttk.Label(self._config_frame, text="Recurso:").grid(row=0, column=0, sticky="w", padx=5, pady=15)
        self._config_vars["resource"] = tk.StringVar(value=initial.get("resource", "cpu"))
        ttk.Combobox(
            self._config_frame,
            textvariable=self._config_vars["resource"],
            values=["cpu", "ram", "disk"],
            state="readonly",
            width=15,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=15)
    
    def _build_plc_config(self, initial: dict):
        row = 0
        pad = 4
        
        # IP
        ttk.Label(self._config_frame, text="IP PLC:").grid(row=row, column=0, sticky="w", padx=pad, pady=pad)
        self._config_vars["ip"] = tk.StringVar(value=initial.get("ip", "192.168.1.100"))
        ttk.Entry(self._config_frame, textvariable=self._config_vars["ip"], width=18).grid(
            row=row, column=1, sticky="w", padx=pad, pady=pad
        )
        row += 1
        
        # Rack / Slot
        f_rs = ttk.Frame(self._config_frame)
        f_rs.grid(row=row, column=0, columnspan=2, sticky="w", padx=pad, pady=pad)
        
        ttk.Label(f_rs, text="Rack:").pack(side="left")
        self._config_vars["rack"] = tk.IntVar(value=initial.get("rack", 0))
        ttk.Spinbox(f_rs, from_=0, to=7, width=3, textvariable=self._config_vars["rack"]).pack(side="left", padx=(2, 10))
        
        ttk.Label(f_rs, text="Slot:").pack(side="left")
        self._config_vars["slot"] = tk.IntVar(value=initial.get("slot", 2))
        ttk.Spinbox(f_rs, from_=0, to=31, width=3, textvariable=self._config_vars["slot"]).pack(side="left", padx=2)
        row += 1
        
        # Dirección
        ttk.Label(self._config_frame, text="Dirección (S7):").grid(row=row, column=0, sticky="w", padx=pad, pady=pad)
        self._config_vars["address"] = tk.StringVar(value=initial.get("address", "DB1.DBW0"))
        ttk.Entry(self._config_frame, textvariable=self._config_vars["address"], width=18).grid(
            row=row, column=1, sticky="w", padx=pad, pady=pad
        )
        # Hint debajo
        ttk.Label(self._config_frame, text="(ej: DB5.DBD10, M10.0)", font=("Segoe UI", 7), foreground="gray").grid(row=row+1, column=1, sticky="w", padx=pad)
        row += 2
        
        # Tipo
        ttk.Label(self._config_frame, text="Tipo de dato:").grid(row=row, column=0, sticky="w", padx=pad, pady=pad)
        self._config_vars["data_type"] = tk.StringVar(value=initial.get("data_type", "UINT"))
        ttk.Combobox(
            self._config_frame,
            textvariable=self._config_vars["data_type"],
            values=["UINT", "INT", "REAL", "BYTE", "BOOL"],
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky="w", padx=pad, pady=pad)

    def _build_settings_config(self, initial: dict):
        ttk.Label(self._config_frame, text="Abrir pestaña:").grid(row=0, column=0, sticky="w", padx=5, pady=15)
        self._config_vars["tab"] = tk.StringVar(value=initial.get("tab", "General"))
        ttk.Combobox(
            self._config_frame,
            textvariable=self._config_vars["tab"],
            values=SETTINGS_TABS,
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=15)
    
    def _build_detections_config(self, initial: dict):
        row = 0
        pad = 5
        
        ttk.Label(self._config_frame, text="Modo de conteo:").grid(row=row, column=0, sticky="w", padx=pad, pady=pad)
        self._config_vars["mode"] = tk.StringVar(value=initial.get("mode", "total"))
        cb = ttk.Combobox(
            self._config_frame,
            textvariable=self._config_vars["mode"],
            values=["total", "by_class"],
            state="readonly",
            width=15,
        )
        cb.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
        row += 1
        
        ttk.Label(self._config_frame, text="Clase específica:").grid(row=row, column=0, sticky="w", padx=pad, pady=pad)
        self._config_vars["class"] = tk.StringVar(value=initial.get("class", ""))
        
        available_classes = []
        if hasattr(self.app, "class_cfg"):
            available_classes = list(self.app.class_cfg.keys())
        
        class_combo = ttk.Combobox(
            self._config_frame,
            textvariable=self._config_vars["class"],
            values=available_classes,
            width=20,
        )
        class_combo.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
    
    def _build_rtsp_config(self, initial: dict):
        ttk.Label(self._config_frame, text="Monitorizar:").grid(row=0, column=0, sticky="w", padx=5, pady=15)
        self._config_vars["target"] = tk.StringVar(value=initial.get("target", "input"))
        ttk.Combobox(
            self._config_frame,
            textvariable=self._config_vars["target"],
            values=["input", "output"],
            state="readonly",
            width=15,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=15)
    
    def apply(self):
        widget_type = self._get_selected_type()
        config = {k: v.get() for k, v in self._config_vars.items() if v.get() is not None}
        
        self.result = {
            "type": widget_type,
            "label": self._label_var.get().strip() or "Widget",
            "config": config,
            "interval_sec": max(0.5, self._interval_var.get()),
        }
    
    def buttonbox(self):
        box = ttk.Frame(self, padding=(5, 10))
        
        args = {"width": 12, "padding": 5}
        
        ttk.Button(box, text="Guardar", command=self.ok, **args).pack(side="left", padx=5)
        ttk.Button(box, text="Cancelar", command=self.cancel, **args).pack(side="left", padx=5)
        
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()


class FPSConfigDialog(simpledialog.Dialog):
    """Diálogo para configurar opciones visuales de FPS."""
    
    def __init__(self, parent, initial_config: dict):
        self.initial_config = initial_config
        self.result: dict | None = None
        
        self.vars = {}
        
        super().__init__(parent, "Configuración FPS")
        
    def body(self, master):
        master.columnconfigure(1, weight=1)
        row = 0
        pad = 6
        
        # Intervalo
        ttk.Label(master, text="Intervalo actualización (s):").grid(row=row, column=0, sticky="w", padx=pad, pady=pad)
        self.vars["interval"] = tk.DoubleVar(value=self.initial_config.get("interval", 0.5))
        ttk.Spinbox(master, from_=0.1, to=5.0, increment=0.1, textvariable=self.vars["interval"], width=8).grid(
            row=row, column=1, sticky="w", padx=pad, pady=pad
        )
        row += 1
        
        ttk.Separator(master, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        ttk.Label(master, text="Umbrales y Colores", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", padx=pad, pady=pad)
        row += 1
        
        # Grid para umbrales
        # Bajo (< X) -> Rojo
        self._build_threshold_row(master, row, "Bajo (< X):", "low_thresh", "low_color", 15.0, "#d32f2f")
        row += 1
        
        # Medio (< Y) -> Naranja
        self._build_threshold_row(master, row, "Medio (< Y):", "med_thresh", "med_color", 24.0, "#f57c00")
        row += 1
        
        # Alto (>= Y) -> Verde
        self._build_threshold_row(master, row, "Alto (>= Y):", None, "high_color", None, "#388e3c") # Threshold implícito
        row += 1

    def _build_threshold_row(self, master, row, label_text, thresh_key, color_key, default_thresh, default_color):
        ttk.Label(master, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        f = ttk.Frame(master)
        f.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        if thresh_key:
            self.vars[thresh_key] = tk.IntVar(value=self.initial_config.get(thresh_key, int(default_thresh)))
            ttk.Spinbox(f, from_=1, to=120, width=5, textvariable=self.vars[thresh_key]).pack(side="left", padx=(0, 10))
        
        # Botón de color
        initial_color = self.initial_config.get(color_key, default_color)
        self.vars[color_key] = tk.StringVar(value=initial_color)
        
        btn = tk.Button(f, text=" ■ ", bg=initial_color, fg=initial_color, width=3, relief="flat",
                        command=lambda k=color_key, v=self.vars[color_key]: self._pick_color(k, v, btn)) # type: ignore
        btn.pack(side="left")
        
        # Update btn color hook
        btn.config(command=lambda: self._pick_color(color_key, self.vars[color_key], btn))

    def _pick_color(self, key, var, btn):
        from tkinter import colorchooser
        color = colorchooser.askcolor(color=var.get(), title="Elegir color")[1] # [1] es hex
        if color:
            var.set(color)
            btn.config(bg=color, fg=color)

    def apply(self):
        self.result = {k: v.get() for k, v in self.vars.items()}
