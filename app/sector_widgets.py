# -*- coding: utf-8 -*-
"""
SectorControlPanel - Widget interactivo para visualización y control de sectores.

Este módulo externaliza la lógica de dibujo del diagrama de sectores que antes
estaba embebida en el diálogo de ajustes de detect_manchas_gui_rtsp.py.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict, List, Tuple


class SectorControlPanel(ttk.Frame):
    """
    Widget interactivo para visualización y control de sectores.
    
    Características:
    - Dibuja una malla de sectores con numeración automática
    - Escalado dinámico según el tamaño del widget
    - Chips de clases con restricciones visuales integradas
    - Interactividad: exclusión, sensibilidad, restricciones de clase
    """
    
    # Modos de restricción y sus colores
    RESTRICTION_MODES = ["sin_restriccion", "solo_malla", "solo_fuera_malla", "solo_sectores"]
    RESTRICTION_COLORS = {
        "sin_restriccion": {"bg": "#ffffff", "border": "#bdbdbd", "text": "#757575"},
        "solo_malla": {"bg": "#e8f5e9", "border": "#66bb6a", "text": "#1b5e20"},
        "solo_fuera_malla": {"bg": "#ffebee", "border": "#ef5350", "text": "#b71c1c"},
        "solo_sectores": {"bg": "#e3f2fd", "border": "#42a5f5", "text": "#0d47a1"},
    }
    RESTRICTION_LABELS = {
        "sin_restriccion": "Todo",
        "solo_malla": "✓ Malla",
        "solo_fuera_malla": "✗ Fuera",
        "solo_sectores": "Sectores",
    }
    
    def __init__(
        self,
        master,
        width: int = 200,
        height: int = 200,
        on_change: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self._on_change = on_change
        self._modo = "vertical"
        self._n_vert = 5
        self._n_horiz = 1
        self._sector_states: Dict[int, bool] = {}  # sector_id -> enabled
        self._sector_sensibilities: Dict[int, float] = {}  # sector_id -> multiplier
        
        # NUEVO: Restricciones por clase
        self._class_restrictions: Dict[str, Dict] = {}  # clase -> {"modo": str, "sectores": list}
        self._available_classes: List[str] = []
        self._selected_class: Optional[str] = None  # Clase seleccionada para asignar sectores
        
        # NUEVO: Clases de delimitación de borde (vista cenital)
        self._border_classes: Dict[str, str] = {"top": "", "bottom": "", "left": "", "right": ""}
        
        # Colores Premium / Material Design
        self._colors = {
            "bg": "#ffffff",
            "line": "#e0e0e0",
            "text": "#37474f",
            "text_dim": "#90a4ae",
            "disabled": "#f5f5f5",
            "disabled_pattern": "#e0e0e0",
            "highlight": "#2196f3",      # Blue 500
            "highlight_bg": "#e3f2fd",   # Blue 50
            "success": "#4caf50",        # Green 500
            "success_bg": "#e8f5e9",     # Green 50
            "error": "#f44336",          # Red 500
            "error_bg": "#ffebee",       # Red 50
            "sens": "#ff9800",           # Orange 500
            "border_bg": "#cfd8dc",      # Blue Grey 100
            "border_text": "#455a64"     # Blue Grey 700
        }
        
        # Compatibilidad hacia atrás (deprecated)
        self._color_bg = self._colors["bg"]
        self._color_line = "#333333" # Mantener contraste alto para líneas de grid visual
        self._color_text = self._colors["text"]
        self._color_sens = self._colors["sens"]
        self._color_disabled = self._colors["disabled"]
        self._color_disabled_pattern = self._colors["disabled_pattern"]
        self._color_class_highlight = self._colors["highlight"]
        
        # Frame contenedor principal
        container = ttk.LabelFrame(self, text="Panel de Sectores Interactivo", padding=8)
        container.pack(fill="both", expand=True)
        
        # Canvas para sectores (parte superior)
        self.canvas = tk.Canvas(
            container,
            bg=self._color_bg,
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(width=width, height=height)
        
        # Bindings del canvas
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        
        # Instrucciones
        self._lbl_instructions = ttk.Label(container, text="L-Click: On/Off | R-Click: Sensibilidad", 
                  font=("Segoe UI", 8, "italic"), foreground="#666")
        self._lbl_instructions.pack(anchor="e", pady=(4, 0))
        
        # Separador
        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=8)
        
        # NUEVO: Frame para chips de clases
        self._class_frame = ttk.Frame(container)
        self._class_frame.pack(fill="x", pady=(0, 4))
        
        ttk.Label(self._class_frame, text="🎯 Restricciones por clase:", 
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        
        # Frame scrollable para chips
        self._chips_container = tk.Frame(self._class_frame, bg="white")
        self._chips_container.pack(fill="x", pady=4)
        
        # Label de ayuda para modo selección de sectores
        self._lbl_class_mode = ttk.Label(container, text="", 
                                         font=("Segoe UI", 8, "italic"), foreground="#2196f3")
        self._lbl_class_mode.pack(anchor="w")
        
        # Dibujo inicial
        self.after(200, self.draw)
    
    def set_available_classes(self, classes: List[str]) -> None:
        """Establece las clases disponibles para mostrar como chips."""
        self._available_classes = sorted(classes)
        for cls in self._available_classes:
            if cls not in self._class_restrictions:
                self._class_restrictions[cls] = {"modo": "sin_restriccion", "sectores": []}
        self._draw_class_chips()
    
    def set_border_classes(self, top: str = "", bottom: str = "", left: str = "", right: str = "") -> None:
        """Establece las clases de delimitación para la vista cenital."""
        self._border_classes = {"top": top, "bottom": bottom, "left": left, "right": right}
        self.draw()
    
    def get_class_restrictions(self) -> Dict[str, Dict]:
        """Retorna las restricciones de clase configuradas."""
        return dict(self._class_restrictions)
    
    def set_class_restrictions(self, restrictions: Dict[str, Dict]) -> None:
        """Carga restricciones de clase desde un dict."""
        self._class_restrictions = dict(restrictions)
        self._draw_class_chips()
    
    def _draw_class_chips(self) -> None:
        """Dibuja los chips de clases con estilo moderno."""
        # Limpiar chips existentes
        for widget in self._chips_container.winfo_children():
            widget.destroy()
        
        if not self._available_classes:
            ttk.Label(self._chips_container, text="(carga un modelo para configurar restricciones)", 
                      font=("Segoe UI", 8, "italic"), foreground="#90a4ae").pack(anchor="w", padx=4)
            return
        
        # Crear chips en filas (flow layout)
        row_frame = tk.Frame(self._chips_container, bg=self._colors["bg"]) # usar bg del tema
        row_frame.pack(fill="x", anchor="w")
        
        chips_in_row = 0
        max_per_row = 4
        
        for clase in self._available_classes:
            if chips_in_row >= max_per_row:
                row_frame = tk.Frame(self._chips_container, bg=self._colors["bg"])
                row_frame.pack(fill="x", anchor="w", pady=4)
                chips_in_row = 0
            
            cfg = self._class_restrictions.get(clase, {"modo": "sin_restriccion", "sectores": []})
            modo = cfg.get("modo", "sin_restriccion")
            sectores = cfg.get("sectores", [])
            
            colors = self.RESTRICTION_COLORS.get(modo, self.RESTRICTION_COLORS["sin_restriccion"])
            label_base = self.RESTRICTION_LABELS.get(modo, "?")
            
            # Texto descriptivo
            if modo == "solo_sectores" and sectores:
                # Mostrar lista compacta
                sec_list = [str(s+1) for s in sectores]
                sec_str = ",".join(sec_list[:3])
                if len(sec_list) > 3: sec_str += "..."
                chip_text = f"{clase}: [{sec_str}]"
            else:
                chip_text = f"{clase}: {label_base}"
            
            # Estado de selección (borde naranja grueso si seleccionado)
            is_selected = (self._selected_class == clase)
            border_color = "#ff9800" if is_selected else colors["border"]
            border_width = 2 if is_selected else 1
            
            # Chip Container (Borde)
            chip = tk.Frame(row_frame, bg=border_color, padx=border_width, pady=border_width)
            chip.pack(side="left", padx=4, pady=2)
            
            # Chip Content
            inner = tk.Label(chip, text=chip_text, bg=colors["bg"], fg=colors["text"],
                            font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2")
            inner.pack(fill="both", expand=True)
            
            # Bindings
            inner.bind("<Button-1>", lambda e, c=clase: self._on_class_left_click(c))
            inner.bind("<Button-3>", lambda e, c=clase: self._on_class_right_click(c))
            
            # Hover effect simple
            def on_enter(e, w=inner, c=colors):
                w.config(bg="#fafafa" if c["bg"]=="#ffffff" else c["bg"]) # highlight slightly? Tkinter limita esto
            
            #inner.bind("<Enter>", on_enter) 
            
            chips_in_row += 1
            
        # Añadir leyenda/tip compacta
        tip_frame = ttk.Frame(self._chips_container)
        tip_frame.pack(fill="x", pady=(6,0))
        ttk.Label(tip_frame, text="Info: L-Click cambia modo | R-Click selecciona sectores",
                  font=("Segoe UI", 7), foreground="#90a4ae").pack(side="right")
    
    def _on_class_left_click(self, clase: str) -> None:
        """Cicla el modo de restricción de la clase."""
        cfg = self._class_restrictions.get(clase, {"modo": "sin_restriccion", "sectores": []})
        current_mode = cfg.get("modo", "sin_restriccion")
        
        # Ciclar al siguiente modo
        try:
            idx = self.RESTRICTION_MODES.index(current_mode)
            next_idx = (idx + 1) % len(self.RESTRICTION_MODES)
            new_mode = self.RESTRICTION_MODES[next_idx]
        except ValueError:
            new_mode = "sin_restriccion"
        
        self._class_restrictions[clase] = {"modo": new_mode, "sectores": cfg.get("sectores", [])}
        
        # Si cambia a modo sectores, activar selección
        if new_mode == "solo_sectores":
            self._selected_class = clase
            self._lbl_class_mode.config(text=f"▶ Haz clic en los sectores para '{clase}'")
            self._lbl_instructions.config(text="L-Click en sector: asignar/quitar clase")
        else:
            self._selected_class = None
            self._lbl_class_mode.config(text="")
            self._lbl_instructions.config(text="L-Click: On/Off | R-Click: Sensibilidad")
        
        self._draw_class_chips()
        self.draw()
        if self._on_change:
            self._on_change()
    
    def _on_class_right_click(self, clase: str) -> None:
        """Activa modo selección de sectores para esta clase."""
        cfg = self._class_restrictions.get(clase, {"modo": "sin_restriccion", "sectores": []})
        
        # Cambiar a modo solo_sectores si no lo está
        if cfg.get("modo") != "solo_sectores":
            self._class_restrictions[clase] = {"modo": "solo_sectores", "sectores": cfg.get("sectores", [])}
        
        # Activar selección
        self._selected_class = clase
        self._lbl_class_mode.config(text=f"▶ Haz clic en los sectores para '{clase}' | Clic derecho aquí para terminar")
        self._lbl_instructions.config(text="L-Click en sector: asignar/quitar | R-Click chip: finalizar")
        
        self._draw_class_chips()
        self.draw()
    
    def update_layout(self, modo: str, n_vert: int, n_horiz: int) -> None:
        """Actualiza la configuración de la malla y redibuja."""
        self._modo = modo
        self._n_vert = max(1, n_vert)
        self._n_horiz = max(1, n_horiz)
        self.draw()
    
    def _draw_pill(self, x: float, y: float, text: str, bg_color: str, text_color: str, font_size: int = 8, anchor: str = "nw") -> None:
        """Dibuja una etiqueta tipo 'pill' con texto."""
        padding_x = 4
        padding_y = 1
        # Estimar ancho texto (aproximado)
        char_w = font_size * 0.6
        w = len(text) * char_w + 2 * padding_x
        h = font_size * 1.5 + 2 * padding_y
        
        # Ajustar coords según anchor
        if "e" in anchor: x -= w
        if "s" in anchor: y -= h
        if "c" in anchor: 
            x -= w/2
            y -= h/2
            
        r = 4 # radio
        self.canvas.create_rectangle(x, y, x+w, y+h, fill=bg_color, outline=bg_color, width=0, tags=("pill", "sector_info"))
        self.canvas.create_text(x + w/2, y + h/2, text=text, fill=text_color, 
                               font=("Segoe UI", font_size, "bold"), tags=("pill_text", "sector_info"))

    def draw(self) -> None:
        """Dibuja la malla de sectores con estilo High-Fidelity e indicadores 'Ricos'."""
        canvas = self.canvas
        canvas.delete("all")
        
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 400, 300
        
        # ─── 0. PRECALCULO DE RESTRICCIONES (Necesario para layout) ───────
        sector_pills = {} 
        outside_pills = []
        
        for clase, cfg in self._class_restrictions.items():
            modo = cfg.get("modo", "sin_restriccion")
            if modo == "solo_malla":
                p_bg, p_fg = self._colors["success_bg"], self._colors["success"]
                for sid in range(1, self._n_vert*self._n_horiz+1):
                    sector_pills.setdefault(sid, []).append({"text": clase, "icon": "✓", "bg": p_bg, "fg": p_fg})
            elif modo == "solo_fuera_malla":
                outside_pills.append({"text": clase, "bg": self._colors["error_bg"], "fg": self._colors["error"]})
            elif modo == "solo_sectores":
                p_bg, p_fg = self._colors["highlight_bg"], self._colors["highlight"]
                for sec_idx in cfg.get("sectores", []):
                    sid = sec_idx + 1
                    sector_pills.setdefault(sid, []).append({"text": clase, "icon": "", "bg": p_bg, "fg": p_fg})

        # ─── 1. LAYOUT DINÁMICO ───────────────────────────────────────────
        margin = 10
        border_margin = 25
        zona_externa_h = 30 if outside_pills else 0
        
        sector_x1 = margin + border_margin
        sector_y1 = margin + border_margin + zona_externa_h
        sector_x2 = width - margin - border_margin
        sector_y2 = height - margin - border_margin
        
        draw_width = max(10, sector_x2 - sector_x1)
        draw_height = max(10, sector_y2 - sector_y1)
        
        n_cols = self._n_vert
        n_rows = self._n_horiz
        if self._modo == "vertical": n_rows = 1
        elif self._modo == "horizontal": n_cols = 1
        
        cell_w = draw_width / n_cols if n_cols > 0 else draw_width
        cell_h = draw_height / n_rows if n_rows > 0 else draw_height
        
        min_cell_size = min(cell_w, cell_h)
        font_size = max(8, min(16, int(min_cell_size / 4)))

        # ─── 2. DIBUJAR COMPONENTES DE CABECERA Y BORDES ──────────────────
        c_border_bg, c_border_txt = self._colors["border_bg"], self._colors["border_text"]
        f_border = ("Segoe UI", 7, "bold")
        
        def draw_border_box(bx1, by1, bx2, by2, text, icon):
            if not text: return
            canvas.create_rectangle(bx1, by1, bx2, by2, fill=c_border_bg, outline=c_border_bg)
            cx, cy = (bx1+bx2)/2, (by1+by2)/2
            angle = 90 if (bx2-bx1) < (by2-by1) else 0
            canvas.create_text(cx, cy, text=f"{icon} {text[:12]}", fill=c_border_txt, font=f_border, angle=angle)

        # Borde superior (siempre arriba del todo)
        draw_border_box(sector_x1, margin, sector_x2, margin + border_margin, self._border_classes.get("top"), "▲")
        
        # Zona Externa (si aplica, entre borde superior y grid)
        if outside_pills:
            franja_y = margin + border_margin + 4
            franja_h = zona_externa_h - 8
            canvas.create_rectangle(sector_x1, franja_y, sector_x2, franja_y + franja_h, 
                                   fill="#fff5f5", outline="#ffcdd2", tags="zona_externa")
            canvas.create_text(sector_x1 + 6, franja_y + franja_h/2, text="ZONA EXTERNA:", 
                               font=("Segoe UI", 7, "bold"), fill="#c62828", anchor="w")
            out_x = sector_x1 + 85
            for p in outside_pills:
                self._draw_pill(out_x, franja_y + 4, p["text"], p["bg"], p["fg"], font_size=7)
                out_x += len(p["text"])*6 + 18
                if out_x > sector_x2 - 10: break

        # Otros bordes
        draw_border_box(sector_x1, sector_y2, sector_x2, height-margin, self._border_classes.get("bottom"), "▼")
        draw_border_box(margin, sector_y1, sector_x1, sector_y2, self._border_classes.get("left"), "◀")
        draw_border_box(sector_x2, sector_y1, width-margin, sector_y2, self._border_classes.get("right"), "▶")

        # ─── 3. DIBUJAR GRID DE SECTORES ──────────────────────────────────
        selected_sectors = self._class_restrictions.get(self._selected_class, {}).get("sectores", []) if self._selected_class else []
        
        sector_id = 1
        for row in range(n_rows):
            for col in range(n_cols):
                x1, y1 = sector_x1 + col * cell_w, sector_y1 + row * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                
                is_enabled = self._sector_states.get(sector_id, True)
                sens = self._sector_sensibilities.get(sector_id, 1.0)
                is_class_selected = (sector_id - 1) in selected_sectors
                
                # Fondo y Borde
                fill = self._colors["bg"]
                outline = self._colors["line"]
                width_line = 1
                
                if not is_enabled:
                    fill, outline = "#ffebee", "#ef9a9a"
                elif is_class_selected and self._selected_class:
                    fill, outline, width_line = self._colors["highlight_bg"], self._colors["highlight"], 2
                elif sens != 1.0:
                    fill = "#fff8e1" if sens < 1.0 else "#e0f7fa"
                elif self._selected_class:
                    outline = "#eeeeee" 

                canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=width_line, tags=("sector", f"s_{sector_id}"))
                
                if not is_enabled:
                    # Mallado Rojo
                    for i in range(int((cell_w + cell_h) / 10)):
                        offset = i * 10
                        if offset < cell_w + cell_h:
                            canvas.create_line(x1 + offset, y1, x1, y1 + offset, fill="#ffcdd2", width=1)
                
                # Contenido
                if min_cell_size > 20:
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    num_color = "#f8bbd0" if not is_enabled else ("#bbdefb" if is_class_selected else "#f0f0f0")
                    canvas.create_text(cx, cy, text=str(sector_id), font=("Segoe UI", int(font_size*2.5), "bold"), fill=num_color)
                    
                    if not is_enabled:
                        canvas.create_text(cx, y2-8, text="EXCLUIDO", font=("Segoe UI", max(6, int(font_size*0.5)), "bold"), fill="#e57373")
                    elif is_class_selected and self._selected_class:
                        canvas.create_text(cx, cy, text="✓", font=("Segoe UI", int(font_size*2)), fill=self._colors["highlight"])

                    if is_enabled:
                        pills = sector_pills.get(sector_id, [])
                        pill_y = y1 + 4
                        for p in pills:
                            txt = p["text"] + (" (Malla)" if p["icon"] == "✓" else "")
                            fs = max(7, int(font_size*0.6)) if len(pills) <= 3 else max(6, int(font_size*0.6)-1)
                            self._draw_pill(x1 + 4, pill_y, txt, p["bg"], p["fg"], font_size=fs)
                            pill_y += (fs * 1.5 + 4)
                            if pill_y > y2 - 15: break

                        if sens != 1.0:
                             canvas.create_text(x2-4, y2-4, text=f"{sens}x", anchor="se", fill=self._colors["sens"], font=("Segoe UI", int(font_size*0.8), "bold"))
                sector_id += 1
    
    def set_sector_enabled(self, sector_id: int, enabled: bool) -> None:
        """Habilita o deshabilita un sector."""
        self._sector_states[sector_id] = enabled
        self.draw()
        if self._on_change: self._on_change()
    
    def set_sector_sensibility(self, sector_id: int, sens: float) -> None:
        """Ajusta la sensibilidad de un sector."""
        self._sector_sensibilities[sector_id] = round(sens, 2)
        self.draw()
        if self._on_change: self._on_change()

    def get_config(self) -> Dict:
        """Retorna la configuración actual para guardar en JSON."""
        return {
            "excluidos": self.get_excluded_sectors(),
            "sensibilidades": {str(k): v for k, v in self._sector_sensibilities.items() if v != 1.0},
            "restricciones_clase": self._class_restrictions,
        }
    
    def set_config(self, config: Dict) -> None:
        """Carga configuración desde un dict."""
        self.set_excluded_sectors(config.get("excluidos", []))
        sens = config.get("sensibilidades")
        if not isinstance(sens, dict):
            sens = config.get("ajustes_locales", {})
        if not isinstance(sens, dict):
            sens = {}
        self._sector_sensibilities = {int(k): float(v) for k, v in sens.items()}
        if "restricciones_clase" in config:
            self._class_restrictions = dict(config["restricciones_clase"])
        self._draw_class_chips()
        self.draw()

    def get_excluded_sectors(self) -> List[int]:
        return [sid for sid, enabled in self._sector_states.items() if not enabled]
    
    def set_excluded_sectors(self, excluded: List[int]) -> None:
        self._sector_states.clear()
        for sid in excluded:
            self._sector_states[sid] = False
        self.draw()
    
    def _get_sector_at(self, x: float, y: float) -> Optional[int]:
        canvas = self.canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1: return None
        
        # Calcular si hay zona externa (para ajustar offset de clics)
        has_outside = False
        for cfg in self._class_restrictions.values():
            if cfg.get("modo") == "solo_fuera_malla":
                has_outside = True
                break
        
        # Usar los mismos márgenes que draw()
        border_margin = 25
        margin = 10
        zona_externa_h = 30 if has_outside else 0
        
        sector_x1 = margin + border_margin
        sector_y1 = margin + border_margin + zona_externa_h
        sector_x2 = width - margin - border_margin
        sector_y2 = height - margin - border_margin
        
        draw_width = max(10, sector_x2 - sector_x1)
        draw_height = max(10, sector_y2 - sector_y1)
        
        n_cols = self._n_vert
        n_rows = self._n_horiz
        if self._modo == "vertical": n_rows = 1
        elif self._modo == "horizontal": n_cols = 1
        
        if n_cols <= 0 or n_rows <= 0: return None
        
        cell_w = draw_width / n_cols
        cell_h = draw_height / n_rows
        
        # Verificar si está dentro del área de sectores
        if x < sector_x1 or x > sector_x2 or y < sector_y1 or y > sector_y2:
            return None
        
        col = int((x - sector_x1) / cell_w)
        row = int((y - sector_y1) / cell_h)
        
        if 0 <= col < n_cols and 0 <= row < n_rows:
            return row * n_cols + col + 1
        return None
    
    def _on_resize(self, event) -> None:
        self.draw()
    
    def _on_left_click(self, event) -> None:
        sector_id = self._get_sector_at(event.x, event.y)
        if sector_id is None:
            return
        
        # Si hay clase seleccionada en modo sectores, asignar/quitar sector
        if self._selected_class:
            cfg = self._class_restrictions.get(self._selected_class, {"modo": "solo_sectores", "sectores": []})
            sectores = list(cfg.get("sectores", []))
            sector_idx = sector_id - 1  # 0-based interno
            
            if sector_idx in sectores:
                sectores.remove(sector_idx)
            else:
                sectores.append(sector_idx)
            
            self._class_restrictions[self._selected_class] = {"modo": "solo_sectores", "sectores": sectores}
            self._draw_class_chips()
            self.draw()
            if self._on_change:
                self._on_change()
        else:
            # Comportamiento normal: toggle sector enabled
            current = self._sector_states.get(sector_id, True)
            self.set_sector_enabled(sector_id, not current)
    
    def _on_right_click(self, event) -> None:
        """Abre popup de sensibilidad o cancela selección de clase."""
        # Si hay clase seleccionada, cancelar selección
        if self._selected_class:
            self._selected_class = None
            self._lbl_class_mode.config(text="")
            self._lbl_instructions.config(text="L-Click: On/Off | R-Click: Sensibilidad")
            self._draw_class_chips()
            self.draw()
            return
        
        sector_id = self._get_sector_at(event.x, event.y)
        if sector_id is None:
            return
        
        if not self._sector_states.get(sector_id, True):
            return

        curr_sens = self._sector_sensibilities.get(sector_id, 1.0)
        
        pop = tk.Toplevel(self)
        pop.title(f"Sensibilidad Sector {sector_id}")
        pop.geometry(f"+{event.x_root}+{event.y_root}")
        pop.resizable(False, False)
        pop.attributes("-toolwindow", True)
        pop.attributes("-topmost", True)
        
        frm = ttk.Frame(pop, padding=10)
        frm.pack()
        
        ttk.Label(frm, text=f"Ajustar sensibilidad para Sector {sector_id}", font=("Segoe UI", 9, "bold")).pack(pady=(0, 10))
        
        val_var = tk.DoubleVar(value=curr_sens)
        lbl_val = ttk.Label(frm, text=f"{curr_sens:.2f}x")
        
        def _update_sens(v):
            val = round(float(v), 2)
            val_var.set(val)
            lbl_val.config(text=f"{val:.2f}x")
            self.set_sector_sensibility(sector_id, val)
            
        scale = ttk.Scale(frm, from_=0.1, to=2.0, variable=val_var, orient="horizontal", length=150, command=_update_sens)
        scale.pack(side="left", padx=5)
        lbl_val.pack(side="left", padx=5)
        
        ttk.Button(frm, text="OK", width=5, command=pop.destroy).pack(side="right", padx=(10, 0))
        
        pop.bind("<FocusOut>", lambda e: pop.destroy())


