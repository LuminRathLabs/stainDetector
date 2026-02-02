# -*- coding: utf-8 -*-
"""
Módulo Sectorizador V2 - Sistema Modular de Sectorización Dinámica con Perspectiva

Este módulo implementa la delimitación dinámica de la región útil de una banda
de acero y su división en sectores configurables para el sistema de detección
de defectos.

Características:
- Delimitación dinámica con ROI cuadrilátero (trapezoidal) que sigue la perspectiva
- Homografía para mapeo imagen <-> coordenadas rectificadas de banda
- Sectorización que sigue la geometría real de la banda (no axis-aligned)
- Suavizado robusto con EMA y rechazo de outliers
- Visualización profesional con sombras y líneas proyectadas
- Asignación de detecciones a sectores usando coordenadas (u,v) rectificadas
"""

from __future__ import annotations

import time
import math
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import deque

import cv2
import numpy as np

LOGGER = logging.getLogger("Sectorizador")


@dataclass
class ConfigBordes:
    """Configuración de clases de borde para delimitar la banda."""
    clase_superior: str = ""
    clase_inferior: str = ""
    clase_izquierdo: str = ""
    clase_derecho: str = ""


# ===================== RESTRICCIONES POR CLASE (NUEVO) =====================

@dataclass
class RestriccionClase:
    """Restricción de filtrado para una clase de detección."""
    modo: str = "sin_restriccion"  # sin_restriccion, solo_malla, solo_sectores, solo_fuera_malla
    sectores: List[int] = field(default_factory=list)  # Solo para modo "solo_sectores"


@dataclass
class ConfigRestricciones:
    """Configuración de restricciones de detección por clase."""
    enabled: bool = False
    por_clase: Dict[str, RestriccionClase] = field(default_factory=dict)


@dataclass
class ConfigSectores:
    """Configuración de sectorización."""
    modo: str = "vertical"  # "vertical", "horizontal", "rejilla"
    num_verticales: int = 1
    num_horizontales: int = 1
    mostrar_etiquetas: bool = True
    mostrar_sectorizacion: bool = True
    mostrar_borde_banda: bool = True
    color_lineas: Tuple[int, int, int] = (255, 255, 255)
    color_bordes: Tuple[int, int, int] = (0, 255, 255)
    grosor_lineas: int = 1
    grosor_bordes: int = 2
    opacidad_lineas: float = 1.0
    # Parámetros de perspectiva
    use_perspective: bool = True
    use_border_masks: bool = True
    modo_delimitacion: str = "Auto"  # "Auto", "Máscara", "BBox"
    smooth_alpha: float = 0.15
    max_corner_jump_px: float = 50.0
    inset_px: int = 0
    debug_overlay: bool = False
    comportamiento_fallo: str = "Congelar"  # "Congelar", "Rectángulo", "Desactivar"
    roi_quant_step_px: int = 500  # Tolerancia para cambios de ROI (Estabilización V2)
    line_quant_step_px: int = 500  # Tolerancia para cambios de líneas (Estabilización V2)
    # === NUEVO: Bordes curvos (polilíneas) ===
    curved_edges_enabled: bool = False
    curved_bins_vertical: int = 7
    curved_bins_horizontal: int = 7
    curved_percentile_trim: float = 0.10
    # === NUEVO: Padding de malla ===
    padding_top_px: int = 0
    padding_bottom_px: int = 0
    padding_left_px: int = 0
    padding_right_px: int = 0


@dataclass
class ROIBanda:
    """Región de interés de la banda - soporta modo rectángulo y cuadrilátero."""
    # Modo rectángulo (fallback)
    x_izquierda: int = 0
    x_derecha: int = 0
    y_superior: int = 0
    y_inferior: int = 0
    
    # Modo cuadrilátero (perspectiva): [tl, tr, br, bl] como array (4,2)
    corners: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None      # Homografía imagen -> rectificado
    H_inv: Optional[np.ndarray] = None  # Homografía rectificado -> imagen
    
    is_quad: bool = False
    valido: bool = False
    
    # === NUEVO: Polilíneas curvas para bordes ===
    poly_left: Optional[np.ndarray] = None    # Nx2 polyline ordenada por Y
    poly_right: Optional[np.ndarray] = None   # Nx2 polyline ordenada por Y
    poly_top: Optional[np.ndarray] = None     # Nx2 polyline ordenada por X
    poly_bottom: Optional[np.ndarray] = None  # Nx2 polyline ordenada por X
    polygon_roi: Optional[np.ndarray] = None  # Polígono cerrado Nx2
    is_curved: bool = False
    
    # === NUEVO: Máscaras ROI para filtrado ===
    roi_mask: Optional[np.ndarray] = None         # Máscara binaria HxW
    roi_mask_padded: Optional[np.ndarray] = None  # Máscara con padding aplicado
    
    @property
    def ancho(self) -> int:
        return max(0, self.x_derecha - self.x_izquierda)
    
    @property
    def alto(self) -> int:
        return max(0, self.y_inferior - self.y_superior)


@dataclass
class InfoSector:
    """Información de un sector individual."""
    id: int
    fila: int
    columna: int
    # Modo rectángulo
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    # Modo perspectiva: 4 esquinas del sector en imagen
    corners_img: Optional[np.ndarray] = None
    # Coordenadas normalizadas [0,1] en espacio rectificado
    u1: float = 0.0
    v1: float = 0.0
    u2: float = 1.0
    v2: float = 1.0
    
    def contiene_punto(self, x: float, y: float) -> bool:
        """Verifica si un punto está dentro del sector (modo rectángulo)."""
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2
    
    @property
    def centro(self) -> Tuple[int, int]:
        if self.corners_img is not None and len(self.corners_img) == 4:
            cx = int(np.mean(self.corners_img[:, 0]))
            cy = int(np.mean(self.corners_img[:, 1]))
            return (cx, cy)
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class Sectorizador:
    """
    Clase principal para la sectorización dinámica de la banda de acero.
    
    Soporta dos modos:
    - Rectángulo (fallback): ROI axis-aligned cuando no hay suficiente info de bordes
    - Cuadrilátero (perspectiva): ROI trapezoidal con homografía para seguir la banda real
    """
    
    # Dimensiones del espacio rectificado (arbitrarias, solo para homografía)
    RECT_WIDTH = 1000.0
    RECT_HEIGHT = 1000.0
    MIN_CURVE_EDGE_PTS = 8
    MIN_CURVE_SPAN_PX = 20
    
    def __init__(self):
        self._perftrace_callback: Optional[callable] = None
        self.config_bordes = ConfigBordes()
        self.config_sectores = ConfigSectores()
        self.config_restricciones = ConfigRestricciones()  # NUEVO: restricciones por clase
        self._roi_actual: ROIBanda = ROIBanda()
        self._sectores: List[InfoSector] = []
        
        # Líneas de sector proyectadas: lista de segmentos [(pt1, pt2), ...]
        self._lineas_sector_proyectadas: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
        
        # Suavizado EMA de las 4 esquinas
        self._corners_smooth: Optional[np.ndarray] = None
        self._last_valid_corners: Optional[np.ndarray] = None
        self._smooth_reject_count: int = 0
        
        # Dimensiones de imagen
        self._img_width: int = 0
        self._img_height: int = 0
        
        # Callback para notificar cambios
        self._on_config_change: Optional[callable] = None
        
        # Debug: puntos usados en el último ajuste de línea
        self._debug_edge_pts: Dict[str, np.ndarray] = {}  # 'left', 'right', 'top', 'bottom'
        self._debug_lineas: Dict[str, Tuple[float, float, float, float]] = {}
        
        # NUEVO: Lock para thread safety entre hilo de dibujo y snapshot
        self._lock = threading.Lock()

        # === OPTIMIZACIÓN V4.6: Caching Gráfico ===
        self._cached_overlay: Optional[np.ndarray] = None
        self._cached_overlay_hash: int = 0
        self._needs_redraw: bool = True
        self._overlay_frame_counter: int = 0  # V4.8: contador para refresh periódico

    def set_perftrace_callback(self, cb: Optional[callable]):
        """Establece la función a llamar para reportar eventos de rendimiento."""
        self._perftrace_callback = cb

    def _log_perf(self, event: str, data: dict):
        if self._perftrace_callback:
            self._perftrace_callback(event, data)

    def _invalidar_overlay(self):
        """Marca el overlay como invalido para forzar redibujado."""
        self._needs_redraw = True

    # ===================== CONFIGURACIÓN =====================
    
    def set_config_bordes(
        self,
        clase_superior: str = "",
        clase_inferior: str = "",
        clase_izquierdo: str = "",
        clase_derecho: str = ""
    ) -> None:
        """Configura las clases de detección para cada borde."""
        self.config_bordes.clase_superior = clase_superior
        self.config_bordes.clase_inferior = clase_inferior
        self.config_bordes.clase_izquierdo = clase_izquierdo
        self.config_bordes.clase_derecho = clase_derecho
        self._limpiar_historial()
        self._invalidar_overlay()  # Invalidate cache
        if self._on_config_change:
            self._on_config_change()
    
    def set_config_sectores(
        self,
        # ... (argumentos omitidos para brevedad en reemplazo se mantienen igual, solo mostramos cambios clave)
        modo: str = "vertical",
        num_verticales: int = 1,
        num_horizontales: int = 1,
        mostrar_etiquetas: bool = True,
        mostrar_sectorizacion: bool = True,
        mostrar_borde_banda: bool = True,
        color_lineas: Tuple[int, int, int] = (255, 255, 255),
        color_bordes: Tuple[int, int, int] = (0, 255, 255),
        grosor_lineas: int = 1,
        opacidad_lineas: float = 1.0,
        grosor_bordes: int = 2,
        use_perspective: bool = True,
        use_border_masks: bool = True,
        modo_delimitacion: str = "Auto",
        smooth_alpha: float = 0.15,
        max_corner_jump_px: float = 50.0,
        inset_px: int = 0,
        debug_overlay: bool = False,
        comportamiento_fallo: str = "Congelar",
        # === NUEVO: Bordes curvos ===
        curved_edges_enabled: bool = False,
        curved_bins_vertical: int = 7,
        curved_bins_horizontal: int = 7,
        # === NUEVO: Padding ===
        padding_top_px: int = 0,
        padding_bottom_px: int = 0,
        padding_left_px: int = 0,
        padding_right_px: int = 0,
        roi_quant_step_px: int = 5,
        line_quant_step_px: int = 5
    ) -> None:
        """Configura los parámetros de sectorización."""
        self.config_sectores.modo = modo
        self.config_sectores.num_verticales = max(1, num_verticales)
        self.config_sectores.num_horizontales = max(1, num_horizontales)
        self.config_sectores.mostrar_etiquetas = mostrar_etiquetas
        self.config_sectores.mostrar_sectorizacion = mostrar_sectorizacion
        self.config_sectores.mostrar_borde_banda = mostrar_borde_banda
        self.config_sectores.color_lineas = color_lineas
        self.config_sectores.color_bordes = color_bordes
        self.config_sectores.grosor_lineas = grosor_lineas
        self.config_sectores.opacidad_lineas = opacidad_lineas
        self.config_sectores.grosor_bordes = grosor_bordes
        self.config_sectores.use_perspective = use_perspective
        self.config_sectores.use_border_masks = use_border_masks
        self.config_sectores.modo_delimitacion = modo_delimitacion
        self.config_sectores.smooth_alpha = smooth_alpha
        self.config_sectores.max_corner_jump_px = max_corner_jump_px
        self.config_sectores.inset_px = inset_px
        self.config_sectores.debug_overlay = debug_overlay
        self.config_sectores.comportamiento_fallo = comportamiento_fallo
        # NUEVO: Bordes curvos
        self.config_sectores.curved_edges_enabled = curved_edges_enabled
        self.config_sectores.curved_bins_vertical = curved_bins_vertical
        self.config_sectores.curved_bins_horizontal = curved_bins_horizontal
        # NUEVO: Padding
        self.config_sectores.padding_top_px = padding_top_px
        self.config_sectores.padding_bottom_px = padding_bottom_px
        self.config_sectores.padding_left_px = padding_left_px
        self.config_sectores.padding_right_px = padding_right_px
        self.config_sectores.roi_quant_step_px = max(1, int(roi_quant_step_px))
        self.config_sectores.line_quant_step_px = max(1, int(line_quant_step_px))
        
        self._recalcular_sectores()
        self._invalidar_overlay() # Invalidate cache
        if self._on_config_change:
            self._on_config_change()
            
    # ... (Otros setters invalidan overlay) ...
    def set_num_sectores_verticales(self, n: int) -> None:
        self.config_sectores.num_verticales = max(1, n)
        self._recalcular_sectores()
        self._invalidar_overlay()

    def set_num_sectores_horizontales(self, n: int) -> None:
        self.config_sectores.num_horizontales = max(1, n)
        self._recalcular_sectores()
        self._invalidar_overlay()

    def set_modo(self, modo: str) -> None:
        if modo in ("vertical", "horizontal", "rejilla"):
            self.config_sectores.modo = modo
            self._recalcular_sectores()
            self._invalidar_overlay()
            
    def set_mostrar_sectorizacion(self, mostrar: bool) -> None:
        self.config_sectores.mostrar_sectorizacion = mostrar
        self._invalidar_overlay()
    
    def set_mostrar_etiquetas(self, mostrar: bool) -> None:
        self.config_sectores.mostrar_etiquetas = mostrar
        self._invalidar_overlay()
    
    def _limpiar_historial(self) -> None:
        """Limpia el historial de suavizado."""
        self._corners_smooth = None
        self._last_valid_corners = None
        self._smooth_reject_count = 0
    
    # ===================== SUAVIZADO =====================
    
    def _suavizar_corners(self, corners_nuevos: np.ndarray) -> np.ndarray:
        """
        Aplica suavizado EMA a las 4 esquinas con rechazo de outliers.
        
        Args:
            corners_nuevos: Array (4,2) con las nuevas esquinas [tl, tr, br, bl]
            
        Returns:
            Array (4,2) con las esquinas suavizadas
        """
        alpha = self.config_sectores.smooth_alpha
        max_jump = self.config_sectores.max_corner_jump_px
        
        if self._corners_smooth is None:
            # Primera vez: inicializar
            self._corners_smooth = corners_nuevos.astype(np.float64).copy()
            self._last_valid_corners = corners_nuevos.copy()
            return corners_nuevos
        
        # Calcular distancia de salto para cada corner
        distances = np.linalg.norm(corners_nuevos - self._corners_smooth, axis=1)
        over = distances > max_jump
        if np.all(over):
            self._smooth_reject_count += 1
            if self._smooth_reject_count >= 2:
                self._corners_smooth = corners_nuevos.astype(np.float64).copy()
                self._last_valid_corners = corners_nuevos.copy()
                self._smooth_reject_count = 0
                return corners_nuevos
        else:
            self._smooth_reject_count = 0
        
        # Si algún corner salta demasiado, rechazarlo y usar el anterior
        result = self._corners_smooth.copy()
        for i in range(4):
            if distances[i] <= max_jump:
                # EMA suave
                result[i] = alpha * corners_nuevos[i] + (1 - alpha) * self._corners_smooth[i]
            # else: mantener el valor anterior (rechazo de outlier)
        
        self._corners_smooth = result
        self._last_valid_corners = result.astype(np.int32)
        return self._last_valid_corners

    # ===================== PROCESAMIENTO PRINCIPAL =====================
    
    def procesar_frame(
        self,
        detecciones: List[Dict],
        imagen: np.ndarray,
        dibujar: bool = True
    ) -> np.ndarray:
        t_start = time.perf_counter()
        t_recalc = 0.0
        t_overlay = 0.0
        t_cond = 0.0
        
        t0_cond = time.perf_counter()
        if imagen is None or imagen.size == 0:
            return imagen
        
        h, w = imagen.shape[:2]
        self._img_height = h
        self._img_width = w

        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        self._frame_counter += 1

        # == OPTIMIZACIÓN MULTI-INSTANCIA ==
        # Aumentar stride de 10 -> 30 para reducir carga CPU
        recalc_stride = 30 
        should_recalc = (self._frame_counter % recalc_stride == 1) or self._frame_counter <= 1 or not self._roi_actual.valido

        if should_recalc and self.config_sectores.comportamiento_fallo == "Congelar" and self._roi_actual.valido:
            has_border = False
            if detecciones:
                clases_borde = {
                    self.config_bordes.clase_superior,
                    self.config_bordes.clase_inferior,
                    self.config_bordes.clase_izquierdo,
                    self.config_bordes.clase_derecho,
                }
                clases_borde.discard(None)
                clases_borde.discard("")
                if clases_borde:
                    for det in detecciones:
                        cls_name = det.get("class_name") or det.get("cls") or ""
                        if cls_name in clases_borde:
                            has_border = True
                            break
            if not has_border:
                should_recalc = False
        t_cond = (time.perf_counter() - t0_cond) * 1000.0

        if self._roi_actual.valido:
            if not hasattr(self, '_overlay_frame_counter'):
                self._overlay_frame_counter = 0
            self._overlay_frame_counter += 1
            if self._overlay_frame_counter >= 300:
                self._invalidar_overlay()
                self._overlay_frame_counter = 0

        # Fast path
        if dibujar and self.config_sectores.mostrar_sectorizacion:
            if not should_recalc and self._cached_overlay is not None and not self._needs_redraw:
                if self._cached_overlay.shape[:2] == (h, w):
                    t0_ov = time.perf_counter()
                    np.maximum(imagen, self._cached_overlay, out=imagen)
                    t_overlay = (time.perf_counter() - t0_ov) * 1000.0
                    self._log_perf("sector_perf", {
                        "total_ms": round((time.perf_counter() - t_start)*1000.0, 2),
                        "recalc_ms": 0.0,
                        "overlay_ms": round(t_overlay, 2),
                        "fast_path": True,
                        "should_recalc": should_recalc,
                        "needs_redraw": self._needs_redraw,
                        "cached_overlay": (self._cached_overlay is not None)
                    })
                    return imagen

        roi_changed = False
        lines_changed = False
        roi_max_dx = None
        roi_max_dy = None
        max_delta_lineas = None
        if should_recalc:
            t0_recalc = time.perf_counter()
            prev_lineas_hash = getattr(self, "_lineas_hash", None)
            prev_roi_key = getattr(self, "_prev_roi_key", None)
            prev_roi_raw_bounds = getattr(self, "_prev_roi_raw_bounds", None)
            prev_lineas_coords_raw = getattr(self, "_prev_lineas_coords_raw", None)
            try:
                self._calcular_roi_perspectiva(detecciones, w, h)
                self._recalcular_sectores()
                cfg = self.config_sectores
                roi_quant_step = max(1, int(getattr(cfg, "roi_quant_step_px", 2) or 1))
                line_quant_step = max(1, int(getattr(cfg, "line_quant_step_px", 2) or 1))

                def _quantize_val(value: int, step: int) -> int:
                    if step <= 1:
                        return int(value)
                    return int(value // step * step)

                def _quantize_array(array: Optional[np.ndarray], step: int) -> Optional[np.ndarray]:
                    if array is None:
                        return None
                    if step <= 1:
                        return array
                    return (np.floor(array.astype(np.float32) / step) * step).astype(np.int32)

                lineas_coords_raw: list[int] = []
                if self._lineas_sector_proyectadas:
                    for pt1, pt2 in self._lineas_sector_proyectadas:
                        lineas_coords_raw.extend([
                            int(pt1[0]),
                            int(pt1[1]),
                            int(pt2[0]),
                            int(pt2[1])
                        ])
                if prev_lineas_coords_raw and len(prev_lineas_coords_raw) == len(lineas_coords_raw):
                    max_delta_lineas = max(
                        abs(curr - prev) for curr, prev in zip(lineas_coords_raw, prev_lineas_coords_raw)
                    )
                self._prev_lineas_coords_raw = lineas_coords_raw

                new_lineas_hash = self._calc_lineas_hash(quant_step=line_quant_step)
                if prev_lineas_hash != new_lineas_hash: lines_changed = True
                self._lineas_hash = new_lineas_hash
                roi = self._roi_actual

                roi_raw_bounds = (roi.x_izquierda, roi.x_derecha, roi.y_superior, roi.y_inferior)
                if prev_roi_raw_bounds is not None:
                    roi_max_dx = max(
                        abs(roi_raw_bounds[0] - prev_roi_raw_bounds[0]),
                        abs(roi_raw_bounds[1] - prev_roi_raw_bounds[1])
                    )
                    roi_max_dy = max(
                        abs(roi_raw_bounds[2] - prev_roi_raw_bounds[2]),
                        abs(roi_raw_bounds[3] - prev_roi_raw_bounds[3])
                    )
                self._prev_roi_raw_bounds = roi_raw_bounds

                q_x_izq = _quantize_val(roi.x_izquierda, roi_quant_step)
                q_x_der = _quantize_val(roi.x_derecha, roi_quant_step)
                q_y_sup = _quantize_val(roi.y_superior, roi_quant_step)
                q_y_inf = _quantize_val(roi.y_inferior, roi_quant_step)
                corners_q = _quantize_array(roi.corners, roi_quant_step)
                poly_left_q = _quantize_array(roi.poly_left, roi_quant_step)
                poly_right_q = _quantize_array(roi.poly_right, roi_quant_step)
                poly_top_q = _quantize_array(roi.poly_top, roi_quant_step)
                poly_bottom_q = _quantize_array(roi.poly_bottom, roi_quant_step)
                polygon_q = _quantize_array(roi.polygon_roi, roi_quant_step)
                corners_key = hash(corners_q.tobytes()) if corners_q is not None else None
                poly_left_key = hash(poly_left_q.tobytes()) if poly_left_q is not None else None
                poly_right_key = hash(poly_right_q.tobytes()) if poly_right_q is not None else None
                poly_top_key = hash(poly_top_q.tobytes()) if poly_top_q is not None else None
                poly_bottom_key = hash(poly_bottom_q.tobytes()) if poly_bottom_q is not None else None
                polygon_key = hash(polygon_q.tobytes()) if polygon_q is not None else None
                roi_key = (
                    h,
                    w,
                    roi.valido,
                    roi.is_quad,
                    roi.is_curved,
                    q_x_izq,
                    q_x_der,
                    q_y_sup,
                    q_y_inf,
                    corners_key,
                    poly_left_key,
                    poly_right_key,
                    poly_top_key,
                    poly_bottom_key,
                    polygon_key
                )
                roi_changed = (prev_roi_key != roi_key)
                self._prev_roi_key = roi_key
                if self._roi_actual.valido:
                    cache_key = (h, w, roi.x_izquierda, roi.x_derecha, roi.y_superior, roi.y_inferior)
                    if not hasattr(self, '_mask_cache_key') or self._mask_cache_key != cache_key or roi.roi_mask is None:
                        roi.roi_mask = self._generar_mascara_roi(h, w)
                        roi.roi_mask_padded = self._aplicar_padding_mascara(roi.roi_mask)
                        self._mask_cache_key = cache_key
            except Exception: pass
            t_recalc = (time.perf_counter() - t0_recalc) * 1000.0
        
        if roi_changed or lines_changed:
            self._invalidar_overlay()

        needs_redraw_before_overlay = self._needs_redraw
        cached_overlay_before = (self._cached_overlay is not None)
        if dibujar and self.config_sectores.mostrar_sectorizacion:
            t0_ov = time.perf_counter()
            self._aplicar_overlay_cached(imagen)
            t_overlay = (time.perf_counter() - t0_ov) * 1000.0
            if self.config_sectores.debug_overlay:
                self._dibujar_debug_overlay(imagen)
        
        t_total = (time.perf_counter() - t_start) * 1000.0
        self._log_perf("sector_perf", {
            "total_ms": round(t_total, 2),
            "cond_ms": round(t_cond, 2),
            "recalc_ms": round(t_recalc, 2),
            "overlay_ms": round(t_overlay, 2),
            "fast_path": False,
            "should_recalc": should_recalc,
            "roi_changed": roi_changed,
            "lines_changed": lines_changed,
            "roi_max_dx": roi_max_dx,
            "roi_max_dy": roi_max_dy,
            "lineas_max_delta": max_delta_lineas,
            "needs_redraw_before_overlay": needs_redraw_before_overlay,
            "cached_overlay_before": cached_overlay_before
        })
        return imagen

    def _aplicar_overlay_cached(self, imagen: np.ndarray) -> None:
        """Genera o usa el overlay cacheado y lo aplica a la imagen."""
        h, w = imagen.shape[:2]
        
        # Validar caché existente (tamaño)
        if self._cached_overlay is not None:
             if self._cached_overlay.shape[:2] != (h, w):
                 self._needs_redraw = True

        # Regenerar si es necesario
        if self._needs_redraw or self._cached_overlay is None:
            try:
                 if LOGGER.isEnabledFor(logging.DEBUG):
                     LOGGER.debug("REGENERANDO CACHE OVERLAY")
                 # Crear canvas transparente (negro)
                 overlay = np.zeros_like(imagen)
                 
                 # Dibujar componentes estáticos
                 self._dibujar_bordes(overlay)
                 self._dibujar_sectores(overlay)
                 if self.config_sectores.mostrar_etiquetas:
                     self._dibujar_etiquetas(overlay)
                 
                 self._cached_overlay = overlay
                 self._needs_redraw = False
            except Exception as e:
                 print(f"Error generando overlay cache: {e}")
                 return

        # Aplicar cache a imagen (V4.7: Blending ultra-rápido)
        if self._cached_overlay is not None:
            try:
                # V4.7: np.maximum es ~10-50x más rápido que mask copy
                # Funciona porque el overlay tiene fondo negro (0,0,0)
                # y las líneas son colores claros, así que max() las "pinta"
                np.maximum(imagen, self._cached_overlay, out=imagen)
            except Exception:
                pass
    
    # ===================== CÁLCULO DE ROI CON PERSPECTIVA =====================
    
    def _calcular_roi_perspectiva(
        self,
        detecciones: List[Dict],
        img_width: int,
        img_height: int
    ) -> None:
        t0 = time.perf_counter()
        t_group = 0.0
        t_lines = 0.0
        t_homography = 0.0
        t_poly = 0.0

        use_perspective = self.config_sectores.use_perspective
        use_masks = self.config_sectores.use_border_masks
        
        t0_g = time.perf_counter()
        dets_por_clase: Dict[str, List[Dict]] = {}
        for det in detecciones:
            cls_name = det.get("class_name", "")
            if cls_name:
                if cls_name not in dets_por_clase: dets_por_clase[cls_name] = []
                dets_por_clase[cls_name].append(det)
        t_group = (time.perf_counter() - t0_g) * 1000.0

        self._debug_edge_pts.clear()
        linea_izq = linea_der = linea_sup = linea_inf = None
        
        t0_l = time.perf_counter()
        if use_perspective and use_masks:
            linea_izq = self._obtener_linea_borde_lateral(dets_por_clase, self.config_bordes.clase_izquierdo, es_izquierdo=True)
            linea_der = self._obtener_linea_borde_lateral(dets_por_clase, self.config_bordes.clase_derecho, es_izquierdo=False)
            linea_sup = self._obtener_linea_borde_horizontal(dets_por_clase, self.config_bordes.clase_superior, es_superior=True)
            linea_inf = self._obtener_linea_borde_horizontal(dets_por_clase, self.config_bordes.clase_inferior, es_superior=False)
        t_lines = (time.perf_counter() - t0_l) * 1000.0
        
        if use_perspective and linea_izq is not None and linea_der is not None:
            t0_h = time.perf_counter()
            pts_left = self._debug_edge_pts.get("left")
            pts_right = self._debug_edge_pts.get("right")
            # ... (Simplified logic for the sake of script replacement)
            # Actually, let's keep it mostly same but wrap the homography call
            inferred_sup = self._inferir_linea_horizontal_desde_laterales(linea_izq, linea_der, pts_left, pts_right, es_superior=True)
            inferred_inf = self._inferir_linea_horizontal_desde_laterales(linea_izq, linea_der, pts_left, pts_right, es_superior=False)
            if linea_sup is None: linea_sup = inferred_sup
            if linea_inf is None: linea_inf = inferred_inf
            
            corners = self._construir_corners_desde_lineas(linea_izq, linea_der, linea_sup, linea_inf, img_width, img_height)
            if corners is not None and self._validar_corners(corners, img_width, img_height):
                corners_suaves = self._suavizar_corners(corners)
                corners_inset = self._aplicar_inset(corners_suaves)
                H, H_inv = self._calcular_homografia(corners_inset)
                if H is not None:
                    self._roi_actual = ROIBanda(corners=corners_inset, H=H, H_inv=H_inv, is_quad=True, valido=True)
                    t_homography = (time.perf_counter() - t0_h) * 1000.0
                    
                    if self.config_sectores.curved_edges_enabled:
                        t0_p = time.perf_counter()
                        try: self._calcular_polilineas_curvas(dets_por_clase)
                        except Exception: pass
                        t_poly = (time.perf_counter() - t0_p) * 1000.0
                    
                    self._log_perf("roi_perf", {
                        "group_ms": round(t_group, 2),
                        "lines_ms": round(t_lines, 2),
                        "homography_ms": round(t_homography, 2),
                        "poly_ms": round(t_poly, 2),
                        "total_ms": round((time.perf_counter() - t0)*1000.0, 2)
                    })
                    return
        
        self._calcular_roi_rectangulo(dets_por_clase, img_width, img_height)

    def _obtener_linea_borde_lateral(
        self,
        dets_por_clase: Dict[str, List[Dict]],
        clase: str,
        es_izquierdo: bool
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Obtiene una línea (x1,y1,x2,y2) para un borde lateral usando contorno o bbox.
        Devuelve puntos superior e inferior del borde.
        """
        if not clase or clase not in dets_por_clase:
            return None
        
        dets = dets_por_clase[clase]
        if not dets:
            return None
        
        # Si la misma clase se usa para ambos bordes, seleccionar por posición
        if clase == self.config_bordes.clase_izquierdo == self.config_bordes.clase_derecho:
            centers = [(d, (d["bbox"][0] + d["bbox"][2]) / 2) for d in dets if len(d.get("bbox", [])) >= 4]
            if len(centers) < 2:
                return None
            centers.sort(key=lambda x: x[1])
            det = centers[0][0] if es_izquierdo else centers[-1][0]
        else:
            if es_izquierdo:
                det = min(dets, key=lambda d: d.get("bbox", [self._img_width])[0])
            else:
                det = max(dets, key=lambda d: d.get("bbox", [0,0,0,0])[2] if len(d.get("bbox",[])) >= 4 else 0)
        
        bbox = det.get("bbox", [])
        linea_bbox = None
        if len(bbox) >= 4:
            x = bbox[2] if es_izquierdo else bbox[0]  # Borde interior hacia la banda
            linea_bbox = (x, bbox[1], x, bbox[3])
        
        debug_key = 'left' if es_izquierdo else 'right'
        
        # Prioridad 1: Usar edge_pts precalculados (más precisos)
        edge_pts = det.get("edge_pts")
        if edge_pts is not None:
            key = 'left' if es_izquierdo else 'right'
            pts = edge_pts.get(key)
            if pts is not None and len(pts) >= 10:
                linea = self._ajustar_linea_desde_edge_pts(pts)
                if linea is not None and self._validar_linea_lateral(linea, bbox, es_izquierdo):
                    self._debug_edge_pts[debug_key] = pts
                    self._debug_lineas[debug_key] = linea
                    return linea
        
        # Prioridad 2: Usar contorno si existe
        contour = det.get("contour")
        if contour is not None and len(contour) >= 4:
            linea = self._ajustar_linea_vertical(contour, es_izquierdo)
            if linea is not None and self._validar_linea_lateral(linea, bbox, es_izquierdo):
                self._debug_lineas[debug_key] = linea
                return linea
        
        # Fallback: usar bbox
        if debug_key in self._debug_edge_pts:
            del self._debug_edge_pts[debug_key]
        return linea_bbox
    
    def _validar_linea_lateral(
        self,
        linea: Tuple[float, float, float, float],
        bbox: list,
        es_izquierdo: bool
    ) -> bool:
        """Valida que la línea lateral sea razonable."""
        if linea is None or len(bbox) < 4:
            return False
        
        x1, y1, x2, y2 = linea
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[:4]
        
        # Calcular pendiente (dx/dy para líneas verticales)
        dy = y2 - y1
        if abs(dy) < 1:
            return False
        pendiente = (x2 - x1) / dy
        
        # Rechazar si pendiente es demasiado pronunciada (línea casi horizontal)
        if abs(pendiente) > 0.5:  # más de 26 grados de inclinación
            return False
        
        # Verificar que la línea esté cerca del bbox (no se ha ido muy lejos)
        x_medio = (x1 + x2) / 2
        bbox_w = max(1.0, bbox_x2 - bbox_x1)
        img_w = max(1.0, float(self._img_width))
        trim = self.config_sectores.curved_percentile_trim
        try:
            trim = float(trim)
        except Exception:
            trim = 0.10
        trim = max(0.0, min(0.45, trim))
        margin_out = max(trim * bbox_w, 0.01 * img_w)
        margin_in = max((trim * 2.0) * bbox_w, 0.02 * img_w)
        if es_izquierdo:
            if x_medio < bbox_x1 - margin_out or x_medio > bbox_x2 + margin_in:
                return False
        else:
            if x_medio < bbox_x1 - margin_in or x_medio > bbox_x2 + margin_out:
                return False

        
        return True
    
    def _obtener_linea_borde_horizontal(
        self,
        dets_por_clase: Dict[str, List[Dict]],
        clase: str,
        es_superior: bool
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Obtiene una línea (x1,y1,x2,y2) para un borde horizontal.
        """
        if not clase or clase not in dets_por_clase:
            return None
        
        dets = dets_por_clase[clase]
        if not dets:
            return None
        
        # Si la misma clase para ambos bordes
        if clase == self.config_bordes.clase_superior == self.config_bordes.clase_inferior:
            centers = [(d, (d["bbox"][1] + d["bbox"][3]) / 2) for d in dets if len(d.get("bbox", [])) >= 4]
            if len(centers) < 2:
                return None
            centers.sort(key=lambda x: x[1])
            det = centers[0][0] if es_superior else centers[-1][0]
        else:
            if es_superior:
                det = min(dets, key=lambda d: d.get("bbox", [0, self._img_height])[1])
            else:
                det = max(dets, key=lambda d: d.get("bbox", [0,0,0,0])[3] if len(d.get("bbox",[])) >= 4 else 0)
        
        bbox = det.get("bbox", [])
        linea_bbox = None
        if len(bbox) >= 4:
            y = bbox[3] if es_superior else bbox[1]  # Borde interior hacia la banda
            linea_bbox = (bbox[0], y, bbox[2], y)
        
        debug_key = 'top' if es_superior else 'bottom'
        
        # Prioridad 1: Usar edge_pts precalculados (más precisos)
        edge_pts = det.get("edge_pts")
        if edge_pts is not None:
            key = 'top' if es_superior else 'bottom'
            pts = edge_pts.get(key)
            if pts is not None and len(pts) >= 10:
                linea = self._ajustar_linea_desde_edge_pts(pts)
                if linea is not None and self._validar_linea_horizontal(linea, bbox, es_superior):
                    self._debug_edge_pts[debug_key] = pts
                    self._debug_lineas[debug_key] = linea
                    return linea
        
        # Prioridad 2: Usar contorno
        contour = det.get("contour")
        if contour is not None and len(contour) >= 4:
            linea = self._ajustar_linea_horizontal(contour, es_superior)
            if linea is not None and self._validar_linea_horizontal(linea, bbox, es_superior):
                self._debug_lineas[debug_key] = linea
                return linea
        
        # Fallback: bbox
        if debug_key in self._debug_edge_pts:
            del self._debug_edge_pts[debug_key]
        return linea_bbox
    
    def _validar_linea_horizontal(
        self,
        linea: Tuple[float, float, float, float],
        bbox: list,
        es_superior: bool
    ) -> bool:
        """Valida que la línea horizontal sea razonable."""
        if linea is None or len(bbox) < 4:
            return False
        
        x1, y1, x2, y2 = linea
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[:4]
        
        # Calcular pendiente (dy/dx para líneas horizontales)
        dx = x2 - x1
        if abs(dx) < 1:
            return False
        pendiente = (y2 - y1) / dx
        
        # Rechazar si pendiente es demasiado pronunciada (línea casi vertical)
        if abs(pendiente) > 0.5:  # más de 26 grados de inclinación
            return False
        
        # Verificar que la línea esté cerca del bbox
        y_medio = (y1 + y2) / 2
        bbox_h = max(1.0, bbox_y2 - bbox_y1)
        img_h = max(1.0, float(self._img_height))
        trim = self.config_sectores.curved_percentile_trim
        try:
            trim = float(trim)
        except Exception:
            trim = 0.10
        trim = max(0.0, min(0.45, trim))
        margin_out = max(trim * bbox_h, 0.01 * img_h)
        margin_in = max((trim * 2.0) * bbox_h, 0.02 * img_h)
        if es_superior:
            if y_medio < bbox_y1 - margin_out or y_medio > bbox_y2 + margin_in:
                return False
        else:
            if y_medio < bbox_y1 - margin_in or y_medio > bbox_y2 + margin_out:
                return False

        
        return True
    
    
    def _interpolar_x_en_linea(
        self,
        linea: Tuple[float, float, float, float],
        y: float
    ) -> Optional[float]:
        """Interpola x en una linea dada una coordenada y."""
        try:
            x1, y1, x2, y2 = linea
            dy = y2 - y1
            if abs(dy) < 1e-6:
                return float(x1)
            t = (y - y1) / dy
            t = max(0.0, min(1.0, t))
            return float(x1 + t * (x2 - x1))
        except Exception:
            return None

    def _interpolar_x_en_polyline(self, poly: Optional[np.ndarray], y: float) -> Optional[float]:
        """Interpola x en una polilinea (x=f(y)) dada una coordenada y."""
        if poly is None or len(poly) < 2:
            return None
        try:
            pts = poly
            if pts[0][1] > pts[-1][1]:
                pts = pts[::-1]
            if y <= float(pts[0][1]):
                return float(pts[0][0])
            if y >= float(pts[-1][1]):
                return float(pts[-1][0])
            for i in range(len(pts) - 1):
                y1 = float(pts[i][1])
                y2 = float(pts[i + 1][1])
                if (y1 <= y <= y2) or (y2 <= y <= y1):
                    dy = y2 - y1
                    if abs(dy) < 1e-6:
                        return float(pts[i][0])
                    t = (y - y1) / dy
                    return float(pts[i][0] + t * (pts[i + 1][0] - pts[i][0]))
        except Exception:
            return None
        return None

    def _interpolar_y_en_polyline(self, poly: Optional[np.ndarray], x: float) -> Optional[float]:
        """Interpola y en una polilinea (y=f(x)) dada una coordenada x."""
        if poly is None or len(poly) < 2:
            return None
        try:
            pts = poly
            if pts[0][0] > pts[-1][0]:
                pts = pts[::-1]
            if x <= float(pts[0][0]):
                return float(pts[0][1])
            if x >= float(pts[-1][0]):
                return float(pts[-1][1])
            for i in range(len(pts) - 1):
                x1 = float(pts[i][0])
                x2 = float(pts[i + 1][0])
                if (x1 <= x <= x2) or (x2 <= x <= x1):
                    dx = x2 - x1
                    if abs(dx) < 1e-6:
                        return float(pts[i][1])
                    t = (x - x1) / dx
                    return float(pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
        except Exception:
            return None
        return None

    def _recortar_segmento_por_mascara(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        mask: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Recorta un segmento usando una mascara binaria."""
        try:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            h, w = mask.shape[:2]
            steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
            if steps <= 1:
                return None
            xs = np.linspace(x1, x2, steps, dtype=np.float32)
            ys = np.linspace(y1, y2, steps, dtype=np.float32)
            xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
            yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
            inside = mask[yi, xi] > 0
            if not np.any(inside):
                return None
            idx = np.where(inside)[0]
            i0 = int(idx[0])
            i1 = int(idx[-1])
            if i1 <= i0:
                return None
            p0 = (int(xs[i0]), int(ys[i0]))
            p1 = (int(xs[i1]), int(ys[i1]))
            return p0, p1
        except Exception:
            return None

    def _inferir_linea_horizontal_desde_laterales(
        self,
        linea_izq: Tuple[float, float, float, float],
        linea_der: Tuple[float, float, float, float],
        pts_izq: np.ndarray | None,
        pts_der: np.ndarray | None,
        es_superior: bool
    ) -> Optional[Tuple[float, float, float, float]]:
        """Infere una linea horizontal desde laterales, aplicando padding si se define."""
        if pts_izq is None or pts_der is None:
            return None
        if len(pts_izq) < 2 or len(pts_der) < 2:
            return None

        y_left = pts_izq[:, 1]
        y_right = pts_der[:, 1]
        if y_left.size < 2 or y_right.size < 2:
            return None

        pad_px = self.config_sectores.padding_top_px if es_superior else self.config_sectores.padding_bottom_px
        try:
            pad_px = float(pad_px)
        except Exception:
            pad_px = 0.0
        pad_px = max(0.0, pad_px)

        y_min_raw = max(float(np.min(y_left)), float(np.min(y_right)))
        y_max_raw = min(float(np.max(y_left)), float(np.max(y_right)))
        if y_max_raw <= y_min_raw:
            return None

        if pad_px > 0.0:
            trim = self.config_sectores.curved_percentile_trim
            try:
                trim = float(trim)
            except Exception:
                trim = 0.10
            trim = max(0.0, min(0.45, trim))

            low_p = trim * 100.0
            high_p = (1.0 - trim) * 100.0
            y_left_low = float(np.percentile(y_left, low_p))
            y_left_high = float(np.percentile(y_left, high_p))
            y_right_low = float(np.percentile(y_right, low_p))
            y_right_high = float(np.percentile(y_right, high_p))

            y_base_min = max(y_min_raw, y_left_low, y_right_low)
            y_base_max = min(y_max_raw, y_left_high, y_right_high)
            if y_base_max <= y_base_min:
                return None
        else:
            y_base_min = y_min_raw
            y_base_max = y_max_raw

        if es_superior:
            y_target = y_base_min + pad_px
        else:
            y_target = y_base_max - pad_px

        y_target = min(max(y_target, y_min_raw), y_max_raw)

        x_left = self._interpolar_x_en_linea(linea_izq, y_target)
        x_right = self._interpolar_x_en_linea(linea_der, y_target)
        if x_left is None or x_right is None:
            return None

        if x_left > x_right:
            x_left, x_right = x_right, x_left

        return (float(x_left), float(y_target), float(x_right), float(y_target))

    def _ajustar_linea_desde_edge_pts(
        self,
        pts: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Ajusta una línea a los edge points precalculados.
        Los pts ya son los puntos del borde interior, no necesitan filtrarse.
        """
        try:
            if len(pts) < 10:
                return None
            
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            
            # Filtrar outliers (percentil 5-95 en la dimensión principal)
            y_range = y_vals.max() - y_vals.min()
            x_range = x_vals.max() - x_vals.min()
            
            if y_range > x_range:
                # Línea más vertical: ajustar x = a*y + b
                y_p5 = y_vals.min() + y_range * 0.05
                y_p95 = y_vals.min() + y_range * 0.95
                mask = (y_vals >= y_p5) & (y_vals <= y_p95)
                x_filt = x_vals[mask]
                y_filt = y_vals[mask]
                
                if len(y_filt) < 10:
                    return None
                
                A = np.vstack([y_filt, np.ones(len(y_filt))]).T
                result = np.linalg.lstsq(A, x_filt, rcond=None)
                a, b = result[0]
                
                y_min, y_max = y_filt.min(), y_filt.max()
                x1 = a * y_min + b
                x2 = a * y_max + b
                return (float(x1), float(y_min), float(x2), float(y_max))
            else:
                # Línea más horizontal: ajustar y = a*x + b
                x_p5 = x_vals.min() + x_range * 0.05
                x_p95 = x_vals.min() + x_range * 0.95
                mask = (x_vals >= x_p5) & (x_vals <= x_p95)
                x_filt = x_vals[mask]
                y_filt = y_vals[mask]
                
                if len(x_filt) < 10:
                    return None
                
                A = np.vstack([x_filt, np.ones(len(x_filt))]).T
                result = np.linalg.lstsq(A, y_filt, rcond=None)
                a, b = result[0]
                
                x_min, x_max = x_filt.min(), x_filt.max()
                y1 = a * x_min + b
                y2 = a * x_max + b
                return (float(x_min), float(y1), float(x_max), float(y2))
        except Exception:
            return None
    
    # ===================== POLILÍNEAS CURVAS (NUEVO) =====================
    
    def _ajustar_polilinea_vertical(
        self,
        pts: np.ndarray,
        num_bins: int = 7,
        percentile_trim: float = 0.10
    ) -> Optional[np.ndarray]:
        """
        Ajusta una polilínea x=f(y) a edge points usando medianas por bins.
        Retorna array Nx2 de (x, y) ordenado por Y (de arriba a abajo).
        """
        try:
            if num_bins <= 1:
                linea = self._ajustar_linea_desde_edge_pts(pts)
                if linea is None:
                    return None
                x1, y1, x2, y2 = linea
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                return np.array([[x1, y1], [x2, y2]], dtype=np.float32)

            if len(pts) < self.MIN_CURVE_EDGE_PTS:
                return None

            y_vals = pts[:, 1]
            x_vals = pts[:, 0]
            y_min, y_max = y_vals.min(), y_vals.max()

            if y_max - y_min < self.MIN_CURVE_SPAN_PX:  # Rango muy pequeno
                return None

            bin_edges = np.linspace(y_min, y_max, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            x_bins = np.full(num_bins, np.nan, dtype=np.float32)

            for i in range(num_bins):
                bin_mask = (y_vals >= bin_edges[i]) & (y_vals < bin_edges[i+1])
                if i == num_bins - 1:  # Incluir endpoint
                    bin_mask |= (y_vals == bin_edges[i+1])

                x_in_bin = x_vals[bin_mask]
                if len(x_in_bin) < 1:
                    continue

                # Recortar outliers por percentil
                if percentile_trim > 0 and len(x_in_bin) >= 4:
                    low = np.percentile(x_in_bin, percentile_trim * 100)
                    high = np.percentile(x_in_bin, (1 - percentile_trim) * 100)
                    x_trimmed = x_in_bin[(x_in_bin >= low) & (x_in_bin <= high)]
                    if len(x_trimmed) > 0:
                        x_in_bin = x_trimmed

                x_bins[i] = np.median(x_in_bin)

            valid = ~np.isnan(x_bins)
            if np.count_nonzero(valid) < 2:
                return None

            if np.count_nonzero(valid) < num_bins:
                x_bins = np.interp(bin_centers, bin_centers[valid], x_bins[valid])

            poly = np.column_stack((x_bins, bin_centers)).astype(np.float32)
            return poly
        except Exception:
            return None

    
    def _ajustar_polilinea_horizontal(
        self,
        pts: np.ndarray,
        num_bins: int = 7,
        percentile_trim: float = 0.10
    ) -> Optional[np.ndarray]:
        """
        Ajusta una polilínea y=f(x) a edge points usando medianas por bins.
        Retorna array Nx2 de (x, y) ordenado por X (de izquierda a derecha).
        """
        try:
            if num_bins <= 1:
                linea = self._ajustar_linea_desde_edge_pts(pts)
                if linea is None:
                    return None
                x1, y1, x2, y2 = linea
                if x1 > x2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                return np.array([[x1, y1], [x2, y2]], dtype=np.float32)

            if len(pts) < self.MIN_CURVE_EDGE_PTS:
                return None

            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            x_min, x_max = x_vals.min(), x_vals.max()

            if x_max - x_min < self.MIN_CURVE_SPAN_PX:  # Rango muy pequeno
                return None

            bin_edges = np.linspace(x_min, x_max, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            y_bins = np.full(num_bins, np.nan, dtype=np.float32)

            for i in range(num_bins):
                bin_mask = (x_vals >= bin_edges[i]) & (x_vals < bin_edges[i+1])
                if i == num_bins - 1:
                    bin_mask |= (x_vals == bin_edges[i+1])

                y_in_bin = y_vals[bin_mask]
                if len(y_in_bin) < 1:
                    continue

                if percentile_trim > 0 and len(y_in_bin) >= 4:
                    low = np.percentile(y_in_bin, percentile_trim * 100)
                    high = np.percentile(y_in_bin, (1 - percentile_trim) * 100)
                    y_trimmed = y_in_bin[(y_in_bin >= low) & (y_in_bin <= high)]
                    if len(y_trimmed) > 0:
                        y_in_bin = y_trimmed

                y_bins[i] = np.median(y_in_bin)

            valid = ~np.isnan(y_bins)
            if np.count_nonzero(valid) < 2:
                return None

            if np.count_nonzero(valid) < num_bins:
                y_bins = np.interp(bin_centers, bin_centers[valid], y_bins[valid])

            poly = np.column_stack((bin_centers, y_bins)).astype(np.float32)
            return poly
        except Exception:
            return None

    
    def _construir_poligono_roi_curvo(self) -> Optional[np.ndarray]:
        """
        Construye polígono cerrado a partir de polilíneas curvas (o fallback a corners).
        Retorna array Nx2 recorriendo: top (L->R), right (T->B), bottom (R->L), left (B->T).
        
        Mejoras:
        - Ancla las polilíneas a las esquinas del cuadrilátero
        - Limpia puntos duplicados y fuera de rango
        - Valida que el polígono sea válido (área mínima)
        """
        try:
            roi = self._roi_actual
            if not roi.valido or roi.corners is None:
                return None
            
            # Esquinas del cuadrilátero base: tl, tr, br, bl
            tl = roi.corners[0].tolist()
            tr = roi.corners[1].tolist()
            br = roi.corners[2].tolist()
            bl = roi.corners[3].tolist()

            left_line = (tl[0], tl[1], bl[0], bl[1])
            right_line = (tr[0], tr[1], br[0], br[1])
            # Ajustar anclas a extremos reales de polilineas para evitar picos en esquinas
            tl_anchor = tl
            tr_anchor = tr
            br_anchor = br
            bl_anchor = bl
            if roi.poly_top is not None and len(roi.poly_top) >= 2:
                top_left = roi.poly_top[0].tolist()
                top_right = roi.poly_top[-1].tolist()
                x_left_line = self._interpolar_x_en_linea(left_line, top_left[1])
                x_left_poly = self._interpolar_x_en_polyline(roi.poly_left, top_left[1])
                if x_left_poly is not None:
                    x_left_line = max(x_left_line, x_left_poly) if x_left_line is not None else x_left_poly
                x_right_line = self._interpolar_x_en_linea(right_line, top_right[1])
                x_right_poly = self._interpolar_x_en_polyline(roi.poly_right, top_right[1])
                if x_right_poly is not None:
                    x_right_line = min(x_right_line, x_right_poly) if x_right_line is not None else x_right_poly
                tl_anchor = [x_left_line, top_left[1]] if x_left_line is not None else top_left
                tr_anchor = [x_right_line, top_right[1]] if x_right_line is not None else top_right
            if roi.poly_bottom is not None and len(roi.poly_bottom) >= 2:
                bottom_left = roi.poly_bottom[0].tolist()
                bottom_right = roi.poly_bottom[-1].tolist()
                x_left_line = self._interpolar_x_en_linea(left_line, bottom_left[1])
                x_left_poly = self._interpolar_x_en_polyline(roi.poly_left, bottom_left[1])
                if x_left_poly is not None:
                    x_left_line = max(x_left_line, x_left_poly) if x_left_line is not None else x_left_poly
                x_right_line = self._interpolar_x_en_linea(right_line, bottom_right[1])
                x_right_poly = self._interpolar_x_en_polyline(roi.poly_right, bottom_right[1])
                if x_right_poly is not None:
                    x_right_line = min(x_right_line, x_right_poly) if x_right_line is not None else x_right_poly
                bl_anchor = [x_left_line, bottom_left[1]] if x_left_line is not None else bottom_left
                br_anchor = [x_right_line, bottom_right[1]] if x_right_line is not None else bottom_right
            trim = self.config_sectores.curved_percentile_trim
            try:
                trim = float(trim)
            except Exception:
                trim = 0.10
            trim = max(0.0, min(0.45, trim))
            width_est = max(1.0, (abs(tr[0] - tl[0]) + abs(br[0] - bl[0])) / 2.0)
            tol = max(0.0, trim * 0.25 * width_est)

            pad_top = self.config_sectores.padding_top_px
            pad_bottom = self.config_sectores.padding_bottom_px
            try:
                pad_top = float(pad_top)
            except Exception:
                pad_top = 0.0
            try:
                pad_bottom = float(pad_bottom)
            except Exception:
                pad_bottom = 0.0
            apply_top = pad_top > 0.0
            apply_bottom = pad_bottom > 0.0

            left_y_top_raw = min(tl[1], bl[1])
            left_y_bottom_raw = max(tl[1], bl[1])
            right_y_top_raw = min(tr[1], br[1])
            right_y_bottom_raw = max(tr[1], br[1])
            overlap_y_min = max(left_y_top_raw, right_y_top_raw)
            overlap_y_max = min(left_y_bottom_raw, right_y_bottom_raw)
            if overlap_y_max <= overlap_y_min:
                overlap_y_min = min(left_y_top_raw, right_y_top_raw)
                overlap_y_max = max(left_y_bottom_raw, right_y_bottom_raw)

            tl_anchor = [tl_anchor[0], min(max(tl_anchor[1], overlap_y_min), overlap_y_max)]
            tr_anchor = [tr_anchor[0], min(max(tr_anchor[1], overlap_y_min), overlap_y_max)]
            bl_anchor = [bl_anchor[0], min(max(bl_anchor[1], overlap_y_min), overlap_y_max)]
            br_anchor = [br_anchor[0], min(max(br_anchor[1], overlap_y_min), overlap_y_max)]

            left_y_top = overlap_y_min
            left_y_bottom = overlap_y_max
            right_y_top = overlap_y_min
            right_y_bottom = overlap_y_max
            top_y_min = overlap_y_min
            top_y_max = overlap_y_max
            bottom_y_min = overlap_y_min
            bottom_y_max = overlap_y_max

            min_corner_dist = 6
            
            polygon_pts = []
            
            # --- Borde superior (izquierda a derecha): tl -> ... -> tr ---
            polygon_pts.append(tl_anchor)  # Anclar a esquina tl
            
            if roi.poly_top is not None and len(roi.poly_top) >= 2:
                # Excluir primer/último punto si están muy cerca de las esquinas
                pts = roi.poly_top.tolist()
                for pt in pts:
                    x_left = self._interpolar_x_en_linea(left_line, pt[1])
                    x_left_poly = self._interpolar_x_en_polyline(roi.poly_left, pt[1])
                    if x_left_poly is not None:
                        x_left = max(x_left, x_left_poly) if x_left is not None else x_left_poly
                    x_right = self._interpolar_x_en_linea(right_line, pt[1])
                    x_right_poly = self._interpolar_x_en_polyline(roi.poly_right, pt[1])
                    if x_right_poly is not None:
                        x_right = min(x_right, x_right_poly) if x_right is not None else x_right_poly
                    if x_left is not None and x_right is not None and x_left > x_right:
                        x_left, x_right = x_right, x_left
                    if pt[1] < top_y_min:
                        pt = [pt[0], top_y_min]
                    elif pt[1] > top_y_max:
                        pt = [pt[0], top_y_max]
                    if x_left is not None and pt[0] < x_left - tol:
                        pt = [x_left, pt[1]]
                    if x_right is not None and pt[0] > x_right + tol:
                        pt = [x_right, pt[1]]
                    if self._dist(pt, tl_anchor) > min_corner_dist and self._dist(pt, tr_anchor) > min_corner_dist:
                        polygon_pts.append(pt)
            polygon_pts.append(tr_anchor)  # Anclar a esquina tr
            
            # --- Borde derecho (arriba a abajo): tr -> ... -> br ---
            if roi.poly_right is not None and len(roi.poly_right) >= 2:
                pts = roi.poly_right.tolist()
                for pt in pts:
                    if apply_top and pt[1] < right_y_top:
                        continue
                    if apply_bottom and pt[1] > right_y_bottom:
                        continue
                    y_top_limit = top_y_min
                    y_top_poly = self._interpolar_y_en_polyline(roi.poly_top, pt[0])
                    if y_top_poly is not None:
                        y_top_limit = max(y_top_limit, y_top_poly)
                    y_bottom_limit = bottom_y_max
                    y_bottom_poly = self._interpolar_y_en_polyline(roi.poly_bottom, pt[0])
                    if y_bottom_poly is not None:
                        y_bottom_limit = min(y_bottom_limit, y_bottom_poly)
                    if pt[1] < y_top_limit:
                        pt = [pt[0], y_top_limit]
                    if pt[1] > y_bottom_limit:
                        pt = [pt[0], y_bottom_limit]
                    x_line = self._interpolar_x_en_linea(right_line, pt[1])
                    if x_line is not None and pt[0] > x_line + tol:
                        pt = [x_line, pt[1]]
                    if self._dist(pt, tr_anchor) > min_corner_dist and self._dist(pt, br_anchor) > min_corner_dist:
                        polygon_pts.append(pt)
            polygon_pts.append(br_anchor)  # Anclar a esquina br
            
            # --- Borde inferior (derecha a izquierda): br -> ... -> bl ---
            if roi.poly_bottom is not None and len(roi.poly_bottom) >= 2:
                pts = roi.poly_bottom[::-1].tolist()  # Invertir orden
                for pt in pts:
                    x_left = self._interpolar_x_en_linea(left_line, pt[1])
                    x_left_poly = self._interpolar_x_en_polyline(roi.poly_left, pt[1])
                    if x_left_poly is not None:
                        x_left = max(x_left, x_left_poly) if x_left is not None else x_left_poly
                    x_right = self._interpolar_x_en_linea(right_line, pt[1])
                    x_right_poly = self._interpolar_x_en_polyline(roi.poly_right, pt[1])
                    if x_right_poly is not None:
                        x_right = min(x_right, x_right_poly) if x_right is not None else x_right_poly
                    if x_left is not None and x_right is not None and x_left > x_right:
                        x_left, x_right = x_right, x_left
                    if pt[1] < bottom_y_min:
                        pt = [pt[0], bottom_y_min]
                    elif pt[1] > bottom_y_max:
                        pt = [pt[0], bottom_y_max]
                    if x_left is not None and pt[0] < x_left - tol:
                        pt = [x_left, pt[1]]
                    if x_right is not None and pt[0] > x_right + tol:
                        pt = [x_right, pt[1]]
                    if self._dist(pt, br_anchor) > min_corner_dist and self._dist(pt, bl_anchor) > min_corner_dist:
                        polygon_pts.append(pt)
            polygon_pts.append(bl_anchor)  # Anclar a esquina bl
            
            # --- Borde izquierdo (abajo a arriba): bl -> ... -> (no añadir tl, cierra solo) ---
            if roi.poly_left is not None and len(roi.poly_left) >= 2:
                pts = roi.poly_left[::-1].tolist()  # Invertir orden
                for pt in pts:
                    if apply_top and pt[1] < left_y_top:
                        continue
                    if apply_bottom and pt[1] > left_y_bottom:
                        continue
                    y_top_limit = top_y_min
                    y_top_poly = self._interpolar_y_en_polyline(roi.poly_top, pt[0])
                    if y_top_poly is not None:
                        y_top_limit = max(y_top_limit, y_top_poly)
                    y_bottom_limit = bottom_y_max
                    y_bottom_poly = self._interpolar_y_en_polyline(roi.poly_bottom, pt[0])
                    if y_bottom_poly is not None:
                        y_bottom_limit = min(y_bottom_limit, y_bottom_poly)
                    if pt[1] < y_top_limit:
                        pt = [pt[0], y_top_limit]
                    if pt[1] > y_bottom_limit:
                        pt = [pt[0], y_bottom_limit]
                    x_line = self._interpolar_x_en_linea(left_line, pt[1])
                    if x_line is not None and pt[0] < x_line - tol:
                        pt = [x_line, pt[1]]
                    if self._dist(pt, bl_anchor) > min_corner_dist and self._dist(pt, tl_anchor) > min_corner_dist:
                        polygon_pts.append(pt)
            # No añadir tl de nuevo (el polígono se cierra solo)
            
            if len(polygon_pts) < 4:
                return None
            
            # Limpiar puntos duplicados consecutivos
            cleaned = [polygon_pts[0]]
            for i in range(1, len(polygon_pts)):
                if self._dist(polygon_pts[i], cleaned[-1]) > 2:
                    cleaned.append(polygon_pts[i])
            
            if len(cleaned) < 4:
                return None
            
            # Convertir a array y clamp a dimensiones de frame
            polygon = np.array(cleaned, dtype=np.float32)
            h = self._img_height if hasattr(self, '_img_height') else 1080
            w = self._img_width if hasattr(self, '_img_width') else 1920
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
            
            # Validar área mínima
            area = cv2.contourArea(polygon.astype(np.int32))
            min_area = max(200.0, 0.0002 * w * h)
            if area < min_area:
                return None
            
            return polygon
        except Exception as e:
            print(f"[ERROR POLY] Crash in construction: {e}")
            import traceback; traceback.print_exc()
            return None
    
    def _dist(self, p1: list, p2: list) -> float:
        """Distancia euclidiana entre dos puntos (ROBUST)."""
        try:
            if p1 is None or p2 is None:
                return 9999.0
            # Asegurar que son accesibles como lista/array y tienen 2 elems
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
        except Exception:
            # print(f"[DEBUG DIST ERR] p1={p1} p2={p2}")
            return 9999.0  # Retornar distancia grande para no filtrar? O pequeña?
            # Si retorna grande -> el punto SE AGREGA (dist > 10).
            # Si retorna pequeña -> el punto NO se agrega (cleaner).
            # Para limpieza (cleaner), queremos que si falla, asuma DISTINTO -> grande.

    
    def _seleccionar_det_para_borde(
        self,
        dets: List[Dict],
        borde: str,
        clase_misma_ambos_lados: bool = False
    ) -> Optional[Dict]:
        """
        Selecciona la detección correcta para un borde, diferenciando por posición
        cuando la misma clase se usa para ambos lados (ej: Extremo izq/der).
        
        Args:
            dets: Lista de detecciones de la clase correspondiente
            borde: "left", "right", "top", "bottom"
            clase_misma_ambos_lados: True si la misma clase se usa para el borde opuesto
        
        Returns:
            La detección correcta, o None si no hay candidatos válidos
        """
        if not dets:
            return None
        
        # Filtrar detecciones que tengan edge_pts válidos para este borde
        pts_key = {
            "left": "left",   # Borde izq del ROI usa borde interior "left"
            "right": "right", # Borde der del ROI usa borde interior "right"
            "top": "top",     # Borde sup del ROI usa borde interior "top"
            "bottom": "bottom" # Borde inf del ROI usa borde interior "bottom"
        }.get(borde, borde)
        
        candidatos = []
        for det in dets:
            edge_pts = det.get("edge_pts", {})
            pts = edge_pts.get(pts_key) if isinstance(edge_pts, dict) else None
            if pts is not None and len(pts) >= self.MIN_CURVE_EDGE_PTS:
                # Calcular centro del bbox
                bbox = det.get("bbox", [])
                if len(bbox) >= 4:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    candidatos.append((det, cx, cy, pts))
        
        if not candidatos:
            return None
        
        if len(candidatos) == 1:
            return candidatos[0][0]
        
        # Si la misma clase se usa para ambos lados, ordenar por posición
        if clase_misma_ambos_lados:
            if borde == "left":
                # Para borde izquierdo, elegir el de menor cx
                candidatos.sort(key=lambda x: x[1])
                return candidatos[0][0]
            elif borde == "right":
                # Para borde derecho, elegir el de mayor cx
                candidatos.sort(key=lambda x: x[1], reverse=True)
                return candidatos[0][0]
            elif borde == "top":
                # Para borde superior, elegir el de menor cy
                candidatos.sort(key=lambda x: x[2])
                return candidatos[0][0]
            elif borde == "bottom":
                # Para borde inferior, elegir el de mayor cy
                candidatos.sort(key=lambda x: x[2], reverse=True)
                return candidatos[0][0]
        
        # Default: devolver el primero con puntos válidos
        return candidatos[0][0]
    
    def _calcular_polilineas_curvas(self, dets_por_clase: Dict[str, List[Dict]]) -> None:
        """Calcula polilíneas curvas para cada borde usando edge_pts."""
        cfg = self.config_sectores
        n_vert = cfg.curved_bins_vertical
        n_horiz = cfg.curved_bins_horizontal
        trim = cfg.curved_percentile_trim
        
        roi = self._roi_actual
        
        clase_izq = self.config_bordes.clase_izquierdo
        clase_der = self.config_bordes.clase_derecho
        clase_sup = self.config_bordes.clase_superior
        clase_inf = self.config_bordes.clase_inferior
        
        # Detectar si la misma clase se usa para bordes opuestos
        mismo_izq_der = (clase_izq and clase_der and 
                         clase_izq == clase_der and 
                         clase_izq not in ("(Ninguno)", "(Auto)"))
        mismo_sup_inf = (clase_sup and clase_inf and 
                         clase_sup == clase_inf and 
                         clase_sup not in ("(Ninguno)", "(Auto)"))
        
        # Borde izquierdo
        # Borde izquierdo
        if clase_izq and clase_izq not in ("(Ninguno)", "(Auto)"):
            det = self._seleccionar_det_para_borde(
                dets_por_clase.get(clase_izq, []),
                "left",
                clase_misma_ambos_lados=mismo_izq_der
            )
            if det:
                edge_pts = det.get("edge_pts", {})
                pts = edge_pts.get("left") if isinstance(edge_pts, dict) else None
                if pts is not None and len(pts) >= self.MIN_CURVE_EDGE_PTS:
                    poly = self._ajustar_polilinea_vertical(pts, n_vert, trim)
                    if poly is not None:
                        roi.poly_left = poly
        
        # Borde derecho
        if clase_der and clase_der not in ("(Ninguno)", "(Auto)"):
            det = self._seleccionar_det_para_borde(
                dets_por_clase.get(clase_der, []),
                "right",
                clase_misma_ambos_lados=mismo_izq_der
            )
            if det:
                edge_pts = det.get("edge_pts", {})
                pts = edge_pts.get("right") if isinstance(edge_pts, dict) else None
                if pts is not None and len(pts) >= self.MIN_CURVE_EDGE_PTS:
                    poly = self._ajustar_polilinea_vertical(pts, n_vert, trim)
                    if poly is not None:
                        roi.poly_right = poly
        
        # Borde superior
        if clase_sup and clase_sup not in ("(Ninguno)", "(Auto)"):
            det = self._seleccionar_det_para_borde(
                dets_por_clase.get(clase_sup, []),
                "top",
                clase_misma_ambos_lados=mismo_sup_inf
            )
            if det:
                edge_pts = det.get("edge_pts", {})
                pts = edge_pts.get("top") if isinstance(edge_pts, dict) else None
                if pts is not None and len(pts) >= self.MIN_CURVE_EDGE_PTS:
                    poly = self._ajustar_polilinea_horizontal(pts, n_horiz, trim)
                    if poly is not None:
                        roi.poly_top = poly
        
        # Borde inferior
        if clase_inf and clase_inf not in ("(Ninguno)", "(Auto)"):
            det = self._seleccionar_det_para_borde(
                dets_por_clase.get(clase_inf, []),
                "bottom",
                clase_misma_ambos_lados=mismo_sup_inf
            )
            if det:
                edge_pts = det.get("edge_pts", {})
                pts = edge_pts.get("bottom") if isinstance(edge_pts, dict) else None
                if pts is not None and len(pts) >= self.MIN_CURVE_EDGE_PTS:
                    poly = self._ajustar_polilinea_horizontal(pts, n_horiz, trim)
                    if poly is not None:
                        roi.poly_bottom = poly
        
        # Construir polígono cerrado
        roi.polygon_roi = self._construir_poligono_roi_curvo()
        roi.is_curved = roi.polygon_roi is not None
    
    # ===================== MÁSCARAS ROI (NUEVO) =====================
    
    def _generar_mascara_roi(self, h: int, w: int) -> np.ndarray:
        """Genera máscara binaria a partir del polígono/quad del ROI."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        roi = self._roi_actual
        if roi.corners is not None and roi.is_quad:
            # Interseccion de mascara vertical (left/right) y horizontal (top/bottom)
            try:
                tl = roi.corners[0].astype(np.float32)
                tr = roi.corners[1].astype(np.float32)
                br = roi.corners[2].astype(np.float32)
                bl = roi.corners[3].astype(np.float32)

                left_poly = roi.poly_left if roi.poly_left is not None and len(roi.poly_left) >= 2 else np.array([tl, bl], dtype=np.float32)
                right_poly = roi.poly_right if roi.poly_right is not None and len(roi.poly_right) >= 2 else np.array([tr, br], dtype=np.float32)
                top_poly = roi.poly_top if roi.poly_top is not None and len(roi.poly_top) >= 2 else np.array([tl, tr], dtype=np.float32)
                bottom_poly = roi.poly_bottom if roi.poly_bottom is not None and len(roi.poly_bottom) >= 2 else np.array([bl, br], dtype=np.float32)

                if left_poly[0][1] > left_poly[-1][1]:
                    left_poly = left_poly[::-1]
                if right_poly[0][1] > right_poly[-1][1]:
                    right_poly = right_poly[::-1]
                if top_poly[0][0] > top_poly[-1][0]:
                    top_poly = top_poly[::-1]
                if bottom_poly[0][0] > bottom_poly[-1][0]:
                    bottom_poly = bottom_poly[::-1]

                # Extender polilineas a las esquinas para cubrir todo el rango
                left_line = (float(tl[0]), float(tl[1]), float(bl[0]), float(bl[1]))
                right_line = (float(tr[0]), float(tr[1]), float(br[0]), float(br[1]))
                top_line = np.array([tl, tr], dtype=np.float32)
                bottom_line = np.array([bl, br], dtype=np.float32)

                y_min = float(min(tl[1], tr[1]))
                y_max = float(max(bl[1], br[1]))
                if left_poly[0][1] > y_min:
                    x_at = self._interpolar_x_en_linea(left_line, y_min)
                    if x_at is not None:
                        left_poly = np.vstack(([x_at, y_min], left_poly))
                if left_poly[-1][1] < y_max:
                    x_at = self._interpolar_x_en_linea(left_line, y_max)
                    if x_at is not None:
                        left_poly = np.vstack((left_poly, [x_at, y_max]))

                if right_poly[0][1] > y_min:
                    x_at = self._interpolar_x_en_linea(right_line, y_min)
                    if x_at is not None:
                        right_poly = np.vstack(([x_at, y_min], right_poly))
                if right_poly[-1][1] < y_max:
                    x_at = self._interpolar_x_en_linea(right_line, y_max)
                    if x_at is not None:
                        right_poly = np.vstack((right_poly, [x_at, y_max]))

                x_min = float(min(tl[0], bl[0]))
                x_max = float(max(tr[0], br[0]))
                if top_poly[0][0] > x_min:
                    y_at = self._interpolar_y_en_polyline(top_line, x_min)
                    if y_at is not None:
                        top_poly = np.vstack(([x_min, y_at], top_poly))
                if top_poly[-1][0] < x_max:
                    y_at = self._interpolar_y_en_polyline(top_line, x_max)
                    if y_at is not None:
                        top_poly = np.vstack((top_poly, [x_max, y_at]))

                if bottom_poly[0][0] > x_min:
                    y_at = self._interpolar_y_en_polyline(bottom_line, x_min)
                    if y_at is not None:
                        bottom_poly = np.vstack(([x_min, y_at], bottom_poly))
                if bottom_poly[-1][0] < x_max:
                    y_at = self._interpolar_y_en_polyline(bottom_line, x_max)
                    if y_at is not None:
                        bottom_poly = np.vstack((bottom_poly, [x_max, y_at]))

                poly_lr = np.vstack([left_poly, right_poly[::-1]]).astype(np.float32)
                poly_tb = np.vstack([top_poly, bottom_poly[::-1]]).astype(np.float32)

                poly_lr[:, 0] = np.clip(poly_lr[:, 0], 0, w - 1)
                poly_lr[:, 1] = np.clip(poly_lr[:, 1], 0, h - 1)
                poly_tb[:, 0] = np.clip(poly_tb[:, 0], 0, w - 1)
                poly_tb[:, 1] = np.clip(poly_tb[:, 1], 0, h - 1)

                mask_lr = np.zeros((h, w), dtype=np.uint8)
                mask_tb = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_lr, [poly_lr.astype(np.int32)], 255)
                cv2.fillPoly(mask_tb, [poly_tb.astype(np.int32)], 255)
                mask = cv2.bitwise_and(mask_lr, mask_tb)
                if mask.any():
                    return mask
            except Exception:
                pass

        if roi.is_curved and roi.polygon_roi is not None:
            pts = roi.polygon_roi.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif roi.is_quad and roi.corners is not None:
            pts = roi.corners.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            x1, y1 = roi.x_izquierda, roi.y_superior
            x2, y2 = roi.x_derecha, roi.y_inferior
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def _aplicar_padding_mascara(self, mask: np.ndarray) -> np.ndarray:
        """Aplica padding borrando los bordes de la máscara ROI."""
        cfg = self.config_sectores
        padded = mask.copy()
        h, w = mask.shape
        
        if cfg.padding_top_px > 0:
            padded[:min(cfg.padding_top_px, h), :] = 0
        if cfg.padding_bottom_px > 0:
            padded[max(0, h - cfg.padding_bottom_px):, :] = 0
        if cfg.padding_left_px > 0:
            padded[:, :min(cfg.padding_left_px, w)] = 0
        if cfg.padding_right_px > 0:
            padded[:, max(0, w - cfg.padding_right_px):] = 0
        
        return padded
    
    def punto_en_malla(self, x: int, y: int, use_padding: bool = True) -> bool:
        """Comprueba si un punto está dentro de la malla (con padding opcional)."""
        roi = self._roi_actual
        mask = roi.roi_mask_padded if use_padding else roi.roi_mask
        
        if mask is None:
            return True  # Sin máscara = sin filtrado
        
        h, w = mask.shape
        if 0 <= x < w and 0 <= y < h:
            return mask[int(y), int(x)] > 0
        return False
    
    # ===================== RESTRICCIONES POR CLASE (NUEVO) =====================
    
    def set_config_restricciones(self, enabled: bool, por_clase: Dict[str, Dict]) -> None:
        """Configura las restricciones de detección por clase."""
        self.config_restricciones.enabled = enabled
        self.config_restricciones.por_clase.clear()
        
        for clase, cfg in por_clase.items():
            self.config_restricciones.por_clase[clase] = RestriccionClase(
                modo=cfg.get("modo", "sin_restriccion"),
                sectores=cfg.get("sectores", [])
            )
    
    def filtrar_deteccion(self, det: Dict) -> bool:
        """
        Retorna True si la detección debe MANTENERSE, False si debe filtrarse.
        Basado en restricciones por clase y pertenencia a malla/sector.
        Acepta ambos formatos de detección: (class_name, bbox) o (cls, bbox_xyxy).
        """
        if not self.config_restricciones.enabled:
            return True
        
        # NORMALIZAR: Aceptar class_name o cls
        class_name = det.get("class_name") or det.get("cls") or ""
        class_name = str(class_name)
        
        restriccion = self.config_restricciones.por_clase.get(class_name)
        
        if restriccion is None:
            return True  # Sin restricción para esta clase
        
        modo = restriccion.modo
        if modo == "sin_restriccion":
            return True
        
        # NORMALIZAR: Aceptar bbox o bbox_xyxy
        bbox = det.get("bbox") or det.get("bbox_xyxy") or []
        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        if len(bbox) < 4:
            return True
        
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        # use_padding=True para que coincida con el borde amarillo visible (que usa padding)
        en_malla = self.punto_en_malla(int(cx), int(cy), use_padding=True)
        
        if modo == "solo_malla":
            return en_malla
        elif modo == "solo_fuera_malla":
            return not en_malla
        elif modo == "solo_sectores":
            if not en_malla:
                return False
            sector = self.obtener_sector_para_punto(cx, cy)
            if sector is None:
                return False
            return sector in restriccion.sectores
        
        return True
    
    def _ajustar_linea_vertical(
        self,
        contour: np.ndarray,
        es_izquierdo: bool
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Ajusta una línea vertical (x = a*y + b) usando SOLO el borde interior del contorno.
        
        - Borde izquierdo: para cada altura Y, usar el X máximo (lado derecho del contorno)
        - Borde derecho: para cada altura Y, usar el X mínimo (lado izquierdo del contorno)
        """
        try:
            pts = np.array(contour).reshape(-1, 2)
            if len(pts) < 4:
                return None
            
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            
            # Filtrar outliers verticales (quedarse con percentil 5-95)
            y_min_raw, y_max_raw = y_vals.min(), y_vals.max()
            y_p5 = y_min_raw + (y_max_raw - y_min_raw) * 0.05
            y_p95 = y_min_raw + (y_max_raw - y_min_raw) * 0.95
            
            # Agrupar por bins de Y (cada 4 píxeles) y extraer el borde interior
            bin_size = 4
            y_bins = {}
            for x, y in zip(x_vals, y_vals):
                if y_p5 <= y <= y_p95:
                    bin_idx = int(y // bin_size)
                    if bin_idx not in y_bins:
                        y_bins[bin_idx] = []
                    y_bins[bin_idx].append((x, y))
            
            # Extraer puntos del borde interior
            edge_points = []
            for bin_idx, points in y_bins.items():
                if es_izquierdo:
                    # Borde izquierdo: usar X máximo (lado derecho del contorno, toca la banda)
                    best = max(points, key=lambda p: p[0])
                else:
                    # Borde derecho: usar X mínimo (lado izquierdo del contorno, toca la banda)
                    best = min(points, key=lambda p: p[0])
                edge_points.append(best)
            
            if len(edge_points) < 10:
                return None
            
            edge_points = np.array(edge_points)
            x_edge = edge_points[:, 0]
            y_edge = edge_points[:, 1]
            
            # Ajustar x = a*y + b usando mínimos cuadrados
            A = np.vstack([y_edge, np.ones(len(y_edge))]).T
            result = np.linalg.lstsq(A, x_edge, rcond=None)
            a, b = result[0]
            
            y_min, y_max = y_edge.min(), y_edge.max()
            x1 = a * y_min + b
            x2 = a * y_max + b
            
            return (float(x1), float(y_min), float(x2), float(y_max))
        except Exception:
            return None
    
    def _ajustar_linea_horizontal(
        self,
        contour: np.ndarray,
        es_superior: bool
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Ajusta una línea horizontal (y = a*x + b) usando SOLO el borde interior del contorno.
        
        - Borde superior: para cada X, usar el Y máximo (lado inferior del contorno, toca la banda)
        - Borde inferior: para cada X, usar el Y mínimo (lado superior del contorno, toca la banda)
        """
        try:
            pts = np.array(contour).reshape(-1, 2)
            if len(pts) < 4:
                return None
            
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            
            # Filtrar outliers horizontales (quedarse con percentil 5-95)
            x_min_raw, x_max_raw = x_vals.min(), x_vals.max()
            x_p5 = x_min_raw + (x_max_raw - x_min_raw) * 0.05
            x_p95 = x_min_raw + (x_max_raw - x_min_raw) * 0.95
            
            # Agrupar por bins de X (cada 4 píxeles) y extraer el borde interior
            bin_size = 4
            x_bins = {}
            for x, y in zip(x_vals, y_vals):
                if x_p5 <= x <= x_p95:
                    bin_idx = int(x // bin_size)
                    if bin_idx not in x_bins:
                        x_bins[bin_idx] = []
                    x_bins[bin_idx].append((x, y))
            
            # Extraer puntos del borde interior
            edge_points = []
            for bin_idx, points in x_bins.items():
                if es_superior:
                    # Borde superior: usar Y máximo (lado inferior del contorno, toca la banda)
                    best = max(points, key=lambda p: p[1])
                else:
                    # Borde inferior: usar Y mínimo (lado superior del contorno, toca la banda)
                    best = min(points, key=lambda p: p[1])
                edge_points.append(best)
            
            if len(edge_points) < 10:
                return None
            
            edge_points = np.array(edge_points)
            x_edge = edge_points[:, 0]
            y_edge = edge_points[:, 1]
            
            # Ajustar y = a*x + b usando mínimos cuadrados
            A = np.vstack([x_edge, np.ones(len(x_edge))]).T
            result = np.linalg.lstsq(A, y_edge, rcond=None)
            a, b = result[0]
            
            x_min, x_max = x_edge.min(), x_edge.max()
            y1 = a * x_min + b
            y2 = a * x_max + b
            
            return (float(x_min), float(y1), float(x_max), float(y2))
        except Exception:
            return None
    
    def _construir_corners_desde_lineas(
        self,
        linea_izq: Tuple[float, float, float, float],
        linea_der: Tuple[float, float, float, float],
        linea_sup: Optional[Tuple[float, float, float, float]],
        linea_inf: Optional[Tuple[float, float, float, float]],
        img_width: int,
        img_height: int
    ) -> Optional[np.ndarray]:
        """
        Construye las 4 esquinas del cuadrilátero a partir de las líneas de borde.
        Orden: [top-left, top-right, bottom-right, bottom-left]
        """
        try:
            # Si no hay linea sup/inf, derivar desde las laterales para no usar bordes de imagen.
            if linea_sup is None or linea_inf is None:
                lx1, ly1, lx2, ly2 = linea_izq
                rx1, ry1, rx2, ry2 = linea_der

                left_top = (lx1, ly1) if ly1 <= ly2 else (lx2, ly2)
                left_bottom = (lx1, ly1) if ly1 > ly2 else (lx2, ly2)
                right_top = (rx1, ry1) if ry1 <= ry2 else (rx2, ry2)
                right_bottom = (rx1, ry1) if ry1 > ry2 else (rx2, ry2)

                if linea_sup is None:
                    linea_sup = (left_top[0], left_top[1], right_top[0], right_top[1])
                if linea_inf is None:
                    linea_inf = (left_bottom[0], left_bottom[1], right_bottom[0], right_bottom[1])

            if linea_sup is None:
                linea_sup = (0, 0, img_width, 0)
            if linea_inf is None:
                linea_inf = (0, img_height, img_width, img_height)

            # Intersecciones
            tl = self._interseccion_lineas(linea_izq, linea_sup)
            tr = self._interseccion_lineas(linea_der, linea_sup)
            br = self._interseccion_lineas(linea_der, linea_inf)
            bl = self._interseccion_lineas(linea_izq, linea_inf)
            
            if None in (tl, tr, br, bl):
                return None
            
            corners = np.array([tl, tr, br, bl], dtype=np.float32)
            return corners
        except Exception:
            return None
    
    def _interseccion_lineas(
        self,
        linea1: Tuple[float, float, float, float],
        linea2: Tuple[float, float, float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Calcula la intersección de dos líneas definidas por dos puntos cada una.
        """
        x1, y1, x2, y2 = linea1
        x3, y3, x4, y4 = linea2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Líneas paralelas
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def _validar_corners(
        self,
        corners: np.ndarray,
        img_width: int,
        img_height: int
    ) -> bool:
        """
        Valida que las esquinas formen un cuadrilátero válido.
        """
        if corners is None or len(corners) != 4:
            return False
        
        # Verificar que estén dentro de la imagen (con margen)
        margin = 50
        for pt in corners:
            if pt[0] < -margin or pt[0] > img_width + margin:
                return False
            if pt[1] < -margin or pt[1] > img_height + margin:
                return False
        
        # Verificar área mínima
        area = cv2.contourArea(corners.astype(np.float32))
        min_area = img_width * img_height * 0.01  # Al menos 1% de la imagen
        if area < min_area:
            return False
        
        # Verificar que sea convexo
        if not cv2.isContourConvex(corners.astype(np.float32)):
            return False
        
        return True
    
    def _aplicar_inset(self, corners: np.ndarray) -> np.ndarray:
        """
        Contrae el cuadrilátero hacia dentro por inset_px píxeles.
        """
        inset = self.config_sectores.inset_px
        if inset <= 0:
            return corners.copy()
        
        # Calcular centroide
        centroid = corners.mean(axis=0)
        
        # Mover cada corner hacia el centroide
        result = corners.copy()
        for i in range(4):
            direction = centroid - corners[i]
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
                result[i] = corners[i] + direction * inset
        
        return result.astype(np.int32)
    
    def _calcular_homografia(
        self,
        corners: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calcula la homografía para mapear entre imagen y espacio rectificado.
        """
        try:
            # Puntos destino: rectángulo normalizado
            dst_pts = np.array([
                [0, 0],
                [self.RECT_WIDTH, 0],
                [self.RECT_WIDTH, self.RECT_HEIGHT],
                [0, self.RECT_HEIGHT]
            ], dtype=np.float32)
            
            src_pts = corners.astype(np.float32)
            
            H, _ = cv2.findHomography(src_pts, dst_pts)
            H_inv, _ = cv2.findHomography(dst_pts, src_pts)
            
            return H, H_inv
        except Exception:
            return None, None
    
    def _calcular_roi_rectangulo(
        self,
        dets_por_clase: Dict[str, List[Dict]],
        img_width: int,
        img_height: int
    ) -> None:
        """
        Fallback: calcula ROI como rectángulo axis-aligned.
        """
        x_izq = self._encontrar_borde_lateral_simple(
            dets_por_clase, self.config_bordes.clase_izquierdo, True, 0
        )
        x_der = self._encontrar_borde_lateral_simple(
            dets_por_clase, self.config_bordes.clase_derecho, False, img_width
        )
        y_sup = self._encontrar_borde_horizontal_simple(
            dets_por_clase, self.config_bordes.clase_superior, True, 0
        )
        y_inf = self._encontrar_borde_horizontal_simple(
            dets_por_clase, self.config_bordes.clase_inferior, False, img_height
        )
        
        # Validar
        if x_izq >= x_der:
            x_izq, x_der = 0, img_width
        if y_sup >= y_inf:
            y_sup, y_inf = 0, img_height
        
        self._roi_actual = ROIBanda(
            x_izquierda=x_izq,
            x_derecha=x_der,
            y_superior=y_sup,
            y_inferior=y_inf,
            is_quad=False,
            valido=True
        )
    
    def _encontrar_borde_lateral_simple(
        self,
        dets_por_clase: Dict[str, List[Dict]],
        clase: str,
        es_izquierdo: bool,
        default: int
    ) -> int:
        """Encuentra posición X de borde lateral usando bbox."""
        if not clase or clase not in dets_por_clase:
            return default
        
        dets = dets_por_clase[clase]
        if not dets:
            return default
        
        if clase == self.config_bordes.clase_izquierdo == self.config_bordes.clase_derecho:
            centers = [(d, (d["bbox"][0] + d["bbox"][2]) / 2) for d in dets if len(d.get("bbox", [])) >= 4]
            if len(centers) >= 2:
                centers.sort(key=lambda x: x[1])
                det = centers[0][0] if es_izquierdo else centers[-1][0]
            elif centers:
                det = centers[0][0]
            else:
                return default
        else:
            if es_izquierdo:
                det = min(dets, key=lambda d: d.get("bbox", [self._img_width])[0])
            else:
                det = max(dets, key=lambda d: d.get("bbox", [0,0,0,0])[2] if len(d.get("bbox",[])) >= 4 else 0)
        
        bbox = det.get("bbox", [])
        if len(bbox) >= 4:
            return int(bbox[2] if es_izquierdo else bbox[0])
        return default
    
    def _encontrar_borde_horizontal_simple(
        self,
        dets_por_clase: Dict[str, List[Dict]],
        clase: str,
        es_superior: bool,
        default: int
    ) -> int:
        """Encuentra posición Y de borde horizontal usando bbox."""
        if not clase or clase not in dets_por_clase:
            return default
        
        dets = dets_por_clase[clase]
        if not dets:
            return default
        
        if clase == self.config_bordes.clase_superior == self.config_bordes.clase_inferior:
            centers = [(d, (d["bbox"][1] + d["bbox"][3]) / 2) for d in dets if len(d.get("bbox", [])) >= 4]
            if len(centers) >= 2:
                centers.sort(key=lambda x: x[1])
                det = centers[0][0] if es_superior else centers[-1][0]
            elif centers:
                det = centers[0][0]
            else:
                return default
        else:
            if es_superior:
                det = min(dets, key=lambda d: d.get("bbox", [0, self._img_height])[1])
            else:
                det = max(dets, key=lambda d: d.get("bbox", [0,0,0,0])[3] if len(d.get("bbox",[])) >= 4 else 0)
        
        bbox = det.get("bbox", [])
        if len(bbox) >= 4:
            return int(bbox[3] if es_superior else bbox[1])
        return default
    
    # ===================== CÁLCULO DE SECTORES =====================
    
    def _recalcular_sectores(self) -> None:
        """Recalcula sectores y líneas divisorias."""
        roi = self._roi_actual
        
        if not roi.valido:
            # Si la ROI es inválida pero tenemos frozen/fallback, intentar recuperar
            if self.config_sectores.comportamiento_fallo == "Congelar" and getattr(self, "_last_valid_roi_backup", None) is not None:
                # Restaurar backup si es válido (fallback silencioso)
                roi = self._last_valid_roi_backup
                self._roi_actual = roi
            else:
                self._sectores = []
                self._lineas_sector_proyectadas = []
                # Invalidar fast path para reintentar pronto
                return
        
        # Guardar backup de ROI válida
        self._last_valid_roi_backup = roi
        
        modo = self.config_sectores.modo
        n_vert = self.config_sectores.num_verticales if modo in ("vertical", "rejilla") else 1
        n_horiz = self.config_sectores.num_horizontales if modo in ("horizontal", "rejilla") else 1
        
        if roi.is_quad and roi.H_inv is not None:
            self._recalcular_sectores_perspectiva(n_vert, n_horiz)
        else:
            self._recalcular_sectores_rectangulo(n_vert, n_horiz)

    def _calc_lineas_hash(self, quant_step: int = 1) -> int:
        if not self._lineas_sector_proyectadas:
            return 0
        def _quantize_val(value: int, step: int) -> int:
            if step <= 1:
                return int(value)
            return int(value // step * step)
        coords: list[int] = []
        for pt1, pt2 in self._lineas_sector_proyectadas:
            coords.append(_quantize_val(pt1[0], quant_step))
            coords.append(_quantize_val(pt1[1], quant_step))
            coords.append(_quantize_val(pt2[0], quant_step))
            coords.append(_quantize_val(pt2[1], quant_step))
        return hash(tuple(coords))
    
    
    def _recalcular_sectores_perspectiva(self, n_vert: int, n_horiz: int) -> None:
        """Calcula sectores y líneas usando homografía."""
        roi = self._roi_actual
        H_inv = roi.H_inv
        
        self._lineas_sector_proyectadas = []
        self._sectores = []
        
        # Líneas verticales en espacio rectificado → proyectar a imagen
        # Extendemos los límites para que siempre crucen el borde de la ROI (incluso curvas)
        margin = 1.0  # 100% de margen extra arriba y abajo
        for i in range(1, n_vert):
            u = i / n_vert * self.RECT_WIDTH
            # Línea extendida verticalmente
            pt1_rect = np.array([[u, -self.RECT_HEIGHT * margin]], dtype=np.float32)
            pt2_rect = np.array([[u, self.RECT_HEIGHT * (1 + margin)]], dtype=np.float32)
            pt1_img = cv2.perspectiveTransform(pt1_rect.reshape(1, 1, 2), H_inv).reshape(2)
            pt2_img = cv2.perspectiveTransform(pt2_rect.reshape(1, 1, 2), H_inv).reshape(2)
            self._lineas_sector_proyectadas.append((
                (int(pt1_img[0]), int(pt1_img[1])),
                (int(pt2_img[0]), int(pt2_img[1]))
            ))
        
        # Líneas horizontales
        for j in range(1, n_horiz):
            v = j / n_horiz * self.RECT_HEIGHT
            # Línea extendida horizontalmente
            pt1_rect = np.array([[-self.RECT_WIDTH * margin, v]], dtype=np.float32)
            pt2_rect = np.array([[self.RECT_WIDTH * (1 + margin), v]], dtype=np.float32)
            pt1_img = cv2.perspectiveTransform(pt1_rect.reshape(1, 1, 2), H_inv).reshape(2)
            pt2_img = cv2.perspectiveTransform(pt2_rect.reshape(1, 1, 2), H_inv).reshape(2)
            self._lineas_sector_proyectadas.append((
                (int(pt1_img[0]), int(pt1_img[1])),
                (int(pt2_img[0]), int(pt2_img[1]))
            ))
        
        # Crear sectores con esquinas proyectadas
        sector_id = 0
        for fila in range(n_horiz):
            for col in range(n_vert):
                u1 = col / n_vert
                u2 = (col + 1) / n_vert
                v1 = fila / n_horiz
                v2 = (fila + 1) / n_horiz
                
                # Proyectar las 4 esquinas del sector
                corners_rect = np.array([
                    [u1 * self.RECT_WIDTH, v1 * self.RECT_HEIGHT],
                    [u2 * self.RECT_WIDTH, v1 * self.RECT_HEIGHT],
                    [u2 * self.RECT_WIDTH, v2 * self.RECT_HEIGHT],
                    [u1 * self.RECT_WIDTH, v2 * self.RECT_HEIGHT]
                ], dtype=np.float32)
                
                corners_img = cv2.perspectiveTransform(
                    corners_rect.reshape(1, -1, 2), H_inv
                ).reshape(-1, 2).astype(np.int32)
                
                sector = InfoSector(
                    id=sector_id,
                    fila=fila,
                    columna=col,
                    corners_img=corners_img,
                    u1=u1, v1=v1, u2=u2, v2=v2
                )
                self._sectores.append(sector)
                sector_id += 1
    
    def _recalcular_sectores_rectangulo(self, n_vert: int, n_horiz: int) -> None:
        """Calcula sectores para ROI rectangular (fallback)."""
        roi = self._roi_actual
        
        self._lineas_sector_proyectadas = []
        self._sectores = []
        
        # Líneas verticales
        if n_vert > 1:
            step = roi.ancho / n_vert
            for i in range(1, n_vert):
                x = int(roi.x_izquierda + i * step)
                self._lineas_sector_proyectadas.append((
                    (x, roi.y_superior),
                    (x, roi.y_inferior)
                ))
        
        # Líneas horizontales
        if n_horiz > 1:
            step = roi.alto / n_horiz
            for j in range(1, n_horiz):
                y = int(roi.y_superior + j * step)
                self._lineas_sector_proyectadas.append((
                    (roi.x_izquierda, y),
                    (roi.x_derecha, y)
                ))
        
        # Sectores
        x_limits = [roi.x_izquierda]
        if n_vert > 1:
            step = roi.ancho / n_vert
            x_limits.extend([int(roi.x_izquierda + i * step) for i in range(1, n_vert)])
        x_limits.append(roi.x_derecha)
        
        y_limits = [roi.y_superior]
        if n_horiz > 1:
            step = roi.alto / n_horiz
            y_limits.extend([int(roi.y_superior + j * step) for j in range(1, n_horiz)])
        y_limits.append(roi.y_inferior)
        
        sector_id = 0
        for fila in range(len(y_limits) - 1):
            for col in range(len(x_limits) - 1):
                sector = InfoSector(
                    id=sector_id,
                    fila=fila,
                    columna=col,
                    x1=x_limits[col],
                    y1=y_limits[fila],
                    x2=x_limits[col + 1],
                    y2=y_limits[fila + 1]
                )
                self._sectores.append(sector)
                sector_id += 1
    
    # ===================== DIBUJO =====================
    
    def _dibujar_bordes(self, imagen: np.ndarray) -> None:
        """Dibuja los bordes de la banda (polígono curvo, quad o rectángulo)."""
        if not self.config_sectores.mostrar_borde_banda:
            return

        roi = self._roi_actual
        if not roi.valido:
            return
        
        color = self.config_sectores.color_bordes
        grosor = self.config_sectores.grosor_bordes
        
        drawn = False

        # USAR MASCARA CON PADDING PARA EL BORDE VISUAL (ACTIVO)
        mask = roi.roi_mask_padded
        if mask is not None and mask.size > 0:
            try:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    cv2.polylines(imagen, [contour], True, (0, 0, 0), grosor + 2)
                    cv2.polylines(imagen, [contour], True, color, grosor)
                    drawn = True
            except Exception:
                drawn = False
        
        # Fallback a quad (cuadrilátero basado en corners)
        if not drawn and roi.is_quad and roi.corners is not None:
            pts = roi.corners.reshape((-1, 1, 2)).astype(np.int32)
            # Sombra
            cv2.polylines(imagen, [pts], True, (0, 0, 0), grosor + 2)
            # Borde principal
            cv2.polylines(imagen, [pts], True, color, grosor)
            drawn = True
        
        # Fallback a rectángulo
        if not drawn:
            pt1 = (roi.x_izquierda, roi.y_superior)
            pt2 = (roi.x_derecha, roi.y_inferior)
            cv2.rectangle(imagen, pt1, pt2, (0, 0, 0), grosor + 2)
            cv2.rectangle(imagen, pt1, pt2, color, grosor)
    
    def _dibujar_sectores(self, imagen: np.ndarray) -> None:
        """Dibuja las lineas de sectorizacion recortadas por la mascara ROI."""
        if not self.config_sectores.mostrar_sectorizacion:
            return

        color = self.config_sectores.color_lineas
        grosor = self.config_sectores.grosor_lineas
        opacity = self.config_sectores.opacidad_lineas
        roi = self._roi_actual

        if roi is None or not roi.valido:
            return
        if not self._lineas_sector_proyectadas:
            return

        # USAR MASCARA CON PADDING PARA DIBUJAR (ACTIVO)
        mask = roi.roi_mask_padded
        if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
            h, w = imagen.shape[:2]
            if mask.shape[:2] == (h, w):
                overlay = np.zeros_like(imagen)
                for pt1, pt2 in self._lineas_sector_proyectadas:
                    cv2.line(overlay, pt1, pt2, (30, 30, 30), grosor + 1)
                    cv2.line(overlay, pt1, pt2, color, grosor)

                if mask.max() <= 1:
                    mask_u8 = (mask.astype(np.uint8) * 255)
                else:
                    mask_u8 = mask.astype(np.uint8)

                overlay_masked = cv2.bitwise_and(overlay, overlay, mask=mask_u8)
                line_pixels = np.any(overlay_masked > 0, axis=2)
                if opacity >= 0.99:
                    imagen[line_pixels] = overlay_masked[line_pixels]
                else:
                    blended = cv2.addWeighted(overlay_masked, opacity, imagen, 1.0, 0)
                    imagen[line_pixels] = blended[line_pixels]
                return

        if opacity >= 0.99:
            for pt1, pt2 in self._lineas_sector_proyectadas:
                cv2.line(imagen, pt1, pt2, (0, 0, 0), grosor + 1)
                cv2.line(imagen, pt1, pt2, color, grosor)
        else:
            overlay = imagen.copy()
            for pt1, pt2 in self._lineas_sector_proyectadas:
                cv2.line(overlay, pt1, pt2, (0, 0, 0), grosor + 1)
                cv2.line(overlay, pt1, pt2, color, grosor)
            cv2.addWeighted(overlay, opacity, imagen, 1 - opacity, 0, imagen)

    def _dibujar_etiquetas(self, imagen: np.ndarray) -> None:
        """Dibuja etiquetas de sector centradas."""
        for sector in self._sectores:
            cx, cy = sector.centro
            texto = str(sector.id + 1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(texto, font, font_scale, thickness)
            
            tx = cx - tw // 2
            ty = cy + th // 2
            
            padding = 3
            cv2.rectangle(
                imagen,
                (tx - padding, ty - th - padding),
                (tx + tw + padding, ty + padding),
                (0, 0, 0),
                -1
            )
            cv2.putText(imagen, texto, (tx, ty), font, font_scale, (255, 255, 255), thickness)
    
    def _dibujar_debug_overlay(self, imagen: np.ndarray) -> None:
        """Dibuja overlay de depuración: puntos usados para ajuste y líneas ajustadas."""
        if not self.config_sectores.debug_overlay:
            return
        
        # Colores para cada borde
        colores = {
            'left': (255, 0, 0),    # Azul
            'right': (0, 0, 255),   # Rojo
            'top': (0, 255, 0),     # Verde
            'bottom': (255, 0, 255) # Magenta
        }
        
        roi = self._roi_actual
        
        # NUEVO: Dibujar polilíneas curvas si existen
        if roi and roi.is_curved:
            polylines = [
                ('left', roi.poly_left, (255, 100, 100)),
                ('right', roi.poly_right, (100, 100, 255)),
                ('top', roi.poly_top, (100, 255, 100)),
                ('bottom', roi.poly_bottom, (255, 100, 255)),
            ]
            for name, poly, color in polylines:
                if poly is not None and len(poly) > 1:
                    pts = poly.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(imagen, [pts], False, color, 2, cv2.LINE_AA)
        
        # NUEVO: Dibujar contorno de roi_mask_padded
        if roi and roi.roi_mask_padded is not None:
            try:
                mask_uint8 = (roi.roi_mask_padded.astype(np.uint8) * 255) if roi.roi_mask_padded.max() <= 1 else roi.roi_mask_padded.astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(imagen, contours, -1, (128, 255, 128), 1, cv2.LINE_AA)
            except Exception:
                pass
        
        # Dibujar puntos edge
        for key, pts in self._debug_edge_pts.items():
            color = colores.get(key, (255, 255, 255))
            if pts is not None and len(pts) > 0:
                # Dibujar cada punto (submuestrear si hay muchos)
                step = max(1, len(pts) // 50)
                for i in range(0, len(pts), step):
                    x, y = int(pts[i][0]), int(pts[i][1])
                    cv2.circle(imagen, (x, y), 2, color, -1)
        
        # Dibujar líneas ajustadas con color diferente (NUEVO: chequear si existen)
        if hasattr(self, '_debug_lineas'):
            for key, linea in self._debug_lineas.items():
                color = colores.get(key, (255, 255, 255))
                x1, y1, x2, y2 = linea
                cv2.line(imagen, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Texto de info
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        for key in ['left', 'right', 'top', 'bottom']:
            n_pts = len(self._debug_edge_pts.get(key, []))
            # Ajuste de lógica de método para evitar fallos si no hay debug_lineas
            has_line = hasattr(self, '_debug_lineas') and key in self._debug_lineas
            method = "edge_pts" if n_pts > 0 else ("contour" if has_line else "bbox")
            texto = f"{key}: {method} (n={n_pts})"
            color = colores.get(key, (255, 255, 255))
            cv2.putText(imagen, texto, (10, y_pos), font, 0.5, (0, 0, 0), 2)
            cv2.putText(imagen, texto, (10, y_pos), font, 0.5, color, 1)
            y_pos += 20
        
        # NUEVO: Texto diagnóstico de curvas y padding
        y_pos += 10
        cfg = self.config_sectores
        info_lines = [
            f"curved: {roi.is_curved if roi else False}",
            f"padding: T{cfg.padding_top_px} B{cfg.padding_bottom_px} L{cfg.padding_left_px} R{cfg.padding_right_px}",
            f"bins: V{cfg.curved_bins_vertical} H{cfg.curved_bins_horizontal}",
        ]
        for line in info_lines:
            cv2.putText(imagen, line, (10, y_pos), font, 0.4, (0, 0, 0), 2)
            cv2.putText(imagen, line, (10, y_pos), font, 0.4, (255, 255, 255), 1)
            y_pos += 15
    
    # ===================== ASIGNACIÓN DE SECTOR =====================
    
    def obtener_sector_para_punto(self, x: float, y: float, require_in_roi: bool = True) -> Optional[int]:
        """
        Obtiene el ID del sector que contiene el punto dado.
        OPTIMIZADO: Usa cálculo directo O(1) en lugar de bucle O(n).
        """
        roi = self._roi_actual
        if not roi.valido:
            return None
        
        # Validar contra máscara ROI con padding (visible)
        if require_in_roi:
            if not self.punto_en_malla(int(x), int(y), use_padding=True):
                return None
        
        cfg = self.config_sectores
        n_cols = cfg.num_verticales
        n_rows = cfg.num_horizontales
        
        if roi.is_quad and roi.H is not None:
            # Proyectar punto a espacio rectificado
            try:
                pt = np.array([[x, y]], dtype=np.float32).reshape(1, 1, 2)
                pt_rect = cv2.perspectiveTransform(pt, roi.H).reshape(2)
                
                # Coordenadas normalizadas [0, 1]
                u = np.clip(pt_rect[0] / self.RECT_WIDTH, 0.0, 0.9999)
                v = np.clip(pt_rect[1] / self.RECT_HEIGHT, 0.0, 0.9999)
                
                # OPTIMIZADO: Cálculo directo del sector sin bucle
                col = int(u * n_cols)
                row = int(v * n_rows)
                sector_id = row * n_cols + col
                
                if 0 <= sector_id < len(self._sectores):
                    return sector_id
            except Exception:
                pass
        
        # Fallback: modo rectángulo con cálculo directo
        if roi.ancho > 0 and roi.alto > 0:
            # Normalizar coordenadas al ROI
            u = (x - roi.x_izquierda) / roi.ancho
            v = (y - roi.y_superior) / roi.alto
            
            if 0 <= u < 1 and 0 <= v < 1:
                col = int(u * n_cols)
                row = int(v * n_rows)
                sector_id = row * n_cols + col
                
                if 0 <= sector_id < len(self._sectores):
                    return sector_id
        
        return None
    
    def asignar_sectores_a_detecciones(self, detecciones: List[Dict]) -> List[Dict]:
        """Asigna sector a cada detección usando el centro del bbox."""
        for det in detecciones:
            bbox = det.get("bbox", [])
            if len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                det["sector"] = self.obtener_sector_para_punto(cx, cy)
            else:
                det["sector"] = None
        return detecciones
    
    # ===================== API PÚBLICA =====================
    
    def obtener_roi(self) -> ROIBanda:
        return self._roi_actual
    
    def obtener_sectores(self) -> List[InfoSector]:
        return self._sectores.copy()
    
    def obtener_num_sectores(self) -> int:
        return len(self._sectores)
    
    def obtener_config_dict(self) -> Dict[str, Any]:
        """Devuelve configuración para persistencia."""
        # Serializar restricciones por clase
        restricciones_dict = {}
        for clase, rest in self.config_restricciones.por_clase.items():
            restricciones_dict[clase] = {
                "modo": rest.modo,
                "sectores": rest.sectores
            }
        
        return {
            "bordes": {
                "clase_superior": self.config_bordes.clase_superior,
                "clase_inferior": self.config_bordes.clase_inferior,
                "clase_izquierdo": self.config_bordes.clase_izquierdo,
                "clase_derecho": self.config_bordes.clase_derecho,
            },
            "sectores": {
                "modo": self.config_sectores.modo,
                "num_verticales": self.config_sectores.num_verticales,
                "num_horizontales": self.config_sectores.num_horizontales,
                "mostrar_etiquetas": self.config_sectores.mostrar_etiquetas,
                "mostrar_sectorizacion": self.config_sectores.mostrar_sectorizacion,
                "color_lineas": list(self.config_sectores.color_lineas),
                "color_bordes": list(self.config_sectores.color_bordes),
                "grosor_lineas": self.config_sectores.grosor_lineas,
                "grosor_bordes": self.config_sectores.grosor_bordes,
                "use_perspective": self.config_sectores.use_perspective,
                "use_border_masks": self.config_sectores.use_border_masks,
                "smooth_alpha": self.config_sectores.smooth_alpha,
                "max_corner_jump_px": self.config_sectores.max_corner_jump_px,
                "inset_px": self.config_sectores.inset_px,
                # NUEVO: bordes curvos
                "curved_edges_enabled": self.config_sectores.curved_edges_enabled,
                "curved_bins_vertical": self.config_sectores.curved_bins_vertical,
                "curved_bins_horizontal": self.config_sectores.curved_bins_horizontal,
                "curved_percentile_trim": self.config_sectores.curved_percentile_trim,
                # NUEVO: padding
                "padding_top_px": self.config_sectores.padding_top_px,
                "padding_bottom_px": self.config_sectores.padding_bottom_px,
                "padding_left_px": self.config_sectores.padding_left_px,
                "padding_right_px": self.config_sectores.padding_right_px,
            },
            # NUEVO: restricciones por clase
            "restricciones": {
                "enabled": self.config_restricciones.enabled,
                "por_clase": restricciones_dict
            }
        }
    
    def cargar_config_dict(self, config: Dict[str, Any]) -> None:
        """Carga configuración desde diccionario."""
        if not config:
            return
        
        bordes = config.get("bordes", {})
        if bordes:
            self.set_config_bordes(
                clase_superior=bordes.get("clase_superior", ""),
                clase_inferior=bordes.get("clase_inferior", ""),
                clase_izquierdo=bordes.get("clase_izquierdo", ""),
                clase_derecho=bordes.get("clase_derecho", ""),
            )
        
        sectores = config.get("sectores", {})
        if sectores:
            color_lineas = sectores.get("color_lineas", [255, 255, 255])
            color_bordes = sectores.get("color_bordes", [0, 255, 255])
            
            # Actualizar campos de ConfigSectores directamente para nuevos campos
            self.config_sectores.modo = sectores.get("modo", "vertical")
            self.config_sectores.num_verticales = sectores.get("num_verticales", 1)
            self.config_sectores.num_horizontales = sectores.get("num_horizontales", 1)
            self.config_sectores.mostrar_etiquetas = sectores.get("mostrar_etiquetas", True)
            self.config_sectores.mostrar_sectorizacion = sectores.get("mostrar_sectorizacion", True)
            self.config_sectores.color_lineas = tuple(color_lineas) if isinstance(color_lineas, list) else color_lineas
            self.config_sectores.color_bordes = tuple(color_bordes) if isinstance(color_bordes, list) else color_bordes
            self.config_sectores.grosor_lineas = sectores.get("grosor_lineas", 1)
            self.config_sectores.grosor_bordes = sectores.get("grosor_bordes", 2)
            self.config_sectores.use_perspective = sectores.get("use_perspective", True)
            self.config_sectores.use_border_masks = sectores.get("use_border_masks", True)
            self.config_sectores.smooth_alpha = sectores.get("smooth_alpha", 0.15)
            self.config_sectores.max_corner_jump_px = sectores.get("max_corner_jump_px", 50.0)
            self.config_sectores.inset_px = sectores.get("inset_px", 0)
            # NUEVO: bordes curvos
            self.config_sectores.curved_edges_enabled = sectores.get("curved_edges_enabled", False)
            self.config_sectores.curved_bins_vertical = sectores.get("curved_bins_vertical", 7)
            self.config_sectores.curved_bins_horizontal = sectores.get("curved_bins_horizontal", 7)
            self.config_sectores.curved_percentile_trim = sectores.get("curved_percentile_trim", 0.10)
            # NUEVO: padding
            self.config_sectores.padding_top_px = sectores.get("padding_top_px", 0)
            self.config_sectores.padding_bottom_px = sectores.get("padding_bottom_px", 0)
            self.config_sectores.padding_left_px = sectores.get("padding_left_px", 0)
            self.config_sectores.padding_right_px = sectores.get("padding_right_px", 0)
        
        # NUEVO: Cargar restricciones por clase
        restricciones = config.get("restricciones", {})
        if restricciones:
            self.set_config_restricciones(
                enabled=restricciones.get("enabled", False),
                por_clase=restricciones.get("por_clase", {})
            )
