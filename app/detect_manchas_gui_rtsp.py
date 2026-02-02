# -*- coding: utf-8 -*-
"""
DetecciÃ³n de Manchas - Interfaz Simplificada

CaracterÃ­sticas:
- VisualizaciÃ³n automÃ¡tica del vÃ­deo ajustado a ventana.
- SelecciÃ³n fÃ¡cil de modelo y vÃ­deo.
- ConfiguraciÃ³n optimizada para detecciÃ³n de manchas.
"""

from __future__ import annotations

import os
import sys
import weakref
import shutil

# Agrupar iconos en la barra de tareas de Windows
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MetalRollBand.ManchasGuida.App.v1")
    except Exception:
        pass

import time
import json
import math
import copy
import uuid
import queue
import socket
import struct
import signal
import logging
import traceback
import random
import string
import threading
import datetime
import subprocess
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from itertools import cycle
from types import TracebackType
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from contextlib import nullcontext
from typing import Optional, Callable, Any

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None
from ultralytics import YOLO
from collections import deque
from itertools import cycle
from typing import Iterable

from sector_widgets import SectorControlPanel

try:
    from tooltip_manager import InfoIcon, init_tooltips
except Exception:  # noqa: BLE001
    InfoIcon = None  # type: ignore[assignment]

    def init_tooltips(*_args, **_kwargs):
        return None

LOGGER = logging.getLogger("DetectorRTSP")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.setLevel(logging.WARNING)
    LOGGER.setLevel(logging.WARNING)
    LOGGER.setLevel(logging.ERROR)
    LOGGER.setLevel(logging.CRITICAL)
    LOGGER.propagate = True

SENDTOPLC_IMPORT_ERROR: Exception | None = None
SENDTOPLC_IMPORT_TRACE: str | None = None

try:
    from sectorizador import Sectorizador
except ImportError:
    Sectorizador = None
    LOGGER.warning("No se pudo importar sectorizador; se desactiva la sectorizaciÃ³n.")

try:
    from sendToPLC_service import (
        SendToPLCService,
        SendToPLCWindow,
        is_detector_started,
        mark_detector_started,
        mark_detector_stopped,
    )
except Exception as exc:  # noqa: BLE001
    SENDTOPLC_IMPORT_ERROR = exc
    SENDTOPLC_IMPORT_TRACE = traceback.format_exc()
    SendToPLCService = None  # type: ignore[assignment]
    SendToPLCWindow = None  # type: ignore[assignment]

    LOGGER.warning(
        "No se pudo importar sendToPLC_service; se desactivan integraciones PLC. Detalle: %s",
        exc,
    )
    if SENDTOPLC_IMPORT_TRACE:
        LOGGER.debug("Traza import sendToPLC_service:\n%s", SENDTOPLC_IMPORT_TRACE)

    def is_detector_started() -> bool:
        return False

    def mark_detector_started() -> None:
        return None

    def mark_detector_stopped() -> None:
        return None

try:
    from dashboard_widgets import DashboardWidgetManager, FPSConfigDialog
except ImportError:
    DashboardWidgetManager = None  # type: ignore[assignment]
    FPSConfigDialog = None # type: ignore[assignment]
    LOGGER.warning("No se pudo importar dashboard_widgets; widgets personalizables desactivados.")

try:
    import garbage_collector
except ImportError:
    garbage_collector = None
    LOGGER.warning("No se pudo importar garbage_collector; limpieza automatica desactivada.")

try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise SystemExit(
        "PyTorch no estÃ¡ instalado. Instala una versiÃ³n con CUDA (cu128):\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    ) from e

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics no estÃ¡ instalado. Ejecuta: pip install -U ultralytics") from e

try:
    from ultralytics.utils.ops import letterbox as _ultra_letterbox, scale_boxes, scale_masks, process_mask, process_mask_native
except ImportError:
    try:
        from ultralytics.yolo.utils.ops import letterbox as _ultra_letterbox, scale_boxes, scale_masks, process_mask, process_mask_native  # type: ignore
    except ImportError:
        _ultra_letterbox = None
        try:
            from ultralytics.utils.ops import scale_boxes, scale_masks, process_mask, process_mask_native  # type: ignore
        except ImportError:
            try:
                from ultralytics.yolo.utils.ops import scale_boxes, scale_masks, process_mask, process_mask_native  # type: ignore
            except ImportError as err:
                raise SystemExit(
                    "Ultralytics instalado no expone `scale_boxes`. Actualiza el paquete con: pip install -U ultralytics"
                ) from err

try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    try:
        from ultralytics.yolo.utils.ops import non_max_suppression  # type: ignore
    except ImportError:
        try:
            from ultralytics.utils.nms import non_max_suppression  # type: ignore
        except ImportError as err:
            raise SystemExit(
                "Ultralytics instalado no expone `non_max_suppression`. Actualiza el paquete con: pip install -U ultralytics"
            ) from err


def get_resource_path(relative_path: str, is_config: bool = False) -> str:
    """
    Obtiene la ruta absoluta de un recurso, compatible con PyInstaller.
    Para archivos de configuraciÃ³n o datos que deben ser editables (fuera del .exe),
    busca en el directorio del ejecutable o del script.
    """
    if getattr(sys, 'frozen', False):
        # Estamos en un ejecutable empaquetado
        base_dir = os.path.dirname(sys.executable)
        if not is_config:
            # Recursos internos si los hubiera (ej: iconos empaquetados)
            meipass = getattr(sys, '_MEIPASS', base_dir)
            return os.path.join(meipass, relative_path)
    else:
        # Estamos en entorno de desarrollo
        base_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.abspath(os.path.join(base_dir, relative_path))


DEFAULT_MODEL = ""
DEFAULT_MODEL2 = ""
DEFAULT_VIDEO = ""

# Rutas relativas desde la raíz del proyecto o el ejecutable (PyInstaller onefile extrae en _MEI)
CONFIG_PATH = get_resource_path(os.path.join("..", "config", "detect_manchas_config.json"), is_config=True)
TOOLTIPS_PATH = get_resource_path(os.path.join("..", "config", "tooltips.json"), is_config=True)
# IMPORTANTE: no usar ".." aquí; apuntar directo a bin/ffmpeg/bin para que funcione desde _MEIPASS
FFMPEG_BIN_DIR = get_resource_path(os.path.join("..", "bin", "ffmpeg", "bin"))
SNAPSHOT_FILENAME = get_resource_path(os.path.join("..", "data", "estado_linea.json"), is_config=True)
MAX_CAPTURE_FILES = 200
HEARTBEAT_INTERVAL_SEC = 5.0
PERF_SAMPLE_INTERVAL_SEC = 0.5
PERF_HISTORY_SECONDS = 600.0  # mantener 10 minutos de historial para la gráfica
SNAPSHOT_DEFAULT_WRITE_INTERVAL_MS = 1500
SNAPSHOT_MIN_WRITE_INTERVAL_MS = 200
SNAPSHOT_DEFAULT_CLEAN_INTERVAL_SEC = 300.0
SNAPSHOT_SHORT_WINDOW_SEC = 3.0
SNAPSHOT_LONG_WINDOW_SEC = 30.0
SNAPSHOT_CHAIN_GAP_SEC = 2.0
SNAPSHOT_RECENT_EVENTS_MAX = 10
SNAPSHOT_MAJOR_CLASSES = {"Mancha", "Soldadura", "Extremo"}
SNAPSHOT_MAJOR_AREA_THRESHOLD = 8000.0
SNAPSHOT_LOW_CONF_THRESHOLD = 0.45

try:
    init_tooltips(TOOLTIPS_PATH)
    LOGGER.info("Tooltips loaded from %s", TOOLTIPS_PATH)
except Exception as exc:  # noqa: BLE001
    LOGGER.warning("No se pudieron cargar tooltips: %s", exc)

RTSP_WATCHDOG_TIMEOUT_SEC = 20.0
RTSP_FAIL_ALERT_SEC = 60.0

CALIBRATION_REQUIRED_KEYS = (
    "cam_height_cm",
    "visible_len_cm",
    "row_near_px",
    "row_far_px",
    "a4_dist_cm",
    "a4_real_cm",
    "a4_px",
)

DEFAULT_CALIBRATION = {
    "cam_height_cm": 110.0,
    "visible_len_cm": 105.0,
    "row_near_px": 320.0,
    "row_far_px": 980.0,
    "a4_dist_cm": 50.0,
    "a4_real_cm": 29.7,
    "a4_px": 400.0,
}


AREA_MODE_OFF = "off"
AREA_MODE_PX = "px"
AREA_MODE_CM2 = "cm2"
AREA_MODE_BOTH = "both"
AREA_MODE_INHERIT = "inherit"
_AREA_MODE_ALLOWED = {AREA_MODE_OFF, AREA_MODE_PX, AREA_MODE_CM2, AREA_MODE_BOTH, AREA_MODE_INHERIT}
AREA_MODE_DISPLAY = {
    AREA_MODE_OFF: "Sin etiqueta",
    # ASCII: OpenCV no renderiza bien superíndices (Â²) y terminan viéndose como "??" en el overlay.
    AREA_MODE_PX: "px^2",
    AREA_MODE_CM2: "cm^2",
    AREA_MODE_BOTH: "px^2 + cm^2",
    AREA_MODE_INHERIT: "Heredar",
}
AREA_MODE_FROM_DISPLAY = {label: key for key, label in AREA_MODE_DISPLAY.items()}


AREA_CM2_TO_M2_THRESHOLD = 100.0


def _coerce_area_mode(value: object, default: str = AREA_MODE_CM2) -> str:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        if raw in AREA_MODE_FROM_DISPLAY:
            return AREA_MODE_FROM_DISPLAY[raw]
        key = raw.lower()
        if key in _AREA_MODE_ALLOWED:
            return key
        for label, mode in AREA_MODE_FROM_DISPLAY.items():
            if label.lower() == key:
                return mode
        compact = key.replace(" ", "")
        compact = compact.replace("^2", "2").replace("Â²", "2")
        compact = "".join(ch for ch in compact if ch.isascii() and (ch.isalnum() or ch in {"+", "_", "-"}))
        if compact in {"off", "none", "sinetiqueta", "sin_etiqueta", "sin-etiqueta"}:
            return AREA_MODE_OFF
        if compact in {"inherit", "heredar"}:
            return AREA_MODE_INHERIT
        if compact in {"px2", "px"}:
            return AREA_MODE_PX
        if compact in {"cm2", "cm"}:
            return AREA_MODE_CM2
        if compact in {"both", "px2+cm2", "px2+cm", "px+cm2", "px+cm"}:
            return AREA_MODE_BOTH
    if isinstance(value, bool):
        return AREA_MODE_CM2 if value else AREA_MODE_OFF
    return default


def _area_enabled_from_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "1", "yes", "on"}:
            return True
        if key in {"false", "0", "no", "off"}:
            return False
    mode = _coerce_area_mode(value, AREA_MODE_OFF)
    if mode == AREA_MODE_INHERIT:
        return False
    return mode != AREA_MODE_OFF


AREA_MODE_CHOICES_GLOBAL = [AREA_MODE_DISPLAY[k] for k in (AREA_MODE_OFF, AREA_MODE_PX, AREA_MODE_CM2, AREA_MODE_BOTH)]
AREA_MODE_CHOICES_CLASS = [AREA_MODE_DISPLAY[k] for k in (AREA_MODE_INHERIT, AREA_MODE_OFF, AREA_MODE_PX, AREA_MODE_CM2)]


def _resolve_area_mode(value: object, global_mode: str) -> str:
    mode = _coerce_area_mode(value, AREA_MODE_INHERIT)
    if mode == AREA_MODE_INHERIT:
        return global_mode
    return mode


def _format_area_cm_value(area_cm2: float) -> str:
    if area_cm2 >= AREA_CM2_TO_M2_THRESHOLD:
        area_m2 = area_cm2 / 10000.0
        return f"/[{area_m2:.2f} m^2]"
    return f"/[{area_cm2:.1f} cm^2]"


def _clamp01(val: object, default: float = 0.0) -> float:
    try:
        v = float(val)
    except Exception:
        return float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_calibration(raw: dict | None) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, float] = {}
    for key in CALIBRATION_REQUIRED_KEYS:
        value = raw.get(key)
        if value is None:
            return None
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            return None
    return normalized


def bbox_area_cm2(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    cy: float,
    calib: dict[str, float],
    area_px: float | None = None,
    *,
    details: dict | None = None,
) -> float:
    """Convierte el área de un bbox en píxeles a cm² usando la geometría del montaje."""

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    px_area = area_px
    if px_area is None:
        px_area = width * height

    y0 = float(calib["row_near_px"])
    y1p = float(calib["row_far_px"])
    denom = float(y1p - y0)
    if abs(denom) < 1e-6:
        raise ValueError("Calibración inválida: row_far_px y row_near_px son iguales")

    t = (float(cy) - y0) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    x_cm = t * float(calib["visible_len_cm"])

    cam_height = float(calib["cam_height_cm"])
    distance = math.sqrt(cam_height * cam_height + x_cm * x_cm)

    a4_px = float(calib["a4_px"])
    a4_dist = float(calib["a4_dist_cm"])
    a4_real = float(calib["a4_real_cm"])
    if abs(a4_real) < 1e-6 or abs(a4_px) < 1e-6:
        raise ValueError("Calibración inválida: a4_real_cm o a4_px nulos")

    f_px = (a4_px * a4_dist) / a4_real
    cm_per_px = distance / f_px
    area_cm2 = px_area * (cm_per_px ** 2)

    if details is not None:
        details["t"] = t
        details["x_cm"] = x_cm
        details["distance_cm"] = distance
        details["cm_per_px"] = cm_per_px
        details["area_px"] = px_area

    return area_cm2


# ---------------------------------------------------------------------------
# Recursos gráficos y eventos externos (overlays, capturas) provenientes de sendToPLC
# ---------------------------------------------------------------------------

def _load_default_font(size: int = 20) -> ImageFont.ImageFont:
    """Carga una fuente TrueType si está disponible; vuelve a la fuente por defecto si falla."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


_OVERLAY_FONT = _load_default_font(20)


def _tk_color_to_bgr(color: str) -> tuple[int, int, int]:
    """Convierte un color Tk (#RRGGBB) a BGR para OpenCV."""
    color = (color or "#ffffff").strip()
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return b, g, r
    # fallback: blanco
    return 255, 255, 255


@dataclass
class _OverlayMessage:
    text: str
    color: tuple[int, int, int]
    duration_ms: int
    opacity: float
    created: float


# ---------------------------------------------------------------------------
# Perfiles globales
# ---------------------------------------------------------------------------

PROFILE_ACTIVE_KEY = "active_profile_id"
PROFILE_LIST_KEY = "profiles"
DEFAULT_PROFILE_ID = "ultimo_usado"
DEFAULT_PROFILE_NAME = "Ultimo usado"


def _safe_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _merge_profile_section(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, val in override.items():
        if val is not None:
            merged[key] = val
    return merged


def _profile_settings_sanitize(data: dict[str, object]) -> dict[str, object]:
    settings = copy.deepcopy(data)
    settings.pop(PROFILE_LIST_KEY, None)
    settings.pop(PROFILE_ACTIVE_KEY, None)
    settings.pop("presets", None)
    return settings


def _slug_profile_id(name: str) -> str:
    cleaned: list[str] = []
    last_us = False
    for ch in name.strip().lower():
        if ch.isascii() and ch.isalnum():
            cleaned.append(ch)
            last_us = False
            continue
        if ch in {" ", "_", "-"}:
            if cleaned and not last_us:
                cleaned.append("_")
                last_us = True
    slug = "".join(cleaned).strip("_")
    return slug or "perfil"


def _generate_profile_id(name: str, existing: set[str]) -> str:
    base = _slug_profile_id(name)
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


@dataclass
class ProfileData:
    profile_id: str
    name: str
    models: dict[str, object]
    rtsp_in: dict[str, object]
    rtsp_out: dict[str, object]
    settings: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        payload = {
            "name": self.name,
            "models": copy.deepcopy(self.models),
            "rtsp_in": copy.deepcopy(self.rtsp_in),
            "rtsp_out": copy.deepcopy(self.rtsp_out),
        }
        if self.settings is not None:
            payload["settings"] = copy.deepcopy(self.settings)
        return payload

    @classmethod
    def from_payload(
        cls,
        profile_id: str,
        payload: dict[str, object] | None,
        *,
        defaults: dict[str, dict[str, object]] | None = None,
    ) -> "ProfileData":
        raw = _safe_dict(payload)
        name = str(raw.get("name") or profile_id)
        fallback = defaults or {}
        models = _merge_profile_section(fallback.get("models", {}), _safe_dict(raw.get("models")))
        rtsp_in = _merge_profile_section(fallback.get("rtsp_in", {}), _safe_dict(raw.get("rtsp_in")))
        rtsp_out = _merge_profile_section(fallback.get("rtsp_out", {}), _safe_dict(raw.get("rtsp_out")))
        settings_raw = raw.get("settings")
        settings = _profile_settings_sanitize(settings_raw) if isinstance(settings_raw, dict) else None
        return cls(
            profile_id=profile_id,
            name=name,
            models=models,
            rtsp_in=rtsp_in,
            rtsp_out=rtsp_out,
            settings=settings,
        )

# ---------------------------------------------------------------------------

def _ffmpeg_binary(name: str) -> str:
    """Devuelve la ruta absoluta a ejecutables ffmpeg/ffplay/ffprobe del proyecto."""
    exe = f"{name}.exe" if os.name == "nt" and not name.lower().endswith(".exe") else name
    path = os.path.join(FFMPEG_BIN_DIR, exe)
    return path


@dataclass
class _InferLaunch:
    done_event: torch.cuda.Event | None
    dets: torch.Tensor | None
    masks: torch.Tensor | None
    names: dict
    frame_shape: tuple[int, int]
    cpu_buf: torch.Tensor | None = None
    gpu_buf: torch.Tensor | None = None
    pre_event: torch.cuda.Event | None = None


def _safe_letterbox(img, new_shape=640, stride=32, color=(114, 114, 114)):
    if _ultra_letterbox is not None:
        return _ultra_letterbox(img, new_shape=new_shape, stride=stride, auto=False)

    if isinstance(new_shape, (int, float)):
        new_shape = (int(new_shape), int(new_shape))
    elif isinstance(new_shape, (tuple, list)):
        if len(new_shape) == 1:
            new_shape = (int(new_shape[0]), int(new_shape[0]))
        else:
            new_shape = (int(new_shape[0]), int(new_shape[1]))
    else:
        new_shape = (int(new_shape), int(new_shape))

    shape = img.shape[:2]  # (h, w)
    if shape[0] == 0 or shape[1] == 0:
        raise ValueError("Imagen vacía en letterbox")

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    ratio = (r, r)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def _letterbox_cuda(
    frame_bgr: np.ndarray,
    target_shape: tuple[int, int],
    *,
    stride: int = 32,
    stream: torch.cuda.Stream | None = None,
    out_tensor: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible para letterbox GPU")

    if len(target_shape) != 2:
        raise ValueError("target_shape debe ser (h, w)")

    target_h = max(1, int(target_shape[0]))
    target_w = max(1, int(target_shape[1]))
    if stride and stride > 1:
        target_h = max(stride, int(round(target_h / stride) * stride))
        target_w = max(stride, int(round(target_w / stride) * stride))

    h0, w0 = frame_bgr.shape[:2]
    if h0 == 0 or w0 == 0:
        raise ValueError("Imagen vacía en letterbox CUDA")

    r = min(target_h / h0, target_w / w0)
    r = min(r, 1.0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = target_w - new_unpad[0]
    dh = target_h - new_unpad[1]
    dw /= 2
    dh /= 2

    pad_left = int(round(dw - 0.1))
    pad_right = int(round(dw + 0.1))
    pad_top = int(round(dh - 0.1))
    pad_bottom = int(round(dh + 0.1))

    desired_shape = (1, 3, target_h, target_w)
    if out_tensor is not None and (out_tensor.dtype != dtype or tuple(out_tensor.shape) != desired_shape):
        out_tensor = None

    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_ctx:
        is_contiguous = bool(getattr(frame_bgr.flags, "c_contiguous", True))
        frame_src = frame_bgr if is_contiguous else np.ascontiguousarray(frame_bgr)
        frame_tensor = torch.from_numpy(frame_src).to(device="cuda", dtype=torch.uint8, non_blocking=True)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor[:, [2, 1, 0], ...]  # BGR -> RGB
        frame_tensor = frame_tensor.to(dtype=torch.float32)

        if new_unpad != (w0, h0):
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(new_unpad[1], new_unpad[0]),
                mode="bilinear",
                align_corners=False,
            )

        if pad_left or pad_right or pad_top or pad_bottom:
            frame_tensor = F.pad(
                frame_tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                value=114.0,
            )
        frame_tensor.mul_(1.0 / 255.0)
        if dtype == torch.float16:
            frame_tensor = frame_tensor.to(dtype=torch.float16)
        else:
            frame_tensor = frame_tensor.to(dtype=torch.float32)
        frame_tensor = frame_tensor.contiguous(memory_format=torch.channels_last)

        if out_tensor is None:
            out_tensor = torch.empty(
                (1, 3, target_h, target_w),
                dtype=dtype,
                device="cuda",
                memory_format=torch.channels_last,
            )
        out_tensor.copy_(frame_tensor, non_blocking=True)

    return out_tensor, (r, r), (dw, dh)


class _PreprocessCache:
    def __init__(self):
        self.cpu_pool: dict[tuple[tuple[int, int, int, int], torch.dtype], list[torch.Tensor]] = {}
        self.gpu_pool: list[torch.Tensor] = []
        self.event_pool: list[torch.cuda.Event] = []

    def acquire_cpu(self, shape: tuple[int, int, int, int], dtype: torch.dtype = torch.float16) -> torch.Tensor:
        key = (shape, dtype)
        pool = self.cpu_pool.get(key)
        if pool:
            return pool.pop()
        tensor = torch.empty(shape, dtype=dtype).pin_memory()
        return tensor

    def release_cpu(self, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        shape = tuple(tensor.shape)
        key = (shape, tensor.dtype)
        self.cpu_pool.setdefault(key, []).append(tensor)

    def acquire_gpu(
        self,
        shape: tuple[int, int, int, int],
        *,
        dtype: torch.dtype = torch.float16,
        allow_greater: bool = False,
    ) -> torch.Tensor:
        idx: int | None = None
        if allow_greater:
            for i, tensor in enumerate(self.gpu_pool):
                th, tw = tensor.shape[2], tensor.shape[3]
                if tensor.dtype == dtype and th >= shape[2] and tw >= shape[3]:
                    idx = i
                    break
        else:
            for i, tensor in enumerate(self.gpu_pool):
                if tensor.dtype == dtype and tuple(tensor.shape) == shape:
                    idx = i
                    break
        if idx is not None:
            return self.gpu_pool.pop(idx)
        tensor = torch.empty(shape, dtype=dtype, device="cuda", memory_format=torch.channels_last)
        return tensor

    def release_gpu(self, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        self.gpu_pool.append(tensor)

    def acquire_event(self) -> torch.cuda.Event:
        if self.event_pool:
            return self.event_pool.pop()
        return torch.cuda.Event()

    def release_event(self, event: torch.cuda.Event | None) -> None:
        if event is None:
            return
        self.event_pool.append(event)


_PREPROCESS_CACHE = _PreprocessCache()


class _PreprocessGpuGuard:
    def __init__(self) -> None:
        self.enabled = True
        self.failures = 0
        self.backoff_until = 0.0

    def can_use_gpu(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if not self.enabled and time.time() >= self.backoff_until:
            self.enabled = True
        return self.enabled

    def register_failure(self, exc: Exception) -> None:
        self.failures += 1
        delay = min(60.0, 5.0 * (2 ** (self.failures - 1)))
        self.enabled = False
        self.backoff_until = time.time() + delay
        LOGGER.warning(
            "Letterbox CUDA deshabilitado durante %.1fs tras error: %s",
            delay,
            exc,
        )

    def register_success(self) -> None:
        if not self.enabled or self.failures > 0:
            LOGGER.info("Letterbox CUDA reactivado tras fallos previos.")
        self.enabled = True
        self.failures = 0
        self.backoff_until = 0.0


_PREPROC_GPU_GUARD = _PreprocessGpuGuard()


def _normalize_imgsz(imgsz) -> tuple[int, int]:
    if isinstance(imgsz, (int, float)):
        val = int(imgsz)
        return (val, val)
    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) == 1:
            val = int(imgsz[0])
            return (val, val)
        return (int(imgsz[0]), int(imgsz[1]))
    val = int(imgsz)
    return (val, val)


def _preprocess_cpu_letterbox(
    frame_bgr: np.ndarray,
    target_shape: tuple[int, int],
    *,
    stride: int = 32,
    stream: torch.cuda.Stream | None = None,
    reuse_gpu: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float16,
):
    im, ratio, (dw, dh) = _safe_letterbox(frame_bgr, new_shape=target_shape, stride=stride)
    im = im[:, :, ::-1].copy()
    h, w = im.shape[:2]
    shape4d = (1, 3, h, w)

    gpu_tensor_base = reuse_gpu
    if gpu_tensor_base is not None and gpu_tensor_base.dtype != dtype:
        _PREPROCESS_CACHE.release_gpu(gpu_tensor_base)
        gpu_tensor_base = None
    if (
        gpu_tensor_base is None
        or gpu_tensor_base.shape[2] < shape4d[2]
        or gpu_tensor_base.shape[3] < shape4d[3]
    ):
        if gpu_tensor_base is not None:
            _PREPROCESS_CACHE.release_gpu(gpu_tensor_base)
        gpu_tensor_base = _PREPROCESS_CACHE.acquire_gpu(shape4d, allow_greater=True, dtype=dtype)
    gpu_tensor = gpu_tensor_base[:, :, : shape4d[2], : shape4d[3]]

    torch_stream = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with torch_stream:
        frame_tensor = torch.from_numpy(im).to(
            device=gpu_tensor.device,
            dtype=dtype,
            non_blocking=True,
        )
        frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()
        frame_tensor.mul_(1.0 / 255.0)
        gpu_tensor[0].copy_(frame_tensor, non_blocking=True)

    pre_event = None
    if stream is not None:
        pre_event = _PREPROCESS_CACHE.acquire_event()
        pre_event.record(stream)

    return gpu_tensor, ratio, (dw, dh), None, gpu_tensor_base, pre_event



def _guess_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def _add_low_latency_flags(url: str, disabled: set[str] | None = None) -> str:
    """Anexa parámetros de baja latencia a una URL RTSP."""
    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url

    disabled_flags: set[str] = set(disabled or [])

    try:
        parsed = urlparse(url)
        host = parsed.hostname if parsed else None
    except Exception:
        host = None

    if host and host not in {"127.0.0.1", "localhost"}:
        local_ip = _guess_local_ip()
        if local_ip and host not in {local_ip, local_ip.split("%", 1)[0]}:
            disabled_flags.add("rtsp_transport")

    # flags inspirados en la integración previa RTSP con OpenCV
    flags = [
        ("rtsp_transport", "udp"),
        ("fflags", "nobuffer"),
        ("flags", "low_delay"),
        ("max_delay", "0"),
        ("analyzeduration", "0"),
        ("probesize", "32"),
    ]
    query_parts = [f"{k}={v}" for k, v in flags if k not in disabled_flags]
    if not query_parts:
        return url
    separator = "&" if "?" in url else "?"
    return url + separator + "&".join(query_parts)


def _apply_rtsp_transport(url: str, transport: str) -> str:
    """Compatibilidad retro: delega en _ensure_rtsp_transport_param."""
    return _ensure_rtsp_transport_param(url, transport)


def _ensure_rtsp_transport_param(url: str, transport: str) -> str:
    """Asegura que la URL RTSP incluya el parámetro rtsp_transport con el valor indicado."""
    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url

    parsed = urlparse(url)
    if parsed.scheme.lower() != "rtsp":
        return url

    normalized = (transport or "").strip().lower()
    if normalized not in {"tcp", "udp"}:
        normalized = "tcp"

    params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "rtsp_transport"]
    params.append(("rtsp_transport", normalized))
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _preprocess_to_cuda(
    frame_bgr: np.ndarray,
    imgsz,
    *,
    stride: int = 32,
    stream_preproc: torch.cuda.Stream | None = None,
    reuse_gpu: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float16,
):
    """Letterbox en GPU si es posible (con fallback CPU) y devuelve tensor en CUDA."""

    target_shape = _normalize_imgsz(imgsz)
    target_shape = (
        max(1, target_shape[0]),
        max(1, target_shape[1]),
    )
    if stride and stride > 1:
        target_shape = (
            max(stride, int(round(target_shape[0] / stride) * stride)),
            max(stride, int(round(target_shape[1] / stride) * stride)),
        )

    shape4d = (1, 3, target_shape[0], target_shape[1])
    gpu_tensor_base = reuse_gpu
    if gpu_tensor_base is not None and gpu_tensor_base.dtype != dtype:
        _PREPROCESS_CACHE.release_gpu(gpu_tensor_base)
        gpu_tensor_base = None
    if (
        gpu_tensor_base is None
        or gpu_tensor_base.shape[2] < shape4d[2]
        or gpu_tensor_base.shape[3] < shape4d[3]
    ):
        if gpu_tensor_base is not None:
            _PREPROCESS_CACHE.release_gpu(gpu_tensor_base)
        gpu_tensor_base = _PREPROCESS_CACHE.acquire_gpu(shape4d, allow_greater=True, dtype=dtype)
    gpu_tensor_view = gpu_tensor_base[:, :, : shape4d[2], : shape4d[3]]

    # Intento GPU directo
    if _PREPROC_GPU_GUARD.can_use_gpu():
        try:
            tensor, ratio, (dw, dh) = _letterbox_cuda(
                frame_bgr,
                target_shape,
                stride=stride,
                stream=stream_preproc,
                out_tensor=gpu_tensor_view,
                dtype=dtype,
            )
            pre_event = None
            if stream_preproc is not None:
                pre_event = _PREPROCESS_CACHE.acquire_event()
                pre_event.record(stream_preproc)
            _PREPROC_GPU_GUARD.register_success()
            return tensor, ratio, (dw, dh), None, gpu_tensor_base, pre_event
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            _PREPROC_GPU_GUARD.register_failure(exc)
        except RuntimeError as exc:
            _PREPROC_GPU_GUARD.register_failure(exc)
        except Exception as exc:
            self.config = {}
            LOGGER.exception("Letterbox CUDA falló; usando ruta CPU.")
            _PREPROC_GPU_GUARD.register_failure(exc)

    # Fallback CPU
    tensor, ratio, (dw, dh), cpu_buf, gpu_buf, pre_event = _preprocess_cpu_letterbox(
        frame_bgr,
        target_shape,
        stride=stride,
        stream=stream_preproc,
        reuse_gpu=gpu_tensor_base,
        dtype=dtype,
    )
    return tensor, ratio, (dw, dh), cpu_buf, gpu_buf, pre_event


@dataclass(slots=True)
class _BoxesAdapter:
    xyxy: np.ndarray
    conf: np.ndarray | None
    cls: np.ndarray

    def __len__(self) -> int:
        return 0 if self.xyxy is None else int(self.xyxy.shape[0])


@dataclass(slots=True)
class _MasksAdapter:
    data_np: np.ndarray | None = None
    xy: Any | None = None
    area_px: np.ndarray | None = None


class _ResultAdapter:
    """Contenedor ligero con acceso en NumPy."""

    __slots__ = ("_boxes", "_names", "_masks", "_xyxy", "_conf", "_cls", "_masks_np", "_masks_area", "__weakref__")

    def __init__(
        self,
        boxes: _BoxesAdapter,
        names: dict,
        masks: _MasksAdapter | None = None,
    ) -> None:
        self._boxes = boxes
        self._names = names
        self._masks = masks
        self._xyxy = boxes.xyxy
        self._conf = boxes.conf
        self._cls = boxes.cls
        self._masks_np = masks.data_np if masks is not None else None
        self._masks_area = masks.area_px if masks is not None else None

    @property
    def names(self) -> dict:
        return self._names

    def xyxy_np(self) -> np.ndarray:
        return self._xyxy

    def conf_np(self) -> np.ndarray | None:
        return self._conf

    def cls_np(self) -> np.ndarray:
        return self._cls

    def masks_np(self) -> np.ndarray | None:
        return self._masks_np

    def has_masks(self) -> bool:
        return self._masks_np is not None or (self._masks is not None and self._masks.xy is not None)

    def masks_xy(self):
        return self._masks.xy if self._masks is not None else None

    def masks_area_px(self) -> np.ndarray | None:
        return self._masks_area


@dataclass
class _FramePacket:
    frame: np.ndarray
    job1: _InferLaunch | None
    job2: _InferLaunch | None


class SnapshotAggregator:
    def __init__(
        self,
        path: str,
        *,
        short_window_sec: float = SNAPSHOT_SHORT_WINDOW_SEC,
        long_window_sec: float = SNAPSHOT_LONG_WINDOW_SEC,
        chain_gap_sec: float = SNAPSHOT_CHAIN_GAP_SEC,
        low_conf_threshold: float = SNAPSHOT_LOW_CONF_THRESHOLD,
        major_area_threshold: float = SNAPSHOT_MAJOR_AREA_THRESHOLD,
        recent_events_max: int = SNAPSHOT_RECENT_EVENTS_MAX,
        line_speed_mpm: float = 0.0,
        write_interval_ms: int = SNAPSHOT_DEFAULT_WRITE_INTERVAL_MS,
        clean_every_sec: float = SNAPSHOT_DEFAULT_CLEAN_INTERVAL_SEC,
    ) -> None:
        self.path = path
        self.tmp_path = f"{path}.tmp"
        self.short_window_sec = float(short_window_sec)
        self.long_window_sec = float(long_window_sec)
        self.chain_gap_sec = float(chain_gap_sec)
        self.low_conf_threshold = float(max(0.0, low_conf_threshold))
        self.major_area_threshold = float(max(0.0, major_area_threshold))
        self.line_speed_mpm = float(line_speed_mpm)
        self.write_interval_ms = max(SNAPSHOT_MIN_WRITE_INTERVAL_MS, int(write_interval_ms))
        self.clean_every_sec = max(0.0, float(clean_every_sec))
        self._short_entries: deque[dict] = deque()
        self._long_entries: deque[dict] = deque()
        self._recent_events: deque[dict] = deque(maxlen=int(max(1, recent_events_max)))
        self._lock = threading.Lock()
        self._dirty = False
        self._fps_ema: float | None = None
        self._frame_counter = 0
        self._last_major_ts: float | None = None
        self._major_chain_start: float | None = None
        self._last_flush_payload: str | None = None
        self._last_flush_wall: float = 0.0
        self._last_clean_wall: float = time.time()
        self._backoff_until: float = 0.0
        self._backoff_interval: float = 0.0
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def update(
        self,
        *,
        frame_ts: float,
        detections: list[dict],
        avg_fps: float | None = None,
        inst_fps: float | None = None,
    ) -> None:
        with self._lock:
            self._frame_counter += 1
            for det in detections:
                entry = self._build_entry(frame_ts, det)
                self._short_entries.append(entry)
                self._long_entries.append(entry)
                if entry["is_major"]:
                    self._register_major(frame_ts, entry)
            self._prune(frame_ts)
            self._update_fps(avg_fps, inst_fps)
            self._dirty = True

    def clear_history(self) -> None:
        with self._lock:
            self._short_entries.clear()
            self._long_entries.clear()
            self._recent_events.clear()
            self._last_major_ts = None
            self._major_chain_start = None
            self._dirty = True

    def configure_timing(self, *, write_interval_ms: int | None = None, clean_every_sec: float | None = None) -> None:
        with self._lock:
            if write_interval_ms is not None:
                self.write_interval_ms = max(SNAPSHOT_MIN_WRITE_INTERVAL_MS, int(write_interval_ms))
            if clean_every_sec is not None:
                self.clean_every_sec = max(0.0, float(clean_every_sec))

    def flush(self, *, force: bool = False) -> bool:
        now_wall = time.time()
        with self._lock:
            now_wall = time.time()
            if not force and now_wall < self._backoff_until:
                return False

            if self.clean_every_sec > 0 and (now_wall - self._last_clean_wall) >= self.clean_every_sec:
                self._short_entries.clear()
                self._long_entries.clear()
                self._recent_events.clear()
                self._last_major_ts = None
                self._major_chain_start = None
                self._last_clean_wall = now_wall
                self._dirty = True

            if not (force or self._dirty):
                return False
            payload_dict = self._build_snapshot(now_wall)
            payload_str = json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":"))
            if not force and payload_str == self._last_flush_payload:
                self._dirty = False
                self._backoff_interval = 0.0
                return False
            try:
                with open(self.tmp_path, "w", encoding="utf-8") as fh:
                    fh.write(payload_str)
                os.replace(self.tmp_path, self.path)
                self._last_flush_payload = payload_str
                self._last_flush_wall = now_wall
                self._dirty = False
                self._backoff_interval = 0.0
                self._backoff_until = 0.0
                return True
            except OSError as exc:
                is_permission = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5
                if is_permission:
                    # Intentar sobrescribir directamente el archivo (no atómico) para permitir lectura concurrente.
                    try:
                        with open(self.path, "w", encoding="utf-8") as fh:
                            fh.write(payload_str)
                    except OSError as exc_direct:
                        if self._backoff_interval <= 0.0:
                            self._backoff_interval = 2.0
                        else:
                            self._backoff_interval = min(self._backoff_interval * 2.0, 30.0)
                        self._backoff_until = now_wall + self._backoff_interval
                        LOGGER.warning(
                            "Snapshot JSON bloqueado (posible fichero abierto). Reintentando en %.1fs (%s)",
                            self._backoff_interval,
                            exc_direct,
                        )
                        return False
                    else:
                        LOGGER.warning(
                            "Snapshot JSON sobrescrito sin rename por bloqueo externo (%s)",
                            exc,
                        )
                        self._last_flush_payload = payload_str
                        self._last_flush_wall = now_wall
                        self._dirty = False
                        self._backoff_interval = 0.0
                        self._backoff_until = 0.0
                        return True
                LOGGER.error("Snapshot JSON: no se pudo escribir %s (%s)", self.path, exc)
                return False

    def build_snapshot_payload(self) -> dict:
        now_wall = time.time()
        with self._lock:
            self._prune(now_wall)
            payload = self._build_snapshot(now_wall)
        return copy.deepcopy(payload)

    def _build_entry(self, frame_ts: float, det: dict) -> dict:
        cls_name = str(det.get("cls", "desconocida"))
        area = float(max(0.0, det.get("area", 0.0)))
        rel_area = float(max(0.0, det.get("relative_area", 0.0)))
        conf = float(det.get("conf", 0.0))
        center = tuple(det.get("center_norm", (None, None)))
        model_tag = det.get("model")
        sector = det.get("sector")  # NUEVO: capturar sector asignado por Sectorizador
        is_major = (cls_name in SNAPSHOT_MAJOR_CLASSES) or (area >= self.major_area_threshold)
        if not is_major and rel_area:
            frame_area = float(det.get("frame_area", 0.0))
            if frame_area > 0:
                pixels_threshold = self.major_area_threshold / frame_area
                is_major = rel_area >= pixels_threshold
        area_enabled = bool(det.get("area_enabled", True))
        entry = {
            "ts": float(frame_ts),
            "cls": cls_name,
            "area": area,
            "relative_area": rel_area,
            "conf": conf,
            "center_norm": center,
            "model": model_tag,
            "sector": sector,  # NUEVO: incluir sector en la entrada
            "is_major": bool(is_major),
            "is_low_conf": bool(conf < self.low_conf_threshold),
            "area_enabled": area_enabled,
            "raw": det,
        }
        area_cm2 = det.get("area_cm2")
        if area_cm2 is not None:
            try:
                entry["area_cm2"] = float(max(0.0, area_cm2))
            except (TypeError, ValueError):
                entry["area_cm2"] = None
        if bool(det.get("is_critical")):
            entry["is_major"] = True
        return entry

    def _register_major(self, frame_ts: float, entry: dict) -> None:
        prev_major_ts = self._last_major_ts
        self._last_major_ts = float(frame_ts)
        if prev_major_ts is None or (frame_ts - prev_major_ts) > self.chain_gap_sec:
            self._major_chain_start = float(frame_ts)
        raw = entry.get("raw", {})
        center = raw.get("center_norm") or entry.get("center_norm")
        event_payload = {
            "ts": frame_ts,
            "cls": entry.get("cls"),
            "area": entry.get("area"),
            "relative_area": entry.get("relative_area"),
            "conf": entry.get("conf"),
            "model": entry.get("model"),
            "center_norm": center,
            "area_cm2": entry.get("area_cm2"),
        }
        self._recent_events.append(event_payload)

    def _prune(self, frame_ts: float) -> None:
        short_cutoff = frame_ts - self.short_window_sec
        while self._short_entries and self._short_entries[0]["ts"] < short_cutoff:
            self._short_entries.popleft()
        long_cutoff = frame_ts - self.long_window_sec
        while self._long_entries and self._long_entries[0]["ts"] < long_cutoff:
            self._long_entries.popleft()

    def _update_fps(self, avg_fps: float | None, inst_fps: float | None) -> None:
        value = None
        if inst_fps is not None and inst_fps > 0:
            value = inst_fps
        elif avg_fps is not None and avg_fps > 0:
            value = avg_fps
        if value is None:
            return
        if self._fps_ema is None:
            self._fps_ema = float(value)
            return
        self._fps_ema = 0.7 * self._fps_ema + 0.3 * float(value)

    def _build_snapshot(self, now_wall: float) -> dict:
        short_stats = self._summarize_window(self._short_entries, self.short_window_sec)
        long_stats = self._summarize_window(self._long_entries, self.long_window_sec)
        trend = self._compute_trend(short_stats, long_stats)
        stability = self._build_stability(now_wall)
        recent = self._format_recent_events()
        meta = {
            "producer": "detect_manchas_gui_rtsp",
            "version": "1.0",
            "timestamp": self._iso8601(now_wall),
            "frames_total": int(self._frame_counter),
            "fps_estimate": round(self._fps_ema, 2) if self._fps_ema is not None else None,
            "line_speed_mpm": self.line_speed_mpm,
        }
        payload = {
            "meta": meta,
            "window_short": {**short_stats, "trend_vs_long": trend},
            "window_long": long_stats,
            "stats_short": short_stats.get("classes", {}),
            "stats_long": long_stats.get("classes", {}),
            "stability_info": stability,
            "recent_critical_events": recent,
        }
        return payload

    def _summarize_window(self, entries: Iterable[dict], duration: float) -> dict:
        classes: dict[str, dict[str, float]] = {}
        # NUEVO: estadísticas agrupadas por sector -> clase
        sectors: dict[int, dict[str, dict[str, float]]] = {}
        low_conf = 0
        major_count = 0
        for entry in entries:
            cls_name = entry["cls"]
            sector = entry.get("sector")
            stats = classes.setdefault(
                cls_name,
                {
                    "count": 0,
                    "area_sum": 0.0,
                    "area_max": 0.0,
                    "area_count": 0,
                    "area_cm2_sum": 0.0,
                    "area_cm2_max": 0.0,
                    "area_cm2_count": 0,
                    "conf_sum": 0.0,
                },
            )
            stats["count"] += 1
            stats["conf_sum"] += entry["conf"]
            area_enabled = entry.get("area_enabled", True)
            area_cm2_val = None
            if area_enabled:
                stats["area_sum"] += entry["area"]
                stats["area_max"] = max(stats["area_max"], entry["area"])
                stats["area_count"] += 1
                area_cm2_val = entry.get("area_cm2")
                if isinstance(area_cm2_val, (int, float)) and area_cm2_val > 0:
                    stats["area_cm2_sum"] += float(area_cm2_val)
                    stats["area_cm2_max"] = max(stats["area_cm2_max"], float(area_cm2_val))
                    stats["area_cm2_count"] += 1
            if entry.get("is_low_conf"):
                low_conf += 1
            if entry.get("is_major"):
                major_count += 1
            # NUEVO: agrupar por sector si está disponible
            if sector is not None:
                try:
                    sector_id = int(sector)
                except (TypeError, ValueError):
                    sector_id = None
                if sector_id is not None:
                    if sector_id not in sectors:
                        sectors[sector_id] = {}
                    sector_class_stats = sectors[sector_id].setdefault(
                        cls_name,
                        {
                            "count": 0,
                            "area_sum": 0.0,
                            "area_count": 0,
                            "area_cm2_sum": 0.0,
                            "area_cm2_max": 0.0,
                            "area_cm2_count": 0,
                            "conf_sum": 0.0,
                        },
                    )
                    sector_class_stats["count"] += 1
                    sector_class_stats["conf_sum"] += entry["conf"]
                    if area_enabled:
                        sector_class_stats["area_sum"] += entry["area"]
                        sector_class_stats["area_count"] += 1
                        if isinstance(area_cm2_val, (int, float)) and area_cm2_val > 0:
                            sector_class_stats["area_cm2_sum"] += float(area_cm2_val)
                            sector_class_stats["area_cm2_max"] = max(
                                sector_class_stats["area_cm2_max"],
                                float(area_cm2_val),
                            )
                            sector_class_stats["area_cm2_count"] += 1
        formatted_classes: dict[str, dict[str, float]] = {}
        for name, stats in classes.items():
            cnt = max(1, stats["count"])
            formatted_classes[name] = {
                "count": int(stats["count"]),
                "conf_avg": stats["conf_sum"] / cnt,
            }
            area_cnt = stats.get("area_count", 0)
            if area_cnt > 0:
                formatted_classes[name]["area_avg"] = stats["area_sum"] / area_cnt
                formatted_classes[name]["area_max"] = stats["area_max"]
            else:
                formatted_classes[name]["area_avg"] = None
                formatted_classes[name]["area_max"] = None
            cm_cnt = stats.get("area_cm2_count", 0)
            if cm_cnt > 0:
                formatted_classes[name]["area_avg_cm2"] = stats["area_cm2_sum"] / cm_cnt
                formatted_classes[name]["area_max_cm2"] = stats["area_cm2_max"]
        # NUEVO: formatear estadísticas por sector
        formatted_sectors: dict[int, dict[str, dict[str, float]]] = {}
        for sector_id, sector_classes in sectors.items():
            formatted_sectors[sector_id] = {}
            for cls_name, s_stats in sector_classes.items():
                s_cnt = max(1, s_stats["count"])
                formatted_sectors[sector_id][cls_name] = {
                    "count": int(s_stats["count"]),
                    "conf_avg": s_stats["conf_sum"] / s_cnt,
                }
                s_area_cnt = s_stats.get("area_count", 0)
                if s_area_cnt > 0:
                    formatted_sectors[sector_id][cls_name]["area_avg"] = s_stats["area_sum"] / s_area_cnt
                else:
                    formatted_sectors[sector_id][cls_name]["area_avg"] = None
                cm_cnt = s_stats.get("area_cm2_count", 0)
                if cm_cnt > 0:
                    formatted_sectors[sector_id][cls_name]["area_avg_cm2"] = s_stats["area_cm2_sum"] / cm_cnt
                    formatted_sectors[sector_id][cls_name]["area_max_cm2"] = s_stats["area_cm2_max"]
        total = sum(v["count"] for v in formatted_classes.values())
        defect_rate = total / duration if duration > 0 else 0.0
        low_conf_ratio = low_conf / total if total > 0 else 0.0
        return {
            "duration_sec": duration,
            "total_detections": total,
            "major_events": major_count,
            "defect_rate": defect_rate,
            "low_conf_ratio": low_conf_ratio,
            "classes": formatted_classes,
            "sectors": formatted_sectors,  # NUEVO: incluir estadísticas por sector
        }

    def _compute_trend(self, short_stats: dict, long_stats: dict) -> str:
        short_rate = short_stats.get("defect_rate", 0.0)
        long_rate = long_stats.get("defect_rate", 0.0)
        if long_rate <= 1e-6:
            if short_rate <= 1e-6:
                return "stable"
            return "spike"
        ratio = short_rate / max(long_rate, 1e-6)
        if ratio >= 1.5:
            return "rising"
        if ratio <= 0.5:
            return "falling"
        return "stable"

    def _build_stability(self, now_wall: float) -> dict:
        seconds_since_last_major = None
        if self._last_major_ts is not None:
            seconds_since_last_major = max(0.0, now_wall - self._last_major_ts)
        chain_duration = 0.0
        if self._major_chain_start is not None and self._last_major_ts is not None:
            chain_duration = max(0.0, self._last_major_ts - self._major_chain_start)
        return {
            "seconds_since_last_major": seconds_since_last_major,
            "continuous_major_duration": chain_duration,
            "last_major_class": self._recent_events[-1].get("cls") if self._recent_events else None,
        }

    def _format_recent_events(self) -> list[dict]:
        events: list[dict] = []
        for event in reversed(self._recent_events):
            events.append(
                {
                    "ts": self._iso8601(event["ts"]),
                    "cls": event.get("cls"),
                    "area": event.get("area"),
                    "relative_area": event.get("relative_area"),
                    "confidence": event.get("conf"),
                    "model": event.get("model"),
                    "center_norm": event.get("center_norm"),
                    "area_cm2": event.get("area_cm2"),
                }
            )
        return events

    @staticmethod
    def _iso8601(ts: float) -> str:
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat(timespec="milliseconds")


HEARTBEAT_DETECTOR_PORT = 9101
HEARTBEAT_VIEWER_PORT = 9102
HEARTBEAT_INTERVAL_SEC = 5.0
_HEARTBEAT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _HeartbeatBridge:
    def __init__(
        self,
        name: str,
        listen_port: int,
        tk_root: tk.Tk,
        on_message: Callable[[dict, tuple[str, int]], None],
        default_targets: Optional[list[tuple[str, int]]] = None,
    ) -> None:
        self.name = name
        self.listen_port = listen_port
        self.root = tk_root
        self._on_message = on_message
        self._targets_lock = threading.Lock()
        self._targets: set[tuple[str, int]] = set(default_targets or [])
        self._running = False
        self._recv_sock: socket.socket | None = None
        self._listen_thread: threading.Thread | None = None
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self) -> bool:
        if self._running:
            return True
        try:
            self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._recv_sock.bind(("", self.listen_port))
            self._recv_sock.settimeout(0.5)
        except OSError as exc:
            LOGGER.warning("Heartbeat %s: no se pudo abrir el puerto %s (%s)", self.name, self.listen_port, exc)
            self._recv_sock = None
            return False
        self._running = True
        self._listen_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._listen_thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._recv_sock is not None:
            try:
                self._recv_sock.close()
            except OSError:
                pass
            self._recv_sock = None
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1.0)
        self._listen_thread = None
        try:
            self._send_sock.close()
        except OSError:
            pass

    def add_target(self, target: tuple[str, int]) -> None:
        host, port = target
        if not host:
            return
        if not isinstance(port, int) or port <= 0 or port > 65535:
            return
        with self._targets_lock:
            self._targets.add((host, port))

    def send_message(self, text: str) -> None:
        payload = {
            "sender": self.name,
            "text": text,
            "ts": time.time(),
            "reply_port": self.listen_port,
        }
        encoded = json.dumps(payload).encode("utf-8")
        with self._targets_lock:
            targets = list(self._targets)
        for host, port in targets:
            try:
                self._send_sock.sendto(encoded, (host, port))
            except OSError:
                continue

    def _recv_loop(self) -> None:
        while self._running and self._recv_sock is not None:
            try:
                data, addr = self._recv_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            reply_port = payload.get("reply_port")
            if isinstance(reply_port, int) and 0 < reply_port <= 65535:
                self.add_target((addr[0], reply_port))
            else:
                self.add_target((addr[0], addr[1]))

            try:
                self.root.after(0, lambda p=payload, a=addr: self._on_message(p, a))
            except Exception:
                pass


def get_cuda_device_or_die() -> str:
    """Verifica que podemos ejecutar kernels en GPU y devuelve 'cuda:0'."""
    if not torch.cuda.is_available():
        raise SystemExit("No hay GPU CUDA disponible. Revisa drivers NVIDIA y PyTorch cu128.")
    try:
        torch.cuda.set_device(0)
        x = torch.randn(256, 256, device="cuda")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
        return "cuda:0"
    except Exception as e:
        msg = str(e).lower()
        if "no kernel image" in msg or "sm_" in msg:
            raise SystemExit(
                "Kernels CUDA incompatibles. Instala PyTorch cu128 y driver R570+ y reinicia:\n"
                "  pip uninstall -y torch torchvision torchaudio\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
            ) from e
        raise SystemExit(f"CUDA disponible, pero falló la prueba de kernels: {e}") from e


class FFmpegWriter:
    def __init__(self, out_path: str, w: int, h: int, fps: float):
        self.out_path = out_path
        ffmpeg_exe = _ffmpeg_binary("ffmpeg")
        # Usar libx264 con preset ultrafast para mínimo impacto en CPU
        # crf 23 es calidad default. ultrafast hace que la compresión sea muy rápida.
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{int(w)}x{int(h)}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
            
        try:
            self.proc = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
        except Exception as e:
            print(f"[FFmpegWriter] Error iniciando ffmpeg: {e}")
            self.proc = None

    def write(self, frame: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            # Escribir bytes crudos al pipe
            self.proc.stdin.write(frame.tobytes())
        except Exception:
            # Si falla (pipe roto), cerrar
            self.release()

    def release(self):
        if self.proc is not None:
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            self.proc.wait()
            self.proc = None
    
    def isOpened(self):
        return self.proc is not None



def _resolve_ffmpeg_path() -> str | None:
    """Intenta encontrar ffmpeg.exe de forma robusta."""
    # 1. Ruta configurada
    candidates = []
    
    # Ruta calculada estándar
    try:
        std_path = _ffmpeg_binary("ffmpeg")
        candidates.append(os.path.abspath(std_path))
    except Exception:
        pass

    # 2. Búsqueda desde la raíz del bundle (_MEIPASS o dir del exe) sin ".."
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")
        # app/ -> ../bin/ffmpeg/bin
        rel_path = os.path.join(base_dir, "bin", "ffmpeg", "bin", "ffmpeg.exe")
        candidates.append(rel_path)
    except Exception:
        pass
        
    # 3. CWD
    candidates.append(os.path.abspath(os.path.join("bin", "ffmpeg", "bin", "ffmpeg.exe")))

    print(f"[FFmpeg Resolve] Buscando en candidatos:", flush=True)
    for path in candidates:
        exists = os.path.exists(path)
        print(f"  - '{path}' -> Exite? {exists}", flush=True)
        if exists:
            return path
            
    return None

class AsyncFFmpegWriter:
    def __init__(self, out_path: str, w: int, h: int, fps: float, ffmpeg_path: str):
        self.out_path = out_path
        self.w = int(w)
        self.h = int(h)
        self.fps = fps
        self.queue = queue.Queue(maxsize=30)
        self.running = True
        self.ffmpeg_path = ffmpeg_path
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()
        self.dropped_frames = 0
        print(f"[AsyncFFmpegWriter] Iniciado. Path: {self.out_path}, NVENC intentado.", flush=True)

    def _writer_thread(self):
        # ... (LÃ³gica idÃ©ntica, pero usando self.ffmpeg_path)
        cmd_nvenc = [
            self.ffmpeg_path, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.w}x{self.h}", "-pix_fmt", "bgr24", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "p1", "-rc", "constqp", "-qp", "28",
            "-pix_fmt", "yuv420p",
            self.out_path
        ]
        
        cmd_cpu = [
            self.ffmpeg_path, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.w}x{self.h}", "-pix_fmt", "bgr24", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            self.out_path
        ]

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        
        proc = None
        try:
             proc = subprocess.Popen(
                cmd_nvenc, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
             # Esperar un poco para ver si crashea
             time.sleep(0.1)
             if proc.poll() is not None:
                 proc = None
        except Exception:
            proc = None

        if proc is None:
             print("[AsyncFFmpegWriter] Fallback a CPU (libx264)", flush=True)
             try:
                 proc = subprocess.Popen(
                    cmd_cpu, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags
                )
             except Exception as e:
                 print(f"[AsyncFFmpegWriter] Error fatal iniciando ffmpeg: {e}", flush=True)
                 return

        self.proc = proc
        
        while self.running:
            try:
                frame = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if frame is None:
                break
                
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except Exception:
                    break
        
        if self.proc:
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            self.proc.wait()
            self.proc = None

    def write(self, frame: np.ndarray):
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1
            if self.dropped_frames % 50 == 0:
                print(f"[AsyncFFmpegWriter] Drop frame {self.dropped_frames}", flush=True)

    def release(self):
        self.running = False
        try:
            self.queue.put(None)
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
    
    def isOpened(self):
        return self.running


def _log_debug(msg):
    try:
        abs_log = "debug_init.txt"
        with open(abs_log, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().time()}] {msg}\n")
    except:
        pass

def build_writer(out_path: str, w: int, h: int, fps: float) -> Any:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not fps or fps <= 0:
        fps = 25.0
    
    # _log_debug(f"build_writer llamado. Out: {out_path}, {w}x{h} @ {fps}")
    
    # 1. Intentar resolver ffmpeg
    ffmpeg_exe = _resolve_ffmpeg_path()
    if ffmpeg_exe:
        try:
            # _log_debug(f"Usando AsyncFFmpegWriter con: {ffmpeg_exe}")
            print(f"[build_writer] Usando AsyncFFmpegWriter con: {ffmpeg_exe}", flush=True)
            return AsyncFFmpegWriter(out_path, w, h, fps, ffmpeg_exe)
        except Exception as e:
            # _log_debug(f"Error instanciando AsyncFFmpegWriter: {e}")
            print(f"[build_writer] Error instanciando AsyncFFmpegWriter: {e}", flush=True)
    else:
        # _log_debug("NO SE ENCONTRO FFMPEG.EXE - Usando fallback lento.")
        print("[build_writer] NO SE ENCONTRO FFMPEG.EXE - Usando fallback lento.", flush=True)

    # print("[build_writer] FORZANDO FALLBACK a cv2.VideoWriter (DEBUG)", flush=True)
    # 2. Fallback a OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, float(fps), (int(w), int(h)))


class DetectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Deteccion de Manchas (GPU)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Estado de ejecuciÃ³n
        self.running = False
        self.paused = False
        self.worker: threading.Thread | None = None
        self.frame_lock = threading.Lock()
        self.last_frame_bgr: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None

        # Recursos de vÃ­deo/modelo
        self.model = None
        self.model2 = None
        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.device = None
        self.using_trt = False
        self.using_trt2 = False
        # Prefetch de frames para mejorar throughput
        self.frame_q: queue.Queue | None = None
        self.read_thread: threading.Thread | None = None
        self.read_stop = False
        self.stream1 = None
        self.stream2 = None
        self.stream_preproc = None
        self.net1 = None
        self.net2 = None
        # Cola de dibujo/escritura
        self.draw_queue: queue.Queue | None = None
        self.draw_thread: threading.Thread | None = None
        self.draw_stop = False
        self._draw_perf_total = 0.0
        self._draw_perf_calls = 0
        self._draw_perf_lock = threading.Lock()
        self._save_after_id: str | None = None
        self._loading_config = False
        self._var_traces: list[tuple[tk.Variable, str]] = []
        self._conf_value_label: ttk.Label | None = None
        self._iou_value_label: ttk.Label | None = None
        self._file_picker_frame: ttk.Frame | None = None
        self._rtsp_summary_frame: ttk.Frame | None = None
        self._rtsp_preview_label: ttk.Label | None = None
        self._current_source_kind: str = "Archivo"
        self._current_input_source: str | None = None
        self._current_input_url: str | None = None
        self._rtsp_manual_at_launch: bool = False
        self._local_ip = _guess_local_ip()
        self._default_rtsp_url = self._build_default_rtsp_url()
        self.rtsp_out_enable = tk.BooleanVar(value=False)
        self.rtsp_out_url = tk.StringVar(value=self._default_rtsp_url)
        self.rtsp_out_codec = tk.StringVar(value="Auto (NVENC)")
        self.rtsp_out_transport = tk.StringVar(value="TCP")
        self._out_spec: tuple[int, int, float] | None = None
        self._ffmpeg: subprocess.Popen | None = None
        # Contadores de depuraciÃ³n (solo primeras iteraciones)
        self._debug_launch_samples = 0
        self._debug_result_samples = 0
        self._debug_log_stride = 60
        self._draw_stride = 2
        self._draw_stride_counter = 0
        self._rtsp_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._rtsp_thread: threading.Thread | None = None
        self._rtsp_fail_streak = 0
        self._rtsp_status = tk.StringVar(value="Detenido")
        self._rtsp_indicator_canvas: tk.Canvas | None = None
        self._rtsp_indicator_circle: int | None = None
        self.video_label: tk.Label | None = None
        self._rtsp_idle_counter = 0
        self._rtsp_frames_sent = 0
        self._rtsp_last_stats = time.monotonic()
        self._rtsp_state = "detenido"
        self._rtsp_state_reason = ""
        self._rtsp_last_state_change = time.monotonic()
        self._rtsp_last_state_change_wall = time.time()
        self._rtsp_last_frame_ts = 0.0
        self._rtsp_last_frame_wall = 0.0
        self._rtsp_fail_notified = False
        self._win_prev_priority: int | None = None
        self._win_priority_applied = False
        self._power_throttling_disabled = False
        self._timer_resolution_active = False
        self._mmcss_local = threading.local()
        self._result_area_cache: dict[int, tuple[weakref.ref, dict[str, object]]] = {}

        self._plc_service: SendToPLCService | None = None
        self._plc_window = None
        self._plc_push_queue: queue.Queue | None = None
        self._plc_push_thread: threading.Thread | None = None
        self._plc_push_stop = threading.Event()
        self._model_names_cache: dict[str, tuple[float, tuple[str, ...]]] = {}
        self._overlay_messages: list[_OverlayMessage] = []
        self._events_poll_job: str | None = None

        # CalibraciÃ³n de conversiÃ³n pÃ­xeles->cm
        self.calibration: dict[str, float] | None = None
        self._calibration_valid = False
        self._calibration_missing_logged = False
        self._calibration_error_logged = False
        self._calibration_debug_logged = False
        self._raw_calibration_config: dict[str, Any] | None = dict(DEFAULT_CALIBRATION)

        # Etiquetas de Ã¡rea (pÃ­xeles/cmÂ²)
        default_area_display = AREA_MODE_DISPLAY[AREA_MODE_CM2]
        self.area_label_mode_display = tk.StringVar(value=default_area_display)
        self._area_label_mode_flag = AREA_MODE_CM2
        self._area_any_enabled_flag = False

        self._hb_local_box: tk.Text | None = None
        self._hb_remote_box: tk.Text | None = None
        self._last_cycle_end_ts: float = 0.0  # PerfTrace V2
        self._hb_bridge: _HeartbeatBridge | None = None
        self._hb_cycle = cycle(_HEARTBEAT_ALPHABET)
        self._hb_job: str | None = None

        # Perstencia de configuraciÃ³n centralizada
        self.config: dict = {}
        self._loading_config = False

        # Rendimiento (serie FPS para grÃ¡fica)
        self.perf_lock = threading.Lock()
        self.fps_series = deque()  # historial limitado a los Ãºltimos PERF_HISTORY_SECONDS segundos
        self.detection_series = deque() # historial de conteos de la clase rastreada
        self.perf_tracked_class = tk.StringVar(value="")
        self.perf_plot_win = None

        # Variables de UI
        self.model_path = tk.StringVar(value=DEFAULT_MODEL)
        self.model_path2 = tk.StringVar(value=DEFAULT_MODEL2)
        self.video_path = tk.StringVar(value=DEFAULT_VIDEO)
        self.out_path = tk.StringVar(value=self._default_out_from_video(DEFAULT_VIDEO))
        self.source_mode = tk.StringVar(value="Archivo")
        self.rtsp_user = tk.StringVar(value="admin")
        self.rtsp_password = tk.StringVar(value="")
        self.rtsp_host = tk.StringVar(value="")
        self.rtsp_port = tk.StringVar(value="554")
        self.rtsp_path = tk.StringVar(value="cam/realmonitor?channel=1&subtype=00")
        self.rtsp_manual_enabled = tk.BooleanVar(value=False)
        self.rtsp_manual_url = tk.StringVar(value="")
        self._rtsp_preview_var = tk.StringVar(value="")
        
        # Valores por defecto (ajustables desde la UI)
        self.conf = 0.05
        self.iou = 0.45
        self.imgsz = 1280
        self.max_det = 400
        self.conf_var = tk.DoubleVar(value=self.conf)
        self.iou_var = tk.DoubleVar(value=self.iou)
        # Rendimiento
        self.perf_mode = tk.StringVar(value="Secuencial")  # Auto | Secuencial | Paralelo
        self.auto_skip = tk.BooleanVar(value=True)
        self.target_fps = tk.IntVar(value=25)
        self._dyn_skip = 0  # frames a saltar dinÃ¡micamente
        self.det_stride = tk.IntVar(value=1)  # procesar cada N frames, reutilizar resultados
        self._det_counter = 0
        # Filtro de clases / colores
        self.filter_in_model = tk.BooleanVar(value=True)  # aplicar filtro en predict(classes=...)
        self.use_retina_masks = tk.BooleanVar(value=True)
        self.masks_as_contours = tk.BooleanVar(value=False)
        self.topk_draw = tk.IntVar(value=0)
        self.highlight_tiny = tk.BooleanVar(value=True)
        self.agnostic_nms = tk.BooleanVar(value=False)
        self.use_half = tk.BooleanVar(value=True)
        # Habilitar/deshabilitar modelos rÃ¡pidamente
        self.enable_m1 = tk.BooleanVar(value=True)
        self.enable_m2 = tk.BooleanVar(value=True)
        # ParÃ¡metros generales (controlados desde 'Ajustes')
        self.imgsz1_var = tk.IntVar(value=self.imgsz)
        self.imgsz2_var = tk.IntVar(value=self.imgsz)
        self.max_det_var = tk.IntVar(value=self.max_det)
        self.stride2_var = tk.IntVar(value=1)
        # Apariencia de dibujo
        self.line_thickness = tk.IntVar(value=2)
        self.font_scale = tk.DoubleVar(value=0.5)
        self.area_text_scale = tk.DoubleVar(value=0.4)
        self.class_cfg: dict[str, dict] = {}  # name -> {color:(B,G,R), m1:bool, m2:bool}
        self.name_to_id_m1: dict[str, int] | None = None
        self.name_to_id_m2: dict[str, int] | None = None
        self.sel_ids_m1: list[int] | None = None
        self.sel_ids_m2: list[int] | None = None
        # Buffer reutilizable para overlays de mÃ¡scaras
        self._overlay_buf: np.ndarray | None = None
        # CachÃ© de contadores de estado
        self._last_counts_frame = 0
        self._last_total1 = 0
        self._last_total2 = 0
        self._last_cls_counts1 = {}
        self._last_cls_counts2 = {}

        # Variables de estado
        self.fps_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Listo. Selecciona modelo y video.")
        self.save_out = False

        # Cache de estado UI (evita tocar Tk desde hilos de trabajo)
        self._ui_cache_lock = threading.Lock()
        self._status_cache: str | None = None
        self._status_last_ui: float = 0.0
        self._status_ui_interval_sec: float = 0.25
        self._status_last_value: str | None = None
        self._fps_cache: float | None = None
        self._fps_last_value: float | None = None
        self._runtime_cfg: dict[str, object] = {}
        self._use_half_active: bool | None = None

        # Variables de estado UI modernas (dashboard)
        self._ui_main_state = tk.StringVar(value="Detenido")  # En marcha/Detenido/Pausado/Reconectando/Error
        self._ui_rtsp_in_state = tk.StringVar(value="--")     # OK/Reconectando/Sin seÃ±al
        self._ui_rtsp_out_state = tk.StringVar(value="--")    # Emitiendo/Detenida
        self._ui_plc_state = tk.StringVar(value="--")         # Conectado/No disponible
        self._ui_cpu = tk.StringVar(value="--")
        self._ui_ram = tk.StringVar(value="--")
        self._ui_disk = tk.StringVar(value="--")
        self._ui_ips = tk.StringVar(value="--")
        self._ui_camera_name = tk.StringVar(value="--")
        self._ui_model1_name = tk.StringVar(value="--")
        self._ui_model2_name = tk.StringVar(value="--")
        self._ui_status_message = tk.StringVar(value="Listo. Selecciona modelo y video.")
        self._ui_fps_value = tk.StringVar(value="--")
        self._resource_job: str | None = None
        self._last_core_settings: dict[str, object] | None = None
        self._restart_in_progress = False
        
        # Datos para widgets del dashboard
        self._current_frame_detections: list[dict] = []
        self._last_detection_counts: dict[str, int] = {}
        
        # ConfiguraciÃ³n FPS
        fps_defaults = {
            "interval": 0.5,
            "low_thresh": 15, "low_color": "#d32f2f",
            "med_thresh": 24, "med_color": "#f57c00",
            "high_color": "#388e3c"
        }
        self.fps_config = dict(fps_defaults)

        # Variables para Presets (Alias)
        self.preset_models_var = tk.StringVar(value="")
        self.preset_rtsp_in_var = tk.StringVar(value="")
        self.preset_rtsp_out_var = tk.StringVar(value="")
        self.active_profile_id = tk.StringVar(value="")
        self._profiles_cache: dict[str, ProfileData] = {}
        self._profiles_order: list[str] = []
        self._profiles_updating_ui = False

        # Sectorizador
        self.sectorizador = Sectorizador() if Sectorizador is not None else None
        if self.sectorizador is not None:
             # Conectar callback de perftrace
             if hasattr(self.sectorizador, 'set_perftrace_callback'):
                  self.sectorizador.set_perftrace_callback(self._log_perftrace_event)

        self.sector_borde_sup = tk.StringVar(value="")
        self.sector_borde_inf = tk.StringVar(value="")
        self.sector_borde_izq = tk.StringVar(value="")
        self.sector_borde_der = tk.StringVar(value="")
        self.sector_modo = tk.StringVar(value="vertical")
        self.sector_num_vert = tk.IntVar(value=1)
        self.sector_num_horiz = tk.IntVar(value=1)
        self.sector_mostrar = tk.BooleanVar(value=True)
        self.sector_mostrar_etiquetas = tk.BooleanVar(value=True)
        # ParÃ¡metros de perspectiva
        self.sector_use_perspective = tk.BooleanVar(value=True)
        self.sector_use_masks = tk.BooleanVar(value=True)
        self.sector_smooth_alpha = tk.DoubleVar(value=0.15)
        self.sector_max_jump = tk.DoubleVar(value=50.0)
        self.sector_inset = tk.IntVar(value=0)
        self.sector_debug_overlay = tk.BooleanVar(value=False)
        self.sector_modo_delimitacion = tk.StringVar(value="Auto")  # Auto|MÃ¡scara|BBox
        self.sector_estabilidad = tk.StringVar(value="Media")      # Baja|Media|Alta
        self.sector_comportamiento_fallo = tk.StringVar(value="Congelar")  # Congelar|RectÃ¡ngulo|Desactivar
        self.sector_opacidad_lineas = tk.DoubleVar(value=1.0)
        self.sector_grosor_lineas = tk.IntVar(value=1)
        self.sector_mostrar_borde_banda = tk.BooleanVar(value=True)
        self.preset_sector_var = tk.StringVar()
        self._sector_avanzado_visible = tk.BooleanVar(value=False)
        # === NUEVO: Variables para bordes curvos y padding ===
        self.sector_curved_enabled = tk.BooleanVar(value=False)
        self.sector_curved_bins_vert = tk.IntVar(value=7)
        self.sector_curved_bins_horiz = tk.IntVar(value=7)
        self.sector_padding_top = tk.IntVar(value=0)
        self.sector_padding_bottom = tk.IntVar(value=0)
        self.sector_padding_left = tk.IntVar(value=0)
        self.sector_padding_right = tk.IntVar(value=0)
        # Tolerancias mÃ¡s agresivas por defecto para multi-instancia
        self.sector_roi_quant_step = tk.IntVar(value=50)
        self.sector_line_quant_step = tk.IntVar(value=50)
        # === NUEVO: Variables para restricciones por clase ===
        self.restricciones_enabled = tk.BooleanVar(value=False)
        self._cached_excluded_sectors_0based: set[int] = set()
        self._restricciones_por_clase: dict[str, dict] = {}  # {clase: {"modo": str, "sectores": list}}

        # Flags de renderizado
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_names = tk.BooleanVar(value=True)
        # Mostrar el score/confianza junto al nombre (visual). Si estÃ¡ desactivado se muestra solo el nombre.
        self.show_confidence = tk.BooleanVar(value=True)
        self.show_masks = tk.BooleanVar(value=True)
        self.use_retina_masks.set(True)
        self.det_stride.set(1)
        self.topk_draw.set(0)

        # Control de refresco UI
        self.ui_fps = tk.IntVar(value=8)
        self._display_rgb_buf: np.ndarray | None = None
        self._display_pil: Image.Image | None = None
        self.last_frame_preview: np.ndarray | None = None
        self._last_ui_refresh: float = 0.0
        self._ui_preview_max_w = 960
        self._ui_preview_max_h = 540
        self._preview_seq: int = 0
        self._last_ui_seq: int = -1

        self.instance_id = self._create_instance_id()
        base_dir = os.path.dirname(__file__)
        self.snapshot_path = self._build_instance_path("snapshots", SNAPSHOT_FILENAME)
        self.snapshot_dir = os.path.dirname(self.snapshot_path)
        self.config_path = os.path.join(base_dir, "..", "config", "sendToPLC_config.json")
        self._metadata_path = self._build_instance_path("metadata", "model_info.json")
        self.capture_dir = self._ensure_instance_dir("captures")
        self.snapshot_write_interval_ms = tk.IntVar(value=SNAPSHOT_DEFAULT_WRITE_INTERVAL_MS)
        self.snapshot_clean_interval_sec = tk.DoubleVar(value=SNAPSHOT_DEFAULT_CLEAN_INTERVAL_SEC)
        self.snapshot_writer = SnapshotAggregator(
            self.snapshot_path,
            write_interval_ms=int(self.snapshot_write_interval_ms.get()),
            clean_every_sec=float(self.snapshot_clean_interval_sec.get()),
        )
        self._snapshot_flush_job: str | None = None
        self._snapshot_last_inst_fps: float | None = None
        self._snapshot_last_avg_fps: float | None = None
        # PERF: throttle PLC pushes and soften draw load during bursts.
        self._last_plc_push_wall: float = 0.0
        self._plc_push_interval_sec: float = 0.15
        self._last_draw_ms: float = 0.0
        self._ui_visible: bool = True
        self._last_perf_debug_ts: float = 0.0
        self._last_sector_ms: float = 0.0
        self._sector_skip_frames: int = 0
        # PerfTrace: Fine-grained performance logging
        self.perf_trace_enabled = tk.BooleanVar(value=False)
        self._perftrace_log_path: str | None = None
        self._perftrace_logger: logging.Logger | None = None
        self._perftrace_handler: logging.FileHandler | None = None
        self._perftrace_run_id: str = ""
        self._perftrace_frame_counter: int = 0
        self._perftrace_config: dict[str, object] = {
            "slow_frame_ms": 45,
            "det_threshold": 15,
            "baseline_interval": 100,
        }
        self._plc_push_queue = queue.Queue(maxsize=1)
        try:
            self._load_config()
            self._normalize_rtsp_out_url()
            # Cargar configuraciÃ³n de FPS desde la configuraciÃ³n general cargada
            fps_saved = self.config.get("fps_settings", {})
            if isinstance(fps_saved, dict):
                self.fps_config.update(fps_saved)
        finally:
            self._loading_config = False

        if not self._calibration_valid:
            self._load_calibration()

        self._on_snapshot_timing_change()

        self._setup_var_traces()
        self._setup_modern_styles()
        self._build_ui()
        self._profiles_refresh_ui()

        # Ajustar visibilidad segÃºn modo cargado y refrescar vista previa RTSP
        self._current_source_kind = ""
        self._on_source_mode_change()
        self._update_rtsp_preview()

        # Lazo de refresco de imagen
        initial_period = int(1000 / max(5, int(self.ui_fps.get())))
        self.root.after(initial_period, self._refresh_image_loop)

        # Garantizar que exista un archivo de configuraciÃ³n inicial
        # Garantizar que existe un archivo de configuración inicial
        self._save_config()

        # Iniciar Garbage Collector (limpieza de archivos obsoletos)
        if garbage_collector:
            garbage_collector.start_garbage_collector()

        self._init_heartbeat()

        self._schedule_snapshot_flush()

        self._set_rtsp_state("detenido", "aplicacion iniciada")

        self._init_send_to_plc_service(); self._init_thick_pulse_timer()

    # ------------------------- UI -------------------------
    def _setup_modern_styles(self):
        """Configura estilos modernos para la UI industrial."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Paleta de colores industrial moderna
        self.colors = {
            # Base
            "bg_app": "#F5F6F8",
            "bg_card": "#FFFFFF",
            "border": "#D9DDE3",
            "text_primary": "#111827",
            "text_secondary": "#6B7280",
            # Acento (solo para acciÃ³n principal)
            "accent": "#2563EB",
            "accent_hover": "#1D4ED8",
            "accent_fg": "#FFFFFF",
            # Estados
            "ok": "#16A34A",
            "warning": "#F59E0B",
            "error": "#DC2626",
            "disabled": "#9CA3AF",
            # Chips de estado
            "chip_ok_bg": "#DCFCE7",
            "chip_ok_fg": "#166534",
            "chip_warn_bg": "#FEF3C7",
            "chip_warn_fg": "#92400E",
            "chip_error_bg": "#FEE2E2",
            "chip_error_fg": "#991B1B",
        }

        
        # Fondo de la aplicaciÃ³n
        self.root.configure(bg=self.colors["bg_app"])

        
        # Estilo base para Frames
        style.configure("TFrame", background=self.colors["bg_app"])
        
        # Estilo para tarjetas (cards)
        style.configure(
            "Card.TFrame",
            background=self.colors["bg_card"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "CardInner.TFrame",
            background=self.colors["bg_card"],
        )
        
        # Dashboard frame
        style.configure("Dashboard.TFrame", background=self.colors["bg_app"])
        
        # Labels bÃ¡sicos
        style.configure(
            "TLabel",
            background=self.colors["bg_app"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Card.TLabel",
            background=self.colors["bg_card"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.colors["bg_card"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "CardSecondary.TLabel",
            background=self.colors["bg_card"],
            foreground=self.colors["text_secondary"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "StateText.TLabel",
            background=self.colors["bg_card"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "FPSValue.TLabel",
            background=self.colors["bg_card"],
            foreground=self.colors["accent"],
            font=("Segoe UI", 24, "bold"),
        )
        style.configure(
            "FPSUnit.TLabel",
            background=self.colors["bg_card"],
            foreground=self.colors["text_secondary"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "StatusBar.TLabel",
            background=self.colors["bg_app"],
            foreground=self.colors["text_secondary"],
            font=("Segoe UI", 10),
            padding=(12, 6),
        )
        style.configure(
            "VideoBar.TLabel",
            background="#1F2937",
            foreground="#E5E7EB",
            font=("Segoe UI", 9),
            padding=(8, 4),
        )
        
        # BotÃ³n Primario (Iniciar/Guardar)
        style.configure(
            "Primary.TButton",
            background=self.colors["accent"],
            foreground=self.colors["accent_fg"],
            font=("Segoe UI", 11, "bold"),
            padding=(16, 10),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[
                ("active", self.colors["accent_hover"]),
                ("pressed", self.colors["accent_hover"]),
                ("disabled", self.colors["disabled"]),
            ],
            foreground=[("disabled", "#FFFFFF")],
        )
        
        # BotÃ³n Secundario (Pausar, Ajustes, Visor)
        style.configure(
            "Secondary.TButton",
            background=self.colors["bg_card"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10),
            padding=(12, 8),
            borderwidth=1,
        )
        style.map(
            "Secondary.TButton",
            background=[
                ("active", "#F3F4F6"),
                ("pressed", "#E5E7EB"),
                ("disabled", self.colors["bg_app"]),
            ],
            foreground=[("disabled", self.colors["disabled"])],
        )
        
        # BotÃ³n Peligro (Detener urgente)
        style.configure(
            "Danger.TButton",
            background=self.colors["error"],
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
            borderwidth=0,
        )
        style.map(
            "Danger.TButton",
            background=[
                ("active", "#B91C1C"),
                ("pressed", "#991B1B"),
                ("disabled", self.colors["disabled"]),
            ],
        )
        
        # Estilo para sliders
        style.configure(
            "TScale",
            background=self.colors["bg_card"],
            troughcolor=self.colors["border"],
        )
        
        # Estilo para LabelFrame
        style.configure(
            "TLabelframe",
            background=self.colors["bg_card"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=self.colors["bg_card"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10, "bold"),
        )
        
        # Estilo para Checkbuttons
        style.configure(
            "TCheckbutton",
            background=self.colors["bg_card"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10),
        )
        
        # Estilo para Combobox
        style.configure(
            "TCombobox",
            font=("Segoe UI", 10),
            padding=4,
        )
        
        # Estilo para Entry
        style.configure(
            "TEntry",
            font=("Segoe UI", 10),
            padding=6,
        )
        
        # Estilo para Notebook (pestaÃ±as)
        style.configure(
            "TNotebook",
            background=self.colors["bg_app"],
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background=self.colors["bg_app"],
            foreground=self.colors["text_primary"],
            font=("Segoe UI", 10),
            padding=(16, 8),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["bg_card"])],
            foreground=[("selected", self.colors["accent"])],
        )
        
        # Estilo para Spinbox
        style.configure(
            "TSpinbox",
            font=("Segoe UI", 10),
            padding=4,
        )
        
        # Estilo para PanedWindow
        style.configure(
            "TPanedwindow",
            background=self.colors["bg_app"],
        )

    def _add_info_icon(self, parent: tk.Misc, row: int, column: int, key: str) -> None:
        if InfoIcon is None:
            return
        icon = InfoIcon(parent, key)
        icon.grid(row=row, column=column, sticky="e", padx=(0, 4))

    def _label_with_info(
        self,
        parent: tk.Misc,
        text: str,
        key: str,
        row: int,
        column: int = 0,
        *,
        sticky: str = "w",
        pady: int | tuple[int, int] = 0,
        padx: int | tuple[int, int] = 0,
        columnspan: int = 1,
        **label_kwargs: object,
    ) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky=sticky, pady=pady, padx=padx, columnspan=columnspan)
        ttk.Label(frame, text=text, **label_kwargs).pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frame, key).pack(side="left", padx=(6, 0))
        return frame

    def _check_with_info(
        self,
        parent: tk.Misc,
        text: str,
        variable: tk.Variable,
        key: str,
        row: int,
        column: int = 0,
        *,
        sticky: str = "w",
        pady: int | tuple[int, int] = 0,
        padx: int | tuple[int, int] = 0,
        columnspan: int = 1,
        **check_kwargs: object,
    ) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky=sticky, pady=pady, padx=padx, columnspan=columnspan)
        ttk.Checkbutton(frame, text=text, variable=variable, **check_kwargs).pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frame, key).pack(side="left", padx=(6, 0))
        return frame

    def _create_status_chip(self, parent, text: str, status: str = "ok") -> ttk.Label:
        """
        Crea un chip de estado visual.
        status: 'ok', 'warning', 'error'
        """
        color_map = {
            "ok": (self.colors["chip_ok_bg"], self.colors["chip_ok_fg"]),
            "warning": (self.colors["chip_warn_bg"], self.colors["chip_warn_fg"]),
            "error": (self.colors["chip_error_bg"], self.colors["chip_error_fg"]),
        }
        bg, fg = color_map.get(status, color_map["ok"])
        chip = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 9),
            padx=8,
            pady=3,
        )
        return chip

    def _build_ui(self):
        """Construye la UI principal con layout horizontal: Dashboard (60%) + Vídeo (40%)."""
        # Configurar el grid principal
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)  # Barra de estado inferior
        
        # Contenedor principal con PanedWindow horizontal
        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned = self.main_paned
        main_paned.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # ============== PANEL IZQUIERDO: DASHBOARD ==============
        dashboard = self._build_dashboard(main_paned)
        main_paned.add(dashboard, weight=40)
        
        # ============== PANEL DERECHO: VÍDEO ==============
        video_panel = self._build_video_panel(main_paned)
        main_paned.add(video_panel, weight=60)
        
        # NUEVO: Forzar posición del sash al 40% inicial de forma explícita (Dashboard 40%, Vídeo 60%).
        # Se usa un bucle de comprobación para asegurar que el root esté maximizado y tenga dimensiones finales.
        def set_default_layout(attempts=0):
            try:
                self.root.update_idletasks()
                total_w = self.main_paned.winfo_width()
                # Consideramos que se ha maximizado si el ancho es razonable (>800px)
                if total_w > 800:
                    target_pos = int(total_w * 0.40)
                    self.main_paned.sashpos(0, target_pos)
                    LOGGER.info(f"[Layout] Sash ajustado al 40%% (D:40/V:60): {target_pos}px de {total_w}px.")
                elif attempts < 10:
                    # Re-intentar si aún no ha crecido la ventana
                    self.root.after(200, lambda: set_default_layout(attempts + 1))
            except Exception as e:
                LOGGER.warning(f"[Layout] Error ajustando sash: {e}")

        # Iniciar secuencia de ajuste
        self.root.after(500, set_default_layout)
        
        # ============== BARRA DE ESTADO INFERIOR ==============
        self._build_status_bar()
        
        # Actualizar indicadores iniciales
        self._update_rtsp_indicator(False)
        self._schedule_resource_update()

    def _build_dashboard(self, parent) -> tk.Frame:
        """Construye el panel izquierdo con scrollbar y tarjetas de dashboard."""
        container = ttk.Frame(parent)
        
        canvas = tk.Canvas(container, bg=self.colors["bg_app"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Dashboard.TFrame", padding=12)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Crear ventana en el canvas para el frame
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Hacer que el frame ocupe todo el ancho del canvas y manejar wheel
        def _on_canvas_configure(e):
             canvas.itemconfig(canvas_frame, width=e.width)
        canvas.bind("<Configure>", _on_canvas_configure)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mouse(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mouse(event):
            canvas.unbind_all("<MouseWheel>")

        container.bind("<Enter>", _bind_mouse)
        container.bind("<Leave>", _unbind_mouse)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        scrollable_frame.columnconfigure(0, weight=1)
        
        row = 0
        
        # ============== TARJETA A: ESTADO GENERAL ==============
        status_card = self._build_status_card(scrollable_frame)
        status_card.grid(row=row, column=0, sticky="new", pady=(0, 12))
        row += 1
        
        # ============== TARJETA B: ACCIONES ==============
        actions_card = self._build_actions_card(scrollable_frame)
        actions_card.grid(row=row, column=0, sticky="new", pady=(0, 12))
        row += 1
        
        # ============== TARJETA C: SENSIBILIDAD ==============
        sensitivity_card = self._build_sensitivity_card(scrollable_frame)
        sensitivity_card.grid(row=row, column=0, sticky="new", pady=(0, 12))
        row += 1
        
        # Espacio flexible
        scrollable_frame.rowconfigure(row, weight=1)
        
        return container

    def _build_status_card(self, parent) -> ttk.Frame:
        """Tarjeta A: Estado general del sistema."""
        card = tk.Frame(parent, bg=self.colors["bg_card"], bd=1, relief="solid", padx=16, pady=12)
        card.columnconfigure(1, weight=1)
        
        # TÃ­tulo de la tarjeta
        ttk.Label(card, text="🛰 Estado", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        
        # Estado principal (grande)
        self._state_label = ttk.Label(
            card, 
            textvariable=self._ui_main_state, 
            style="StateText.TLabel"
        )
        self._state_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))
        
        # Info de cÃ¡mara y modelos
        info_frame = tk.Frame(card, bg=self.colors["bg_card"])
        info_frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 12))
        
        ttk.Label(info_frame, text="📷 Camara:", style="CardSecondary.TLabel").pack(side="left")
        ttk.Label(info_frame, textvariable=self._ui_camera_name, style="Card.TLabel").pack(side="left", padx=(4, 16))
        
        ttk.Label(info_frame, text="🧠 M1:", style="CardSecondary.TLabel").pack(side="left")
        ttk.Label(info_frame, textvariable=self._ui_model1_name, style="Card.TLabel").pack(side="left", padx=(4, 16))
        
        ttk.Label(info_frame, text="🧠 M2:", style="CardSecondary.TLabel").pack(side="left")
        ttk.Label(info_frame, textvariable=self._ui_model2_name, style="Card.TLabel").pack(side="left", padx=(4, 0))

        # Perfil activo
        profile_frame = tk.Frame(card, bg=self.colors["bg_card"])
        profile_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 12))
        ttk.Label(profile_frame, text="👤 Perfil:", style="CardSecondary.TLabel").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(profile_frame, "detect_manchas.profiles.active").pack(side="left", padx=(6, 0))
        self.combo_profiles_main = ttk.Combobox(profile_frame, state="readonly", values=(), width=22)
        self.combo_profiles_main.pack(side="left", padx=(4, 8))
        self.combo_profiles_main.bind("<<ComboboxSelected>>", lambda *_: self._profiles_on_selected("main"))
        ttk.Button(
            profile_frame,
            text="Gestionar...",
            command=lambda: self._open_settings_dialog(tab_name="General"),
            style="Secondary.TButton",
        ).pack(side="left")

        # Chips de estado
        chips_frame = tk.Frame(card, bg=self.colors["bg_card"])
        chips_frame.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 12))
        
        self._chip_rtsp_in = self._create_status_chip(chips_frame, "📥 Entrada RTSP: --", "ok")
        self._chip_rtsp_in.pack(side="left", padx=(0, 8))
        
        self._chip_rtsp_out = self._create_status_chip(chips_frame, "📤 Salida RTSP: --", "ok")
        self._chip_rtsp_out.pack(side="left", padx=(0, 8))
        
        self._chip_plc = self._create_status_chip(chips_frame, "🔌 PLC: --", "ok")
        self._chip_plc.pack(side="left")
        
        # FPS grande y widgets personalizables + fila compacta de recursos
        fps_widgets_frame = tk.Frame(card, bg=self.colors["bg_card"])
        fps_widgets_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        fps_widgets_frame.columnconfigure(0, weight=1)  # widgets expand
        
        # Widgets personalizables (a la izquierda)
        self._dashboard_widget_manager = None
        if DashboardWidgetManager is not None:
            try:
                self._dashboard_widget_manager = DashboardWidgetManager(self, fps_widgets_frame)
                widgets_area = self._dashboard_widget_manager.build_ui()
                widgets_area.pack(side="left", fill="x", expand=True)
            except Exception as e:
                LOGGER.warning("Error inicializando widgets personalizables: %s", e)
        
        # FPS (a la derecha)
        fps_frame = tk.Frame(fps_widgets_frame, bg=self.colors["bg_card"])
        fps_frame.pack(side="right", padx=(8, 0))
        
        self._fps_display = ttk.Label(fps_frame, textvariable=self._ui_fps_value, style="FPSValue.TLabel", width=6, anchor="e")
        self._fps_display.pack(side="left")
        
        lbl_unit = ttk.Label(fps_frame, text="FPS", style="FPSUnit.TLabel")
        lbl_unit.pack(side="left", padx=(4, 0), pady=(8, 0))
        
        # Binding para configuraciÃ³n FPS
        for w in (self._fps_display, lbl_unit, fps_frame):
             w.bind("<Button-3>", self._open_fps_config)
             w.config(cursor="hand2")

        # Recursos en una fila compacta bajo FPS
        resources_row = tk.Frame(card, bg=self.colors["bg_card"])
        resources_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        for i, (label, var) in enumerate((
            ("CPU", self._ui_cpu),
            ("RAM", self._ui_ram),
            ("Disco", self._ui_disk),
            ("IPS", self._ui_ips),
        )):
            col = tk.Frame(resources_row, bg=self.colors["bg_card"])
            col.pack(side="left", expand=True, fill="x", padx=(0 if i == 0 else 8, 0))
            ttk.Label(col, text=f"{label}", style="CardSecondary.TLabel").pack(anchor="w")
            ttk.Label(col, textvariable=var, style="CardTitle.TLabel").pack(anchor="w")
        
        return card

    def _build_actions_card(self, parent) -> ttk.Frame:
        """Tarjeta B: Botones de acciÃ³n principales."""
        card = tk.Frame(parent, bg=self.colors["bg_card"], bd=1, relief="solid", padx=16, pady=12)
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        
        # TÃ­tulo
        ttk.Label(card, text="Acciones", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        
        # BotÃ³n principal (Iniciar/Detener)
        self.btn_start = ttk.Button(
            card, 
            text="▶ Iniciar", 
            command=self.start, 
            style="Primary.TButton"
        )
        self.btn_start.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 8))
        
        # BotÃ³n secundario (Pausar/Reanudar)
        self.btn_pause = ttk.Button(
            card, 
            text="⏸ Pausar", 
            command=self.toggle_pause, 
            state="disabled",
            style="Secondary.TButton"
        )
        self.btn_pause.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 12))
        
        # BotÃ³n Detener
        self.btn_stop = ttk.Button(
            card, 
            text="⏹ Detener", 
            command=self.stop, 
            state="disabled",
            style="Danger.TButton"
        )
        self.btn_stop.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 12))
        
        # Fila de botones pequeÃ±os
        small_btns = tk.Frame(card, bg=self.colors["bg_card"])
        small_btns.grid(row=4, column=0, columnspan=2, sticky="we")
        small_btns.columnconfigure(0, weight=1)
        small_btns.columnconfigure(1, weight=1)
        small_btns.columnconfigure(2, weight=1)
        
        ttk.Button(
            small_btns, 
            text="⚙ Ajustes", 
            command=self._open_settings_dialog,
            style="Secondary.TButton"
        ).grid(row=0, column=0, sticky="we", padx=(0, 4))
        
        self.btn_viewer = ttk.Button(
            small_btns, 
            text="📡 Visor", 
            command=self._open_rtsp_viewer,
            style="Secondary.TButton"
        )
        self.btn_viewer.grid(row=0, column=1, sticky="we", padx=(4, 4))
        
        self.btn_plc = ttk.Button(
            small_btns, 
            text="🔌 PLC", 
            command=self._open_plc_window,
            style="Secondary.TButton"
        )
        self.btn_plc.grid(row=0, column=2, sticky="we", padx=(4, 0))
        self.btn_plc.state(["disabled"])
        
        return card

    def _build_sensitivity_card(self, parent) -> ttk.Frame:
        """Tarjeta C: Ajustes de sensibilidad rapida."""
        card = tk.Frame(parent, bg=self.colors["bg_card"], bd=1, relief="solid", padx=16, pady=12)
        card.columnconfigure(0, weight=1)
        
        # Titulo
        ttk.Label(card, text="🎚 Sensibilidad", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )
        
        # Slider Conf
        conf_frame = tk.Frame(card, bg=self.colors["bg_card"])
        conf_frame.grid(row=1, column=0, sticky="we", pady=(0, 8))
        conf_frame.columnconfigure(1, weight=1)
        
        self._label_with_info(
            conf_frame,
            "🎯 Conf:",
            "detect_manchas.sensitivity.conf",
            row=0,
            column=0,
            sticky="w",
            style="Card.TLabel",
        )
        self.lbl_conf_val = ttk.Label(conf_frame, text=f"{self.conf*100:.0f}%", style="Card.TLabel")
        self.lbl_conf_val.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.scale_conf = ttk.Scale(
            conf_frame, 
            from_=0.0, 
            to=1.0, 
            orient="horizontal",
            command=self._on_conf_change
        )
        self.scale_conf.configure(length=220)
        self.scale_conf.set(self.conf)
        self.scale_conf.grid(row=0, column=1, sticky="we", padx=8)
        
        # Slider IoU
        iou_frame = tk.Frame(card, bg=self.colors["bg_card"])
        iou_frame.grid(row=2, column=0, sticky="we", pady=(0, 8))
        iou_frame.columnconfigure(1, weight=1)
        
        self._label_with_info(
            iou_frame,
            "🔗 IoU:",
            "detect_manchas.sensitivity.iou",
            row=0,
            column=0,
            sticky="w",
            style="Card.TLabel",
        )
        self.lbl_iou_val = ttk.Label(iou_frame, text=f"{self.iou*100:.0f}%", style="Card.TLabel")
        self.lbl_iou_val.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.scale_iou = ttk.Scale(
            iou_frame, 
            from_=0.0, 
            to=1.0, 
            orient="horizontal",
            command=self._on_iou_change
        )
        self.scale_iou.configure(length=220)
        self.scale_iou.set(self.iou)
        self.scale_iou.grid(row=0, column=1, sticky="we", padx=8)
        
        # Texto de ayuda
        help_frame = tk.Frame(card, bg=self.colors["bg_card"])
        help_frame.grid(row=3, column=0, sticky="we")
        
        ttk.Label(
            help_frame, 
            text="Detecta demasiado? sube Conf | Junta manchas? sube IoU",
            style="CardSecondary.TLabel"
        ).pack(anchor="w")
        
        return card

    def _build_video_panel(self, parent) -> ttk.Frame:
        """Construye el panel derecho con la visualizaciÃ³n del vÃ­deo."""
        video_container = ttk.Frame(parent, style="TFrame")
        
        # Panel dividido verticalmente: Superior (Activa) / Inferior (Futura)
        video_paned = ttk.PanedWindow(video_container, orient="vertical")
        video_paned.pack(fill="both", expand=True)

        # ==================== CAMARA SUPERIOR ====================
        top_cam_frame = tk.Frame(video_paned, bg=self.colors["bg_app"])
        video_paned.add(top_cam_frame, weight=2)  # Darle mÃ¡s peso inicialmente si se quiere
        
        # Barra superior (Header)
        top_bar = tk.Frame(top_cam_frame, bg="#111827", height=32)
        top_bar.pack(side="top", fill="x")
        top_bar.pack_propagate(False)
        
        # Indicador de "En vivo" o tÃ­tulo
        tk.Label(
            top_bar, 
            text="CAMARA SUPERIOR", 
            bg="#111827", 
            fg="#60A5FA",  # Azul claro
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=12)

        # Info de cÃ¡mara y modo (existente, movida aquÃ­)
        tk.Label(top_bar, text="|", bg="#111827", fg="#374151").pack(side="left", padx=4)
        
        self._video_bar_camera = tk.Label(
            top_bar, 
            textvariable=self._ui_camera_name, 
            bg="#111827", 
            fg="#9CA3AF",
            font=("Segoe UI", 9)
        )
        self._video_bar_camera.pack(side="left", padx=8)

        tk.Label(
            top_bar, 
            textvariable=self.source_mode, 
            bg="#111827", 
            fg="#4B5563",
            font=("Segoe UI", 8)
        ).pack(side="right", padx=12)

        # Ãrea de vÃ­deo real
        video_area = tk.Frame(top_cam_frame, bg=self.colors["bg_app"])
        video_area.pack(side="top", fill="both", expand=True)
        
        # Label alineado arriba
        self.video_label = tk.Label(video_area, text="", bg=self.colors["bg_app"])
        self.video_label.pack(side="top", anchor="n")

        # ==================== CAMARA INFERIOR (Placeholder) ====================
        bot_cam_frame = tk.Frame(video_paned, bg=self.colors["bg_app"])
        video_paned.add(bot_cam_frame, weight=1)

        # Barra inferior (Header)
        bot_bar = tk.Frame(bot_cam_frame, bg="#1F2937", height=32)
        bot_bar.pack(side="top", fill="x")
        bot_bar.pack_propagate(False)

        tk.Label(
            bot_bar, 
            text="CAMARA INFERIOR", 
            bg="#1F2937", 
            fg="#9CA3AF",  # Gris desactivado
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=12)

        tk.Label(
            bot_bar, 
            text="NO DETECTADO", 
            bg="#1F2937", 
            fg="#DC2626", # Rojo oscuro
            font=("Segoe UI", 8, "bold")
        ).pack(side="right", padx=12)

        # Placeholder visual
        placeholder_area = tk.Frame(bot_cam_frame, bg="#E5E7EB") # Gris muy claro
        placeholder_area.pack(side="top", fill="both", expand=True, padx=1, pady=(0,1))
        
        # Contenido del placeholder (centro)
        center_ph = tk.Frame(placeholder_area, bg="#E5E7EB")
        center_ph.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(
            center_ph, 
            text="Sin senal de video", 
            bg="#E5E7EB", 
            fg="#9CA3AF",
            font=("Segoe UI", 12)
        ).pack()
        
        tk.Label(
            center_ph, 
            text="Verifique conexion o configuracion de doble camara", 
            bg="#E5E7EB", 
            fg="#9CA3AF",
            font=("Segoe UI", 9)
        ).pack(pady=(4,0))
        
        return video_container

    def _build_status_bar(self):
        """Construye la barra de estado inferior global."""
        status_bar = tk.Frame(self.root, bg=self.colors["bg_app"], height=36)
        status_bar.grid(row=1, column=0, sticky="we")
        status_bar.grid_propagate(False)
        
        # Mensaje de estado a la izquierda
        self.lbl_status = ttk.Label(
            status_bar, 
            textvariable=self._ui_status_message,
            style="StatusBar.TLabel"
        )
        self.lbl_status.pack(side="left", padx=(12, 0))
        
        # FPS y estado a la derecha
        right_frame = tk.Frame(status_bar, bg=self.colors["bg_app"])
        right_frame.pack(side="right", padx=(0, 12))
        
        self.lbl_fps = ttk.Label(
            right_frame, 
            textvariable=self.fps_var,
            style="StatusBar.TLabel"
        )
        self.lbl_fps.pack(side="right")
        
        # Indicador RTSP
        rtsp_indicator = tk.Frame(right_frame, bg=self.colors["bg_app"])
        rtsp_indicator.pack(side="right", padx=(0, 16))
        
        self._rtsp_indicator_canvas = tk.Canvas(
            rtsp_indicator, 
            width=12, 
            height=12, 
            highlightthickness=0,
            bg=self.colors["bg_app"]
        )
        self._rtsp_indicator_canvas.pack(side="left", padx=(0, 4))
        self._rtsp_indicator_circle = self._rtsp_indicator_canvas.create_oval(
            1, 1, 11, 11, 
            fill="#7a7f89", 
            outline=""
        )
        
        ttk.Label(rtsp_indicator, textvariable=self._rtsp_status, style="StatusBar.TLabel").pack(side="left")
        
        # Guardar referencia para diagnÃ³stico heartbeat (se construirÃ¡ en settings ahora)
        # Se mantiene la llamada por compatibilidad pero el contenido va a Ajustes
        self._hb_local_box = None
        self._hb_remote_box = None

    def _rtsp_schedule_indicator(self, active: bool) -> None:
        try:
            self.root.after(0, lambda a=active: self._update_rtsp_indicator(a))
        except Exception:
            pass

    def _update_rtsp_indicator(self, active: bool) -> None:
        color = "#2ecc71" if active else "#7a7f89"
        text = "Emitiendo" if active else "Detenido"
        try:
            self._rtsp_status.set(text)
        except Exception:
            pass
        if self._rtsp_indicator_canvas is not None and self._rtsp_indicator_circle is not None:
            try:
                self._rtsp_indicator_canvas.itemconfigure(self._rtsp_indicator_circle, fill=color)
            except Exception:
                pass

    def _schedule_resource_update(self) -> None:
        if self._resource_job is not None:
            try:
                self.root.after_cancel(self._resource_job)
            except Exception:
                pass
            self._resource_job = None
        # Ejecutar ahora y luego cada segundo
        self._update_resource_metrics()
        try:
            self._resource_job = self.root.after(1000, self._schedule_resource_update)
        except Exception:
            self._resource_job = None

    def _update_resource_metrics(self) -> None:
        """Refresca CPU/RAM/Disco/IPS mostrando valores porcentuales."""
        # CPU / RAM
        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=None)
                self._ui_cpu.set(f"{cpu:.0f}%")
            except Exception:
                self._ui_cpu.set("--")

            try:
                ram = psutil.virtual_memory().percent
                self._ui_ram.set(f"{ram:.0f}%")
            except Exception:
                self._ui_ram.set("--")
        else:
            self._ui_cpu.set("--")
            self._ui_ram.set("--")

        # Disco: usar la unidad donde corre la app
        try:
            base_path = os.path.abspath(os.path.dirname(__file__))
            total, used, free = shutil.disk_usage(base_path)
            used_pct = (used / total) * 100 if total else 0.0
            self._ui_disk.set(f"{used_pct:.0f}%")
        except Exception:
            self._ui_disk.set("--")

        # IPS: reutilizar FPS mostrado (instantÃ¡neo)
        try:
            fps_val = self._fps_last_value
            if fps_val is None:
                fps_val = float(self._ui_fps_value.get()) if self._ui_fps_value.get() not in {"--", ""} else None
            self._ui_ips.set(f"{fps_val:.1f}" if fps_val is not None else "--")
        except Exception:
            self._ui_ips.set("--")

    def _build_heartbeat_section(self, parent: ttk.Frame) -> None:
        hb_frame = ttk.Labelframe(parent, text="Diagnostico Heartbeat")
        hb_frame.grid(row=11, column=0, columnspan=3, sticky="we", pady=(8, 0))
        hb_frame.columnconfigure(1, weight=1)

        ttk.Label(hb_frame, text="Enviado (Detector):").grid(row=0, column=0, sticky="nw", padx=6, pady=4)
        self._hb_local_box = tk.Text(hb_frame, height=2, width=48, state="disabled", wrap="word")
        self._hb_local_box.grid(row=0, column=1, sticky="we", padx=(0, 6), pady=4)

        ttk.Label(hb_frame, text="Recibido (Visor):").grid(row=1, column=0, sticky="nw", padx=6, pady=(0, 4))
        self._hb_remote_box = tk.Text(hb_frame, height=2, width=48, state="disabled", wrap="word")
        self._hb_remote_box.grid(row=1, column=1, sticky="we", padx=(0, 6), pady=(0, 4))

        ttk.Label(
            hb_frame,
            text="El heartbeat envia cada 5s un texto distinto.",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

    def _set_textarea(self, widget: tk.Text | None, text: str) -> None:
        if widget is None:
            return
        try:
            widget.configure(state="normal")
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, text)
            widget.configure(state="disabled")
        except tk.TclError:
            pass

    def _init_heartbeat(self) -> None:
        self._hb_bridge = _HeartbeatBridge(
            name="detector",
            listen_port=HEARTBEAT_DETECTOR_PORT,
            tk_root=self.root,
            on_message=self._on_heartbeat_message,
        )
        ok = self._hb_bridge.start()
        if not ok:
            self._set_textarea(self._hb_local_box, "Heartbeat inactivo (puerto ocupado)")
            return
        self._refresh_heartbeat_targets()
        self._schedule_heartbeat()

    def _refresh_heartbeat_targets(self) -> None:
        bridge = self._hb_bridge
        if bridge is None:
            return
        # asegurar destinos locales para pruebas en la misma mÃ¡quina
        bridge.add_target(("127.0.0.1", HEARTBEAT_VIEWER_PORT))
        if self._local_ip:
            bridge.add_target((self._local_ip, HEARTBEAT_VIEWER_PORT))

        try:
            url = self.rtsp_out_url.get().strip()
        except Exception:
            url = ""
        if not url:
            return
        parsed = urlparse(url)
        host = parsed.hostname
        if host and host not in {"127.0.0.1", "localhost"}:
            bridge.add_target((host, HEARTBEAT_VIEWER_PORT))

    def _schedule_heartbeat(self) -> None:
        self._check_rtsp_watchdogs()
        letter = next(self._hb_cycle)
        payload = self._heartbeat_payload()
        payload["beat"] = f"DET-{letter}"
        payload_text = json.dumps(payload, ensure_ascii=False)
        self._set_textarea(self._hb_local_box, payload_text)
        if self._hb_bridge is not None:
            self._hb_bridge.send_message(payload_text)
        self._hb_job = self.root.after(int(HEARTBEAT_INTERVAL_SEC * 1000), self._schedule_heartbeat)

    def _update_plc_button_state(self) -> None:
        if not hasattr(self, "btn_plc"):
            return
        try:
            if self._plc_service is None or SendToPLCWindow is None:
                self.btn_plc.state(["disabled"])
            else:
                self.btn_plc.state(["!disabled"])
        except Exception:  # noqa: BLE001
            pass

    def _init_send_to_plc_service(self) -> None:
        if SendToPLCService is None:
            if SENDTOPLC_IMPORT_ERROR is not None:
                LOGGER.warning(
                    "sendToPLCService no estÃ¡ disponible (fallÃ³ importaciÃ³n). Error: %s",
                    SENDTOPLC_IMPORT_ERROR,
                )
            else:
                LOGGER.warning("sendToPLCService no estÃ¡ disponible en este entorno (tipo desconocido).")
            self._plc_service = None
        else:
            try:
                LOGGER.info(
                    "Creando SendToPLCService con snapshot=%s, config=%s, capture_dir=%s",
                    self.snapshot_path,
                    self.config_path,
                    self.capture_dir,
                )
                self._plc_service = SendToPLCService(
                    snapshot_path=self.snapshot_path,
                    config_path=self.config_path,
                    persist_snapshots=False,
                    capture_dir=self.capture_dir,
                )
                self._plc_service.start()
                LOGGER.info("sendToPLCService arrancado en segundo plano (instancia %s).", self.instance_id)
                self._publish_model_metadata()
                if self._events_poll_job is None:
                    self._events_poll_job = self.root.after(300, self._poll_service_events)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("No se pudo iniciar sendToPLCService: %s", exc)
                self._plc_service = None
        self._update_plc_button_state()

    def _stop_plc_service(self) -> None:
        if self._plc_service is None:
            return
        try:
            self._plc_service.stop()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Error al detener sendToPLCService: %s", exc)
        finally:
            self._plc_service = None
            self._cancel_events_poll()
            self._update_plc_button_state()

    def _cancel_events_poll(self) -> None:
        if self._events_poll_job is not None:
            try:
                self.root.after_cancel(self._events_poll_job)
            except Exception:  # noqa: BLE001
                pass
            self._events_poll_job = None
            self._update_plc_button_state()

    def _open_trainer_gui(self) -> None:
        """Abre la interfaz de entrenamiento trainer_gui.py"""
        try:
            trainer_path = os.path.join(os.path.dirname(__file__), "trainer_gui.py")
            if not os.path.exists(trainer_path):
                messagebox.showerror("Entrenamiento", "No se encuentra el archivo trainer_gui.py")
                return
            
            # Ejecutar trainer_gui.py en un nuevo proceso
            subprocess.Popen([sys.executable, trainer_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            
        except Exception as exc:
            LOGGER.error("No se pudo abrir la interfaz de entrenamiento: %s", exc)
            messagebox.showerror("Entrenamiento", f"No se pudo abrir la interfaz de entrenamiento.\nDetalle: {exc}")

    def _open_plc_window(self) -> None:
        if self._plc_service is None:
            messagebox.showerror("sendToPLC", "El servicio sendToPLC no esta disponible.")
            return
        if SendToPLCWindow is None:
            messagebox.showerror("sendToPLC", "La interfaz sendToPLC no esta disponible en este entorno.")
            return
        if self._plc_window is not None:
            wnd = getattr(self._plc_window, "window", None)
            if wnd is not None and wnd.winfo_exists():
                try:
                    wnd.deiconify()
                    wnd.lift()
                    wnd.focus_force()
                except Exception:  # noqa: BLE001
                    pass
                return
            self._plc_window = None
        try:
            self._plc_window = SendToPLCWindow(self.root, self._plc_service, on_close=self._on_plc_window_closed)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("No se pudo abrir la ventana sendToPLC: %s", exc)
            messagebox.showerror("sendToPLC", f"No se pudo abrir la ventana sendToPLC.\nDetalle: {exc}")
            self._plc_window = None

    def _on_plc_window_closed(self) -> None:
        self._plc_window = None
        self._update_plc_button_state()

    def _destroy_plc_window(self) -> None:
        if self._plc_window is None:
            return
        try:
            self._plc_window.destroy()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudo cerrar la ventana sendToPLC: %s", exc)
        finally:
            self._plc_window = None
            self._update_plc_button_state()

    def _poll_service_events(self) -> None:
        self._events_poll_job = None
        if not self.root.winfo_exists() or self._plc_service is None:
            return
        try:
            events = self._plc_service.get_pending_events()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("No se pudieron obtener eventos del servicio PLC: %s", exc)
            events = []
        for event in events or ():
            self._handle_service_event(event)
        self._cleanup_overlay_messages()
        if self.root.winfo_exists() and self._plc_service is not None:
            self._events_poll_job = self.root.after(400, self._poll_service_events)

    def _handle_service_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        etype = str(event.get("type", ""))
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        now = time.time()
        if etype == "overlay_message":
            # Deshabilitado por peticiÃ³n del usuario: no mostrar textos naranjas superpuestos
            pass
            # text = str(payload.get("text", "")).strip()
            # if not text:
            #     return
            # color = _tk_color_to_bgr(str(payload.get("color", "#ffbc00")))
            # duration = int(payload.get("duration_ms", 4000) or 4000)
            # opacity = float(payload.get("opacity", 0.8) or 0.8)
            # self._overlay_messages.append(
            #     _OverlayMessage(
            #         text=text,
            #         color=color,
            #         duration_ms=max(500, duration),
            #         opacity=max(0.0, min(1.0, opacity)),
            #         created=now,
            #     )
            # )
        elif etype == "snapshot_request":
            file_path = payload.get("file_path")
            annotate = bool(payload.get("annotate", False))
            if isinstance(file_path, str) and file_path:
                self._handle_snapshot_request(file_path, annotate)

    def _cleanup_overlay_messages(self) -> None:
        if not self._overlay_messages:
            return
        cutoff = time.time()
        self._overlay_messages = [
            msg
            for msg in self._overlay_messages
            if (cutoff - msg.created) * 1000.0 < msg.duration_ms
        ]

    def _handle_snapshot_request(self, file_path: str, annotate: bool) -> None:
        frame = None
        with self.frame_lock:
            if self.last_frame_bgr is not None:
                frame = self.last_frame_bgr.copy()
        if frame is None:
            LOGGER.warning("Snapshot solicitado pero no hay frame disponible.")
            return
        if annotate:
            try:
                if getattr(self, "_last_result1", None) is not None:
                    self._draw_result_on(frame, self._last_result1, color=None, name_prefix="M1")
                if getattr(self, "_last_result2", None) is not None:
                    self._draw_result_on(frame, self._last_result2, color=None, name_prefix="M2")
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Fallo al anotar snapshot: %s", exc)
        directory = os.path.dirname(file_path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                LOGGER.warning("No se pudo preparar carpeta de captura %s: %s", directory, exc)
                return
        try:
            if not cv2.imwrite(file_path, frame):
                raise OSError("cv2.imwrite devolviÃ³ False")
            LOGGER.info("Captura guardada en %s", file_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudo guardar la captura en %s: %s", file_path, exc)

    def _open_captures_folder(self) -> None:
        path = self.capture_dir
        if not path:
            messagebox.showerror("Capturas", "No hay carpeta de capturas configurada.")
            return
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Capturas", f"No se pudo preparar la carpeta de capturas:\n{exc}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Capturas", f"No se pudo abrir la carpeta de capturas:\n{exc}")

    def _open_documentation_folder(self) -> None:
        """Abrir la carpeta de documentaciÃ³n (Doc) relativa al programa."""
        docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")
        if os.path.exists(docs_path):
            os.startfile(docs_path)
        else:
            messagebox.showinfo("Documentacion", "No se encontro la carpeta de documentacion.")

    def _open_rtsp_viewer(self):
        try:
            import subprocess
            import sys
            
            viewer_path = os.path.join(os.path.dirname(__file__), "rtsp_from_officeA.py")
            
            if os.path.exists(viewer_path):
                subprocess.Popen([sys.executable, viewer_path])
            else:
                messagebox.showerror("Visor RTSP", f"No se encontro el archivo del visor:\n{viewer_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Visor RTSP", f"No se pudo iniciar el visor RTSP:\n{exc}")

    def _create_instance_id(self) -> str:
        return uuid.uuid4().hex[:10]

    def _build_instance_path(self, subdir: str, filename: str) -> str:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "runtime", subdir, self.instance_id)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, filename)

    def _ensure_instance_dir(self, subdir: str) -> str:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "runtime", subdir, self.instance_id)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _publish_model_metadata(self) -> None:
        if self._plc_service is None:
            return
        primary = self.model_path.get().strip()
        secondary = self.model_path2.get().strip()

        def _task() -> None:
            classes: set[str] = set()
            for path in (primary, secondary):
                classes.update(self._load_model_names(path))

            if not classes:
                # fallback a diccionarios ya cargados en sesiÃ³n
                try:
                    if isinstance(self.name_to_id_m1, dict):
                        classes.update(map(str, self.name_to_id_m1.keys()))
                    if isinstance(self.name_to_id_m2, dict):
                        classes.update(map(str, self.name_to_id_m2.keys()))
                except Exception:
                    pass

            metadata = {
                "instance_id": self.instance_id,
                "models": {
                    "primary": primary,
                    "secondary": secondary,
                },
                "classes": sorted(classes),
            }
            try:
                self._plc_service.update_model_metadata(metadata)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudo publicar metadatos del modelo: %s", exc)

        threading.Thread(target=_task, name="ModelMetadata", daemon=True).start()

    def _load_model_names(self, path: str) -> set[str]:
        if not path:
            return set()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return set()
        cache_entry = self._model_names_cache.get(path)
        if cache_entry and cache_entry[0] == mtime:
            return set(cache_entry[1])

        names: set[str] = set()

        # Intento ligero: leer directo el checkpoint con torch
        try:
            ckpt = torch.load(path, map_location="cpu")
            raw_names = None
            if isinstance(ckpt, dict):
                raw_names = ckpt.get("names")
                if raw_names is None:
                    model_obj = ckpt.get("model")
                    if hasattr(model_obj, "names"):
                        raw_names = model_obj.names  # type: ignore[attr-defined]
            if isinstance(raw_names, dict):
                names.update(str(v) for _, v in sorted(raw_names.items()))
            elif isinstance(raw_names, (list, tuple)):
                names.update(str(v) for v in raw_names)
        except Exception:
            names = set()

        # Fallback: cargar mediante la API YOLO solo si aÃºn no tenemos nombres
        if not names:
            try:
                model = YOLO(path)
                raw_names = getattr(model, "names", None)
                if isinstance(raw_names, dict):
                    names.update(str(v) for _, v in sorted(raw_names.items()))
                elif isinstance(raw_names, (list, tuple)):
                    names.update(str(v) for v in raw_names)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pueden leer las clases del modelo %s: %s", path, exc)

        sorted_names = tuple(sorted(names))
        self._model_names_cache[path] = (mtime, sorted_names)
        return set(sorted_names)

    def _build_current_snapshot_payload(self) -> dict | None:
        if self.snapshot_writer is None:
            return None
        try:
            return self.snapshot_writer.build_snapshot_payload()
        except Exception as exc:
            LOGGER.debug("No se pudo generar snapshot actual: %s", exc)
            return None

    def _start_plc_push_thread(self) -> None:
        if self._plc_service is None:
            return
        if self._plc_push_queue is None:
            self._plc_push_queue = queue.Queue(maxsize=1)
        if self._plc_push_thread is not None and self._plc_push_thread.is_alive():
            return
        self._plc_push_stop.clear()
        self._plc_push_thread = threading.Thread(target=self._plc_push_loop, daemon=True)
        self._plc_push_thread.start()

    def _plc_push_loop(self) -> None:
        while not self._plc_push_stop.is_set():
            if self._plc_push_queue is None:
                time.sleep(0.05)
                continue
            try:
                payload = self._plc_push_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._plc_service is None:
                continue
            try:
                self._plc_service.push_snapshot(payload, persist=False)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("No se pudo enviar snapshot al servicio PLC: %s", exc)

    def _push_snapshot_to_plc(self, payload: dict | None) -> None:
        if self._plc_service is None:
            return
        self._start_plc_push_thread()
        if self._plc_push_queue is None:
            return
        try:
            self._plc_push_queue.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._plc_push_queue.get_nowait()
            except Exception:
                pass
            try:
                self._plc_push_queue.put_nowait(payload)
            except Exception:
                pass

    def _on_heartbeat_message(self, payload: dict, addr: tuple[str, int]) -> None:
        text = payload.get("text") or ""
        sender = payload.get("sender") or f"{addr[0]}"
        ts_value = payload.get("ts")
        try:
            ts = float(ts_value)
            stamp = time.strftime("%H:%M:%S", time.localtime(ts))
        except (TypeError, ValueError):
            stamp = time.strftime("%H:%M:%S")
        display = f"{sender} @ {addr[0]}:{addr[1]} {stamp}\n{text}"
        self._set_textarea(self._hb_remote_box, display)

        try:
            info = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            info = None
        if isinstance(info, dict):
            beat = info.get("beat") or info.get("role") or "visor"
            state = info.get("state") or "desconocido"
            since = info.get("state_since")
            reason = info.get("state_reason") or ""
            extra = []
            if since is not None:
                extra.append(f"{since:.0f}s")
            if reason:
                extra.append(reason)
            summary = " ".join(filter(None, extra))
            LOGGER.info("Heartbeat %s: estado visor=%s %s", beat, state, summary)

    def _stop_heartbeat(self) -> None:
        if self._hb_job is not None:
            try:
                self.root.after_cancel(self._hb_job)
            except Exception:
                pass
            self._hb_job = None
        if self._hb_bridge is not None:
            try:
                self._hb_bridge.send_message("DET-STOP")
            except Exception:
                pass
            self._hb_bridge.stop()
            self._hb_bridge = None

    # ------------------------- Snapshot JSON -------------------------
    def _schedule_snapshot_flush(self) -> None:
        if self.snapshot_writer is None:
            return
        self._reschedule_snapshot_flush()

    def _reschedule_snapshot_flush(self, interval_ms: int | None = None) -> None:
        if self.snapshot_writer is None:
            return
        if interval_ms is None:
            interval_ms = int(max(SNAPSHOT_MIN_WRITE_INTERVAL_MS, self.snapshot_writer.write_interval_ms))
        if self._snapshot_flush_job is not None:
            try:
                self.root.after_cancel(self._snapshot_flush_job)
            except Exception:
                pass
            self._snapshot_flush_job = None
        try:
            self._snapshot_flush_job = self.root.after(interval_ms, self._snapshot_flush_tick)
        except Exception:
            self._snapshot_flush_job = None

    def _snapshot_flush_tick(self) -> None:
        self._snapshot_flush_job = None
        writer = self.snapshot_writer
        if writer is not None:
            wrote = False
            try:
                wrote = writer.flush()
            except Exception as exc:
                LOGGER.debug("Snapshot JSON: flush fallÃ³ (%s)", exc)
            finally:
                if writer is not None and not wrote:
                    self._ensure_snapshot_presence(writer)
                if self.snapshot_writer is not None:
                    self._reschedule_snapshot_flush()

    def _stop_snapshot_flush(self) -> None:
        if self._snapshot_flush_job is not None:
            try:
                self.root.after_cancel(self._snapshot_flush_job)
            except Exception:
                pass
            self._snapshot_flush_job = None
        if self.snapshot_writer is not None:
            try:
                self.snapshot_writer.flush(force=True)
            except Exception:
                pass
        self._push_snapshot_to_plc(self._build_current_snapshot_payload())

    def _write_initial_snapshot(self) -> None:
        writer = self.snapshot_writer
        if writer is None:
            return
        try:
            payload = writer.build_snapshot_payload()
        except Exception as exc:
            LOGGER.debug("Snapshot inicial: no se pudo construir payload (%s)", exc)
            payload = None
        try:
            writer.flush(force=True)
            LOGGER.info("Snapshot inicial disponible en %s", self.snapshot_path)
        except Exception as exc:
            LOGGER.warning("Snapshot inicial: no se pudo escribir archivo (%s)", exc)
            return
        if payload is not None:
            self._push_snapshot_to_plc(payload)

    def _ensure_snapshot_presence(self, writer: SnapshotAggregator) -> None:
        path = getattr(writer, "path", None) or self.snapshot_path
        try:
            exists = bool(path) and os.path.isfile(path)
        except Exception:
            exists = True
        if exists:
            return
        try:
            payload = writer.build_snapshot_payload()
        except Exception as exc:
            LOGGER.debug("Snapshot JSON: no se pudo construir payload al recrear archivo (%s)", exc)
            payload = None
        try:
            writer.flush(force=True)
            LOGGER.warning("Snapshot JSON recreado tras detectar ausencia (%s)", path)
        except Exception as exc:
            LOGGER.error("Snapshot JSON: no se pudo recrear archivo en %s (%s)", path, exc)
            return
        if payload is not None:
            self._push_snapshot_to_plc(payload)


    def _update_sector_filter_cache(self) -> None:
        """Actualiza el conjunto de sectores excluidos (0-based) para filtrado rÃ¡pido."""
        try:
            sect_cfg = self.config.get("sectores", {})
            if not isinstance(sect_cfg, dict): sect_cfg = {}
            excluidos = sect_cfg.get("excluidos", [])
            s0 = set()
            if isinstance(excluidos, list):
                for x in excluidos:
                    if isinstance(x, int) and x > 0:
                        s0.add(x - 1)
            self._cached_excluded_sectors_0based = s0
            
            # Asegurar sincronizaciÃ³n de estado 'enabled'
            if self.sectorizador:
                 legacy = self.config.get("sectorizador", {})
                 # Forzar re-lectura si viene del panel legacy o config
                 pass
        except Exception:
            self._cached_excluded_sectors_0based = set()

    def _is_detection_allowed(self, det: dict) -> bool:
        """
        Retorna True si la detecciÃ³n debe mantenerse (pasa filtros de sector y clase).
        Centraliza la lÃ³gica de filtrado para Drawing, Snapshot y PLC.
        """
        # 1. Filtro por sectores excluidos
        sector_id = det.get("sector")
        
        # Si no se ha inicializado la cachÃ© (caso raro al inicio), actualizarla
        if not hasattr(self, "_cached_excluded_sectors_0based"):
            self._update_sector_filter_cache()
            
        if sector_id is not None and sector_id in self._cached_excluded_sectors_0based:
            return False
            
        # 2. Filtro por restricciones (delegado al sectorizador)
        if self.sectorizador and self.sectorizador.config_restricciones.enabled:
            # Asegurar normalizaciÃ³n antes de llamar
            if not self.sectorizador.filtrar_deteccion(det):
                return False
                
        return True

    def _snapshot_register_results(self, frame: np.ndarray, result1, result2) -> None:
        if self.snapshot_writer is None or frame is None:
            return
        try:
            h, w = frame.shape[:2]
        except Exception:
            return
        detections: list[dict] = []
        frame_ts = time.time()
        if h > 0 and w > 0:
            frame_shape = (int(h), int(w))
            detections.extend(self._detections_from_result(result1, "M1", frame_shape))
            detections.extend(self._detections_from_result(result2, "M2", frame_shape))
        
        # NUEVO: Filtrar detecciones con lÃ³gica centralizada
        if self.sectorizador is not None and detections:
            filtered = []
            # Asegurar que la cachÃ© estÃ© actualizada (por si no hubo cambios de config recientes)
            if not hasattr(self, "_cached_excluded_sectors_0based"):
                self._update_sector_filter_cache()
                
            for det in detections:
                if self._is_detection_allowed(det):
                    filtered.append(det)
            detections = filtered
        
        self.snapshot_writer.update(
            frame_ts=frame_ts,
            detections=detections,
            avg_fps=self._snapshot_last_avg_fps,
            inst_fps=self._snapshot_last_inst_fps,
        )
        now = time.time()
        if (now - self._last_plc_push_wall) >= self._plc_push_interval_sec:
            self._last_plc_push_wall = now
            self._push_snapshot_to_plc(self._build_current_snapshot_payload())

    def _on_snapshot_timing_change(self) -> None:
        if self._loading_config:
            return
        try:
            write_ms = max(SNAPSHOT_MIN_WRITE_INTERVAL_MS, int(self.snapshot_write_interval_ms.get()))
        except Exception:
            write_ms = SNAPSHOT_DEFAULT_WRITE_INTERVAL_MS
            self.snapshot_write_interval_ms.set(write_ms)
        try:
            clean_sec = max(0.0, float(self.snapshot_clean_interval_sec.get()))
        except Exception:
            clean_sec = SNAPSHOT_DEFAULT_CLEAN_INTERVAL_SEC
            self.snapshot_clean_interval_sec.set(clean_sec)
        if self.snapshot_writer is not None:
            self.snapshot_writer.configure_timing(write_interval_ms=write_ms, clean_every_sec=clean_sec)
        self._reschedule_snapshot_flush(write_ms)

    def _on_sector_config_change(self, *, save_config: bool = True) -> None:
        """Callback cuando cambia la configuraciÃ³n del sectorizador."""
        if self._loading_config or self.sectorizador is None:
            return
        
        # Obtener valores de las clases de borde (limpiar "(Ninguno)")
        borde_sup = self.sector_borde_sup.get()
        borde_inf = self.sector_borde_inf.get()
        borde_izq = self.sector_borde_izq.get()
        borde_der = self.sector_borde_der.get()
        
        if borde_sup == "(Ninguno)" or borde_sup == "(Auto)":
            borde_sup = ""
        if borde_inf == "(Ninguno)" or borde_inf == "(Auto)":
            borde_inf = ""
        if borde_izq == "(Ninguno)" or borde_izq == "(Auto)":
            borde_izq = ""
        if borde_der == "(Ninguno)" or borde_der == "(Auto)":
            borde_der = ""
        
        # Actualizar vista cenital del panel si existe
        if hasattr(self, "_sector_panel") and self._sector_panel is not None:
            try:
                self._sector_panel.set_border_classes(
                    top=borde_sup, bottom=borde_inf, left=borde_izq, right=borde_der
                )
            except Exception:
                pass
        
        # Configurar bordes
        self.sectorizador.set_config_bordes(
            clase_superior=borde_sup,
            clase_inferior=borde_inf,
            clase_izquierdo=borde_izq,
            clase_derecho=borde_der
        )
        
        # Configurar sectores
        try:
            num_vert = max(1, int(self.sector_num_vert.get()))
        except Exception:
            num_vert = 1
        try:
            num_horiz = max(1, int(self.sector_num_horiz.get()))
        except Exception:
            num_horiz = 1
        
        # Obtener parÃ¡metros de perspectiva
        try:
            smooth_alpha = max(0.01, min(1.0, float(self.sector_smooth_alpha.get())))
        except Exception:
            smooth_alpha = 0.15
        try:
            max_jump = max(5.0, float(self.sector_max_jump.get()))
        except Exception:
            max_jump = 50.0
        try:
            inset = max(0, int(self.sector_inset.get()))
        except Exception:
            inset = 4
        try:
            opacidad = max(0.0, min(1.0, float(self.sector_opacidad_lineas.get())))
        except:
            opacidad = 1.0
        try:
            grosor = max(1, int(self.sector_grosor_lineas.get()))
        except:
            grosor = 1
        try:
            roi_quant = max(1, int(self.sector_roi_quant_step.get()))
        except Exception:
            roi_quant = 2
        try:
            line_quant = max(1, int(self.sector_line_quant_step.get()))
        except Exception:
            line_quant = 2
        
        self.sectorizador.set_config_sectores(
            modo=self.sector_modo.get(),
            num_verticales=num_vert,
            num_horizontales=num_horiz,
            mostrar_etiquetas=bool(self.sector_mostrar_etiquetas.get()),
            mostrar_sectorizacion=bool(self.sector_mostrar.get()),
            mostrar_borde_banda=bool(self.sector_mostrar_borde_banda.get()),
            opacidad_lineas=opacidad,
            grosor_lineas=grosor,
            use_perspective=bool(self.sector_use_perspective.get()),
            use_border_masks=bool(self.sector_use_masks.get()),  # Legacy/Avanzado
            modo_delimitacion=self.sector_modo_delimitacion.get(),
            smooth_alpha=smooth_alpha,
            max_corner_jump_px=max_jump,
            inset_px=inset,
            debug_overlay=bool(self.sector_debug_overlay.get()),
            comportamiento_fallo=self.sector_comportamiento_fallo.get(),
            # NUEVO: Bordes curvos
            curved_edges_enabled=bool(self.sector_curved_enabled.get()),
            curved_bins_vertical=max(0, int(self.sector_curved_bins_vert.get())),
            curved_bins_horizontal=max(0, int(self.sector_curved_bins_horiz.get())),
            # NUEVO: Padding
            padding_top_px=max(0, int(self.sector_padding_top.get())),
            padding_bottom_px=max(0, int(self.sector_padding_bottom.get())),
            padding_left_px=max(0, int(self.sector_padding_left.get())),
            padding_right_px=max(0, int(self.sector_padding_right.get())),
            roi_quant_step_px=roi_quant,
            line_quant_step_px=line_quant
        )
        
        # Actualizar validaciÃ³n UI
        self._validar_delimitacion()

        if getattr(self, "_sector_panel", None) is not None:
            try:
                panel_data = self._sector_panel.get_config()
            except Exception:
                panel_data = None
            if isinstance(panel_data, dict):
                if not isinstance(self.config.get("sectores"), dict):
                    self.config["sectores"] = {}
                self.config["sectores"]["excluidos"] = panel_data.get("excluidos", [])
                self.config["sectores"]["sensibilidades"] = panel_data.get("sensibilidades", {})
                self.config["sectores"]["ajustes_locales"] = panel_data.get("sensibilidades", {})
                if "restricciones_clase" in panel_data:
                    self.config["sectores"]["restricciones_clase"] = panel_data["restricciones_clase"]
                    if isinstance(panel_data["restricciones_clase"], dict):
                        self._restricciones_por_clase = dict(panel_data["restricciones_clase"])
                        self._apply_restricciones_to_sectorizador()
                        self._actualizar_resumen_restricciones()
        
        # Actualizar cachÃ© de filtrado (excluidos 0-based)
        self._update_sector_filter_cache()

        if save_config:
            self._save_config()

    def _init_thick_pulse_timer(self) -> None:
        """Inicia el temporizador para el pulso de grosor cada 30 segundos."""
        try:
            # Programar el primer pulso en 30 segundos
            self.root.after(30000, self._thick_pulse_start)
        except Exception as e:
            LOGGER.warning("No se pudo iniciar el temporizador de pulso de grosor: %s", e)

    def _thick_pulse_start(self) -> None:
        """Sube el grosor un punto (mÃ¡x 5) durante 1 segundo."""
        try:
            if not hasattr(self, "sector_grosor_lineas"):
                # Re-programar si el objeto no estÃ¡ listo
                self.root.after(30000, self._thick_pulse_start)
                return

            original_val = int(self.sector_grosor_lineas.get())
            
            # Subir grosor un punto solo si es menor que 5
            if original_val < 5:
                self.sector_grosor_lineas.set(original_val + 1)
                # Aplicar visualmente sin guardar config
                self._on_sector_config_change(save_config=False)
                # Programar vuelta al valor original en 1 segundo
                self.root.after(1000, lambda: self._thick_pulse_end(original_val))
            
            # Programar el siguiente inicio en 30 segundos
            self.root.after(30000, self._thick_pulse_start)
        except Exception as e:
            LOGGER.warning("Error en inicio de pulso de grosor: %s", e)
            # Intentar re-programar a pesar del error
            try: self.root.after(30000, self._thick_pulse_start)
            except: pass

    def _thick_pulse_end(self, original_val: int) -> None:
        """Vuelve al grosor original despuÃ©s del pulso."""
        try:
            if hasattr(self, "sector_grosor_lineas"):
                self.sector_grosor_lineas.set(original_val)
                self._on_sector_config_change(save_config=False)
        except Exception as e:
            LOGGER.warning("Error al finalizar pulso de grosor: %s", e)

    def _actualizar_visibilidad_spinners(self):
        """Muestra/oculta spinners segÃºn el modo de sectorizaciÃ³n."""
        if not hasattr(self, "_frm_spinners_sector"): return
        
        modo = self.sector_modo.get()
        # Limpiar frame
        for w in self._frm_spinners_sector.winfo_children():
            w.grid_forget()
        
        if modo == "vertical":
            self._lbl_sect_vert.grid(row=0, column=0, sticky="w", padx=(0, 4))
            if getattr(self, "_icon_sect_vert", None) is not None:
                self._icon_sect_vert.grid(row=0, column=1, sticky="w")
            self._spin_sect_vert.grid(row=0, column=2, sticky="w")
        elif modo == "horizontal":
            self._lbl_sect_horiz.grid(row=0, column=0, sticky="w", padx=(0, 4))
            if getattr(self, "_icon_sect_horiz", None) is not None:
                self._icon_sect_horiz.grid(row=0, column=1, sticky="w")
            self._spin_sect_horiz.grid(row=0, column=2, sticky="w")
        else:  # rejilla
            self._lbl_sect_vert.grid(row=0, column=0, sticky="w", padx=(0, 4))
            if getattr(self, "_icon_sect_vert", None) is not None:
                self._icon_sect_vert.grid(row=0, column=1, sticky="w")
            self._spin_sect_vert.grid(row=0, column=2, sticky="w")
            self._lbl_sect_horiz.grid(row=0, column=3, sticky="w", padx=(16, 4))
            if getattr(self, "_icon_sect_horiz", None) is not None:
                self._icon_sect_horiz.grid(row=0, column=4, sticky="w")
            self._spin_sect_horiz.grid(row=0, column=5, sticky="w")
        
        # Resaltar botÃ³n activo
        for btn, m in [(getattr(self, "_btn_modo_vert", None), "vertical"), 
                       (getattr(self, "_btn_modo_horiz", None), "horizontal"),
                       (getattr(self, "_btn_modo_rejilla", None), "rejilla")]:
            if btn:
                if m == modo:
                    btn.state(["pressed"])
                else:
                    btn.state(["!pressed"])

    def _validar_delimitacion(self):
        """Valida la configuraciÃ³n de delimitaciÃ³n y muestra warnings."""
        lbl = getattr(self, "_lbl_warning_delim", None)
        if lbl is None:
            return
        try:
            if not lbl.winfo_exists():
                self._lbl_warning_delim = None
                return
        except tk.TclError:
            self._lbl_warning_delim = None
            return
        
        sup = self.sector_borde_sup.get()
        inf = self.sector_borde_inf.get()
        izq = self.sector_borde_izq.get()
        der = self.sector_borde_der.get()
        
        sin_config = all(v in ("", "(Ninguno)", "(Auto)") for v in [sup, inf, izq, der])
        
        if sin_config:
            try:
                lbl.configure(
                    text="Sin delimitacion: la sectorizacion no podra calcular la banda.")
            except tk.TclError:
                self._lbl_warning_delim = None
        else:
            try:
                lbl.configure(text="")
            except tk.TclError:
                self._lbl_warning_delim = None

    def _probar_delimitacion(self):
        """Activa overlay de depuraciÃ³n por 3s y muestra estado."""
        self.sector_debug_overlay.set(True)
        self._on_sector_config_change()
        lbl = getattr(self, "_lbl_estado_delim", None)
        if lbl is not None:
            try:
                if lbl.winfo_exists():
                    lbl.configure(text="Probando...", foreground="#0066cc")
            except tk.TclError:
                self._lbl_estado_delim = None
        
        def _check_resultado():
            roi = getattr(self.sectorizador, '_roi_actual', None)
            valid = roi.valido if roi else False
            
            lbl_local = getattr(self, "_lbl_estado_delim", None)
            if lbl_local is not None:
                try:
                    if lbl_local.winfo_exists():
                        if valid:
                            if roi.is_quad:
                                lbl_local.configure(text="? OK (perspectiva)", foreground="#228b22")
                            else:
                                lbl_local.configure(text="OK (rectangulo)", foreground="#228b22")
                        else:
                            lbl_local.configure(text="? Sin bordes detectados", foreground="#cc0000")
                except tk.TclError:
                    self._lbl_estado_delim = None
            
            self.sector_debug_overlay.set(False)
            self._on_sector_config_change()
        
        self.root.after(3000, _check_resultado)

    def _on_estabilidad_change(self):
        """Mapea nivel de estabilizaciÃ³n a parÃ¡metros tÃ©cnicos."""
        nivel = self.sector_estabilidad.get()
        mapeo = {
            "Baja": (0.3, 80.0),    # alpha alto = menos suavizado, salto alto = mÃ¡s tolerante
            "Media": (0.15, 50.0),  # valores por defecto
            "Alta": (0.05, 25.0)    # alpha bajo = mÃ¡s suavizado, salto bajo = mÃ¡s restrictivo
        }
        alpha, salto = mapeo.get(nivel, (0.15, 50.0))
        self.sector_smooth_alpha.set(alpha)
        self.sector_max_jump.set(salto)
        self._on_sector_config_change()

    def _guardar_preset_sector(self):
        """Guarda la configuraciÃ³n actual de sectores como preset."""
        nombre = self.preset_sector_var.get().strip()
        if not nombre:
            messagebox.showwarning("Aviso", "Introduce un nombre para el preset.")
            return
        
        data = {
            "modo": self.sector_modo.get(),
            "num_vert": self.sector_num_vert.get(),
            "num_horiz": self.sector_num_horiz.get(),
            "borde_sup": self.sector_borde_sup.get(),
            "borde_inf": self.sector_borde_inf.get(),
            "borde_izq": self.sector_borde_izq.get(),
            "borde_der": self.sector_borde_der.get(),
            "use_perspective": self.sector_use_perspective.get(),
            "estabilidad": self.sector_estabilidad.get(),
            "comportamiento_fallo": self.sector_comportamiento_fallo.get(),
            "modo_delimitacion": self.sector_modo_delimitacion.get(),
            "roi_quant_step_px": self.sector_roi_quant_step.get(),
            "line_quant_step_px": self.sector_line_quant_step.get()
        }
        
        if "presets" not in self.config:
            self.config["presets"] = {}
        if "sectores" not in self.config["presets"]:
            self.config["presets"]["sectores"] = {}
        
        self.config["presets"]["sectores"][nombre] = data
        self._save_config()
        
        # Actualizar lista
        if hasattr(self, "combo_preset_sector"):
            self.combo_preset_sector.config(values=sorted(self.config["presets"]["sectores"].keys()))
        messagebox.showinfo("Exito", f"Preset '{nombre}' guardado.")

    def _cargar_preset_sector(self):
        """Carga un preset de sectores."""
        nombre = self.preset_sector_var.get()
        if not nombre:
            return
        
        presets = self.config.get("presets", {}).get("sectores", {})
        data = presets.get(nombre)
        if not data:
            return
        
        self.sector_modo.set(data.get("modo", "vertical"))
        self.sector_num_vert.set(data.get("num_vert", 1))
        self.sector_num_horiz.set(data.get("num_horiz", 1))
        self.sector_borde_sup.set(data.get("borde_sup", "(Ninguno)"))
        self.sector_borde_inf.set(data.get("borde_inf", "(Ninguno)"))
        self.sector_borde_izq.set(data.get("borde_izq", "(Ninguno)"))
        self.sector_borde_der.set(data.get("borde_der", "(Ninguno)"))
        self.sector_use_perspective.set(data.get("use_perspective", True))
        self.sector_estabilidad.set(data.get("estabilidad", "Media"))
        
        if "comportamiento_fallo" in data:
            self.sector_comportamiento_fallo.set(data["comportamiento_fallo"])
        if "modo_delimitacion" in data:
            self.sector_modo_delimitacion.set(data["modo_delimitacion"])
        if "roi_quant_step_px" in data:
            self.sector_roi_quant_step.set(max(1, int(data["roi_quant_step_px"])))
        if "line_quant_step_px" in data:
            self.sector_line_quant_step.set(max(1, int(data["line_quant_step_px"])))
        
        self._actualizar_visibilidad_spinners()
        self._on_estabilidad_change()
        self._set_status(f"Preset '{nombre}' aplicado.")

    def _restablecer_sector_defaults(self):
        """Restaura valores por defecto de sectorizaciÃ³n."""
        self.sector_modo.set("vertical")
        self.sector_num_vert.set(1)
        self.sector_num_horiz.set(1)
        self.sector_use_perspective.set(True)
        self.sector_estabilidad.set("Media")
        self.sector_modo_delimitacion.set("Auto")
        self.sector_comportamiento_fallo.set("Congelar ultima valida")
        self.sector_roi_quant_step.set(2)
        self.sector_line_quant_step.set(2)
        self._actualizar_visibilidad_spinners()
        self._on_estabilidad_change()
        self._set_status("Valores de sectorizacion restablecidos.")

    def _on_restricciones_toggle(self) -> None:
        """Callback cuando se activa/desactiva el checkbox de restricciones."""
        if self.sectorizador is not None:
            self.sectorizador.config_restricciones.enabled = self.restricciones_enabled.get()
        self._actualizar_resumen_restricciones()
        self._save_config()

    def _actualizar_resumen_restricciones(self) -> None:
        """Actualiza el label de resumen de restricciones."""
        if not hasattr(self, '_lbl_restricciones_resumen'):
            return
        
        if not self.restricciones_enabled.get():
            self._lbl_restricciones_resumen.config(text="(deshabilitado)")
            return
        
        n_clases = len([c for c, cfg in self._restricciones_por_clase.items() 
                        if cfg.get("modo", "sin_restriccion") != "sin_restriccion"])
        if n_clases == 0:
            self._lbl_restricciones_resumen.config(text="Ninguna restriccion configurada")
        else:
            self._lbl_restricciones_resumen.config(text=f"{n_clases} clase(s) con restriccion")

    def _apply_restricciones_to_sectorizador(self) -> None:
        """Aplica las restricciones configuradas al sectorizador."""
        if self.sectorizador is None:
            return
        
        from sectorizador import RestriccionClase
        self.sectorizador.config_restricciones.enabled = self.restricciones_enabled.get()
        self.sectorizador.config_restricciones.por_clase.clear()
        
        for clase, cfg in self._restricciones_por_clase.items():
            modo = cfg.get("modo", "sin_restriccion")
            sectores = cfg.get("sectores", [])
            self.sectorizador.config_restricciones.por_clase[clase] = RestriccionClase(
                modo=modo, 
                sectores=sectores
            )

    def _open_restricciones_dialog(self) -> None:
        """Abre el diÃ¡logo de configuraciÃ³n de restricciones por clase."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Restricciones de Deteccion por Clase")
        dlg.geometry("550x400")
        dlg.transient(self.root)
        dlg.grab_set()
        
        # Header
        header = ttk.Frame(dlg, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Configura que clases se detectan dentro/fuera de la malla",
                  font=("Segoe UI", 10)).pack(anchor="w")
        
        # Scrollable frame for classes
        container = ttk.Frame(dlg)
        container.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Headers
        ttk.Label(scrollable, text="Clase", font=("Segoe UI", 9, "bold"), width=18).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(scrollable, text="Restriccion", font=("Segoe UI", 9, "bold"), width=20).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(scrollable, text="Sectores", font=("Segoe UI", 9, "bold"), width=15).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # Get classes from class_cfg
        clases = sorted(self.class_cfg.keys()) if self.class_cfg else []
        
        # Store widgets for later access
        widgets: dict[str, tuple] = {}
        modos = ["sin_restriccion", "solo_malla", "solo_fuera_malla", "solo_sectores"]
        modos_display = {
            "sin_restriccion": "Sin restriccion",
            "solo_malla": "Solo dentro malla",
            "solo_fuera_malla": "Solo fuera malla",
            "solo_sectores": "Solo en sectores..."
        }
        
        for row, clase in enumerate(clases, start=1):
            ttk.Label(scrollable, text=clase).grid(row=row, column=0, padx=5, pady=3, sticky="w")
            
            # Get current config for this class
            cfg = self._restricciones_por_clase.get(clase, {"modo": "sin_restriccion", "sectores": []})
            
            modo_var = tk.StringVar(value=cfg.get("modo", "sin_restriccion"))
            combo = ttk.Combobox(scrollable, textvariable=modo_var, values=modos, state="readonly", width=18)
            combo.grid(row=row, column=1, padx=5, pady=3, sticky="w")
            
            sectores_var = tk.StringVar(value=",".join(str(s+1) for s in cfg.get("sectores", [])))
            entry = ttk.Entry(scrollable, textvariable=sectores_var, width=12)
            entry.grid(row=row, column=2, padx=5, pady=3, sticky="w")
            
            # Enable/disable sectores entry based on mode
            def update_entry_state(var=modo_var, ent=entry):
                ent.configure(state="normal" if var.get() == "solo_sectores" else "disabled")
            
            modo_var.trace_add("write", lambda *args, v=modo_var, e=entry: update_entry_state(v, e))
            update_entry_state(modo_var, entry)
            
            widgets[clase] = (modo_var, sectores_var)
        
        # Buttons
        btn_frame = ttk.Frame(dlg, padding=10)
        btn_frame.pack(fill="x")
        
        def on_apply():
            for clase, (modo_var, sectores_var) in widgets.items():
                modo = modo_var.get()
                sectores = []
                if modo == "solo_sectores":
                    try:
                        # Parse "1,2,3" to [0,1,2] (convert 1-based UI to 0-based internal)
                        sectores = [int(s.strip()) - 1 for s in sectores_var.get().split(",") if s.strip().isdigit()]
                    except:
                        pass
                self._restricciones_por_clase[clase] = {"modo": modo, "sectores": sectores}
            
            self._apply_restricciones_to_sectorizador()
            self._actualizar_resumen_restricciones()
            self._save_config()
            dlg.destroy()
        
        def on_cancel():
            dlg.destroy()
        
        ttk.Button(btn_frame, text="Aplicar", command=on_apply, width=12).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=on_cancel, width=12).pack(side="right", padx=5)

    def _extraer_detecciones_para_sectorizador(self, result1, result2) -> list[dict]:
        """
        Extrae detecciones de los resultados para el sectorizador.
        Incluye contornos de mascaras para clases de borde (para perspectiva).
        """
        detecciones = []

        if self.sectorizador is None:
            return detecciones

        # Obtener clases de borde configuradas (con roles)
        cb = self.sectorizador.config_bordes
        class_roles: dict[str, set[str]] = {}
        def _add_role(class_name: str | None, role: str) -> None:
            if class_name:
                class_roles.setdefault(class_name, set()).add(role)
        _add_role(cb.clase_superior, "top")
        _add_role(cb.clase_inferior, "bottom")
        _add_role(cb.clase_izquierdo, "left")
        _add_role(cb.clase_derecho, "right")

        if not class_roles:
            return detecciones

        use_border_masks = bool(self.sectorizador.config_sectores.use_border_masks)

        for result, tag in [(result1, "M1"), (result2, "M2")]:
            if result is None:
                continue
            boxes_xyxy, confs, clss, names = self._result_arrays(result)
            if boxes_xyxy is None or clss is None:
                continue

            clss_arr = clss.astype(int, copy=False)
            name_to_id = None
            if isinstance(names, dict) and names:
                try:
                    name_to_id = {str(v): int(k) for k, v in names.items()}
                except Exception:
                    name_to_id = None

            selected_indices: set[int] = set()
            primary_mask_indices: set[int] = set()
            if name_to_id:
                for cls_name, roles in class_roles.items():
                    cls_id = name_to_id.get(cls_name)
                    if cls_id is None:
                        continue
                    idxs = np.where(clss_arr == cls_id)[0]
                    if idxs.size == 0:
                        continue

                    if boxes_xyxy is not None:
                        x1 = boxes_xyxy[idxs, 0]
                        y1 = boxes_xyxy[idxs, 1]
                        x2 = boxes_xyxy[idxs, 2]
                        y2 = boxes_xyxy[idxs, 3]
                        if "left" in roles:
                            sel = int(idxs[int(np.argmin(x1))])
                            selected_indices.add(sel)
                            primary_mask_indices.add(sel)
                        if "right" in roles:
                            sel = int(idxs[int(np.argmax(x2))])
                            selected_indices.add(sel)
                            primary_mask_indices.add(sel)
                        if "top" in roles:
                            sel = int(idxs[int(np.argmin(y1))])
                            selected_indices.add(sel)
                            primary_mask_indices.add(sel)
                        if "bottom" in roles:
                            sel = int(idxs[int(np.argmax(y2))])
                            selected_indices.add(sel)
                            primary_mask_indices.add(sel)
                    else:
                        if confs is not None and idxs.size:
                            sel = int(idxs[int(np.argmax(confs[idxs]))])
                        else:
                            sel = int(idxs[0])
                        selected_indices.add(sel)
                        primary_mask_indices.add(sel)

            if not selected_indices:
                continue

            masks_xy = None
            mask_arrays = None
            need_masks = use_border_masks and len(primary_mask_indices) > 0
            if need_masks:
                try:
                    masks_xy = result.masks_xy()
                except Exception:
                    masks_xy = None
                try:
                    if hasattr(result, "masks_np") and callable(result.masks_np):
                        mask_arrays = result.masks_np()
                    elif getattr(result, "masks", None) is not None:
                        raw_masks = getattr(result.masks, "data", None)
                        if raw_masks is not None:
                            try:
                                mask_arrays = raw_masks.detach().cpu().numpy()
                            except Exception:
                                mask_arrays = raw_masks.numpy() if hasattr(raw_masks, "numpy") else None
                except Exception:
                    mask_arrays = None

            for i in sorted(selected_indices):
                cls_id = int(clss[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                bbox = boxes_xyxy[i].tolist() if boxes_xyxy is not None else []
                conf = float(confs[i]) if confs is not None and i < len(confs) else 0.0

                det = {
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "bbox": bbox,
                    "conf": conf,
                    "model": tag,
                }

                # Extraer contorno y edge_pts para clases de borde
                if use_border_masks and i in primary_mask_indices:
                    try:
                        contour = None
                        if masks_xy is not None and i < len(masks_xy):
                            contour = masks_xy[i]
                        if contour is not None:
                            det["contour"] = contour.reshape(-1, 2) if len(contour.shape) == 3 else contour

                        if mask_arrays is not None and i < len(mask_arrays):
                            mask = mask_arrays[i]
                            if mask is not None and mask.size > 0:
                                if isinstance(mask, np.ndarray) and mask.dtype == np.uint8:
                                    mask_bin = mask
                                else:
                                    mask_bin = (mask > 0.5).astype(np.uint8)
                                mask_h, mask_w = mask_bin.shape[:2]
                                x1p = 0
                                y1p = 0
                                x2p = mask_w
                                y2p = mask_h
                                if len(bbox) >= 4:
                                    try:
                                        x1f, y1f, x2f, y2f = [float(v) for v in bbox]
                                        if math.isfinite(x1f) and math.isfinite(y1f) and math.isfinite(x2f) and math.isfinite(y2f):
                                            pad = 2
                                            x1p = max(0, int(math.floor(x1f)) - pad)
                                            y1p = max(0, int(math.floor(y1f)) - pad)
                                            x2p = min(mask_w, int(math.ceil(x2f)) + pad)
                                            y2p = min(mask_h, int(math.ceil(y2f)) + pad)
                                            if x2p <= x1p or y2p <= y1p:
                                                x1p, y1p, x2p, y2p = 0, 0, mask_w, mask_h
                                    except Exception:
                                        x1p, y1p, x2p, y2p = 0, 0, mask_w, mask_h
                                mask_crop = mask_bin[y1p:y2p, x1p:x2p]
                                if contour is None and mask_crop.size > 0:
                                    mask_uint8 = mask_crop if mask_crop.dtype == np.uint8 else mask_crop.astype(np.uint8)
                                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        contour = max(contours, key=cv2.contourArea)
                                        if x1p or y1p:
                                            try:
                                                contour = contour + np.array([[[x1p, y1p]]], dtype=contour.dtype)
                                            except Exception:
                                                pass
                                        det["contour"] = contour.reshape(-1, 2) if len(contour.shape) == 3 else contour

                                edge_pts = self._calcular_edge_points_desde_mascara(mask_crop, [x1p, y1p, x2p, y2p])
                                if edge_pts:
                                    if x1p or y1p:
                                        for key, pts in edge_pts.items():
                                            try:
                                                if isinstance(pts, np.ndarray) and pts.size > 0:
                                                    pts[:, 0] += x1p
                                                    pts[:, 1] += y1p
                                            except Exception:
                                                continue
                                    det["edge_pts"] = edge_pts
                    except Exception:
                        pass

                detecciones.append(det)

        return detecciones

    def _calcular_edge_points_desde_mascara(self, mask_bin: np.ndarray, bbox: list) -> dict:
        """
        Calcula edge points (bordes interiores) directamente desde una mÃ¡scara binaria.
        V4.7: OPTIMIZADO con operaciones NumPy vectorizadas (50-100x mÃ¡s rÃ¡pido).
        """
        result = {}
        if mask_bin is None or mask_bin.size == 0:
            return result
        
        h, w = mask_bin.shape[:2]
        if len(bbox) < 4:
            return result
        
        try:
            # V4.7: VectorizaciÃ³n completa - evitar bucles Python
            mask_bool = mask_bin > 0
            
            # === Edge points por filas (para bordes laterales) ===
            # Encontrar filas con al menos un pixel activo
            row_has_data = np.any(mask_bool, axis=1)
            valid_rows = np.where(row_has_data)[0]
            
            if len(valid_rows) > 0:
                # Para cada fila vÃ¡lida, encontrar x_min y x_max
                # Usar argmax desde la izquierda y derecha
                left_pts = []
                right_pts = []
                
                # Submuestrear si hay muchas filas (optimizaciÃ³n para mÃ¡scaras grandes)
                step = max(1, len(valid_rows) // 200)  # MÃ¡ximo ~200 puntos
                sampled_rows = valid_rows[::step]
                
                for y in sampled_rows:
                    row = mask_bool[y, :]
                    nonzero = np.nonzero(row)[0]
                    if len(nonzero) > 0:
                        left_pts.append((float(nonzero[-1]), float(y)))
                        right_pts.append((float(nonzero[0]), float(y)))
                
                if left_pts:
                    result['left'] = np.array(left_pts)
                if right_pts:
                    result['right'] = np.array(right_pts)
            
            # === Edge points por columnas (para bordes horizontales) ===
            col_has_data = np.any(mask_bool, axis=0)
            valid_cols = np.where(col_has_data)[0]
            
            if len(valid_cols) > 0:
                top_pts = []
                bottom_pts = []
                
                # Submuestrear columnas tambiÃ©n
                step = max(1, len(valid_cols) // 200)
                sampled_cols = valid_cols[::step]
                
                for x in sampled_cols:
                    col = mask_bool[:, x]
                    nonzero = np.nonzero(col)[0]
                    if len(nonzero) > 0:
                        top_pts.append((float(x), float(nonzero[-1])))
                        bottom_pts.append((float(x), float(nonzero[0])))
                
                if top_pts:
                    result['top'] = np.array(top_pts)
                if bottom_pts:
                    result['bottom'] = np.array(bottom_pts)
        except Exception:
            pass
        
        return result

    def _result_arrays(self, result) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
        """Devuelve arrays numpy (xyxy, conf, cls) para un resultado, sea YOLO puro o _ResultAdapter."""
        boxes_xyxy = None
        confs = None
        clss = None
        names = {}
        if result is None:
            return boxes_xyxy, confs, clss, names

        try:
            if hasattr(result, "xyxy_np") and callable(result.xyxy_np):
                boxes_xyxy = result.xyxy_np()
            elif getattr(result, "boxes", None) is not None and hasattr(result.boxes, "xyxy"):
                boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        except Exception:
            boxes_xyxy = None

        try:
            if hasattr(result, "conf_np") and callable(result.conf_np):
                confs = result.conf_np()
            elif getattr(result, "boxes", None) is not None and hasattr(result.boxes, "conf"):
                confs = result.boxes.conf.detach().cpu().numpy()
        except Exception:
            confs = None

        try:
            if hasattr(result, "cls_np") and callable(result.cls_np):
                clss = result.cls_np().astype(int, copy=False)
            elif getattr(result, "boxes", None) is not None and hasattr(result.boxes, "cls"):
                clss = result.boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            clss = None

        try:
            if hasattr(result, "names") and isinstance(result.names, dict):
                names = result.names
        except Exception:
            names = {}

        return boxes_xyxy, confs, clss, names

    def _detections_from_result(
        self,
        result,
        model_tag: str,
        frame_shape: tuple[int, int],
    ) -> list[dict]:
        h, w = frame_shape
        if h <= 0 or w <= 0 or result is None:
            return []
        frame_area = float(w * h)
        boxes_xyxy, confs, clss, names = self._result_arrays(result)
        if boxes_xyxy is None or clss is None or len(boxes_xyxy) == 0:
            return []

        if not names:
            try:
                if model_tag == "M2" and getattr(self, "model2", None) is not None:
                    names = getattr(self.model2, "names", {})
                else:
                    names = getattr(self.model, "names", {})
            except Exception:
                names = {}

        need_area_px = self._area_any_enabled_for_model(model_tag)
        need_area_cm2 = need_area_px and self._calibration_valid and self.calibration is not None
        area_map_px, area_map_cm2 = self._get_area_maps_for_result(
            result,
            model_tag,
            frame_shape,
            require_px=need_area_px,
            require_cm2=need_area_cm2,
        )

        detections: list[dict] = []
        for idx, box in enumerate(boxes_xyxy):
            if idx >= len(clss):
                break
            x1, y1, x2, y2 = [float(v) for v in box]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            bbox_area_px = width * height
            cls_id = int(clss[idx])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            cfg = self.class_cfg.get(str(cls_name), {}) if hasattr(self, "class_cfg") else {}
            area_enabled = self._class_area_enabled(cfg, model_tag)
            if area_enabled:
                area_px = area_map_px.get(idx, bbox_area_px)
                rel_area = area_px / frame_area if frame_area > 0 else 0.0
            else:
                # Si la clase no tiene permiso de Ã¡rea, no generamos valores (evita triggers y ahorra trabajo).
                area_px = 0.0
                rel_area = 0.0
            conf_value = 0.0
            if confs is not None and idx < len(confs):
                try:
                    conf_value = float(confs[idx])
                except Exception:
                    conf_value = 0.0
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center_norm = (
                cx / w if w > 0 else None,
                cy / h if h > 0 else None,
            )
            area_cm2: float | None = area_map_cm2.get(idx) if area_enabled else None
            # Obtener sector de la detecciÃ³n si el sectorizador estÃ¡ disponible
            sector_id: int | None = None
            if self.sectorizador is not None:
                try:
                    sector_id = self.sectorizador.obtener_sector_para_punto(cx, cy)
                except Exception:
                    sector_id = None
            
            detections.append(
                {
                    "cls": cls_name,
                    "cls_id": cls_id,
                    "area": area_px,
                    "area_px": area_px,
                    "area_cm2": area_cm2,
                    "area_enabled": area_enabled,
                    "area_map": {
                        AREA_MODE_PX: area_px,
                        AREA_MODE_CM2: area_cm2,
                    },
                    "relative_area": rel_area,
                    "conf": conf_value,
                    "center_norm": center_norm,
                    "model": model_tag,
                    "frame_area": frame_area,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "is_critical": cls_name in SNAPSHOT_MAJOR_CLASSES
                    or (area_enabled and area_px >= SNAPSHOT_MAJOR_AREA_THRESHOLD),
                    "sector": sector_id,
                }
            )
        return detections

    def _get_area_maps_for_result(
        self,
        result,
        model_tag: str,
        frame_shape: tuple[int, int],
        *,
        require_px: bool,
        require_cm2: bool,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Construye o reutiliza un cachÃ© de Ã¡reas por detecciÃ³n para este resultado."""
        if not (require_px or require_cm2):
            return {}, {}

        frame_key = (int(frame_shape[0]), int(frame_shape[1]))
        result_id = id(result)
        if len(self._result_area_cache) > 200:
            for key, (cached_ref, cached_data) in list(self._result_area_cache.items()):
                if cached_ref is not None and cached_ref() is None:
                    self._result_area_cache.pop(key, None)

        cache_entry = self._result_area_cache.get(result_id)
        cache = None
        if cache_entry is not None:
            cached_ref, cached_data = cache_entry
            if cached_ref is not None:
                cached_target = cached_ref()
                if cached_target is result:
                    cache = cached_data
                else:
                    self._result_area_cache.pop(result_id, None)
            elif cached_data.get("strong_result") is result:
                cache = cached_data
            else:
                self._result_area_cache.pop(result_id, None)
        strong_result = None
        result_ref = None
        try:
            result_ref = weakref.ref(result)
        except TypeError:
            strong_result = result
        if (
            cache is None
            or cache.get("frame_shape") != frame_key
            or cache.get("model_tag") != model_tag
        ):
            cache = {
                "frame_shape": frame_key,
                "model_tag": model_tag,
                "area_px": {},
                "area_cm2": {},
                "px_ready": False,
                "cm2_ready": False,
                "strong_result": strong_result,
            }
            self._result_area_cache[result_id] = (result_ref, cache)

        need_loop = (require_px and not cache["px_ready"]) or (require_cm2 and not cache["cm2_ready"])
        if not need_loop:
            return cache["area_px"], cache["area_cm2"]

        xyxy, _, clss, _ = self._result_arrays(result)
        if xyxy is None or clss is None:
            if require_px:
                cache["px_ready"] = True
            if require_cm2:
                cache["cm2_ready"] = True
            return cache["area_px"], cache["area_cm2"]

        class_cache = getattr(self, "_class_cache", {})
        cache_entry = class_cache.get("M1" if model_tag == "M1" else "M2", {})
        area_enabled_by_cid = cache_entry.get("area_mode")
        if not isinstance(area_enabled_by_cid, np.ndarray):
            area_enabled_by_cid = None
        elif not bool(area_enabled_by_cid.any()):
            if require_px:
                cache["px_ready"] = True
            if require_cm2:
                cache["cm2_ready"] = True
            return cache["area_px"], cache["area_cm2"]

        mask_area_arr = None
        mask_arrays = None
        if require_px or (require_cm2 and self._calibration_valid):
            try:
                if hasattr(result, "masks_area_px") and callable(result.masks_area_px):
                    mask_area_arr = result.masks_area_px()
            except Exception:
                mask_area_arr = None
            if mask_area_arr is None:
                try:
                    if hasattr(result, "masks_np") and callable(result.masks_np):
                        mask_arrays = result.masks_np()
                    elif getattr(result, "masks", None) is not None:
                        mask_arrays = getattr(result.masks, "data_np", None)
                        if mask_arrays is None:
                            raw_masks = getattr(result.masks, "data", None)
                            if raw_masks is not None:
                                try:
                                    mask_arrays = raw_masks.detach().cpu().numpy()
                                except Exception:
                                    mask_arrays = raw_masks.numpy()
                except Exception:
                    mask_arrays = None

        area_px_map: dict[int, float] = {}
        area_cm2_map: dict[int, float] = {}
        for idx, box in enumerate(xyxy):
            if idx >= len(clss):
                break
            cid = int(clss[idx])
            if (
                area_enabled_by_cid is not None
                and cid < area_enabled_by_cid.size
                and not area_enabled_by_cid[cid]
            ):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area_px = width * height
            if mask_area_arr is not None and idx < len(mask_area_arr):
                try:
                    area_px = float(mask_area_arr[idx])
                except Exception:
                    pass
            elif mask_arrays is not None and idx < len(mask_arrays):
                try:
                    area_px = float(mask_arrays[idx].sum())
                except Exception:
                    try:
                        area_px = float(np.count_nonzero(mask_arrays[idx]))
                    except Exception:
                        pass
            if area_px > 0:
                area_px_map[idx] = area_px
            if require_cm2 and self._calibration_valid and self.calibration is not None:
                cy = (y1 + y2) / 2.0
                debug_details: dict[str, float] | None = None
                if not self._calibration_debug_logged:
                    debug_details = {}
                try:
                    area_val = bbox_area_cm2(
                        [x1, y1, x2, y2],
                        cy,
                        self.calibration,
                        area_px=area_px,
                        details=debug_details,
                    )
                    if debug_details is not None and not self._calibration_debug_logged:
                        LOGGER.info(
                            "CalibraciÐ˜n bbox=%s cy=%.1f x_cm=%.2f d=%.2fcm area_px=%.1f area_cm2=%.3f",
                            [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                            cy,
                            debug_details.get("x_cm", float("nan")),
                            debug_details.get("distance_cm", float("nan")),
                            debug_details.get("area_px", area_px),
                            area_val,
                        )
                        self._calibration_debug_logged = True
                except Exception as exc:
                    if not self._calibration_error_logged:
                        LOGGER.warning("Error al calcular ã‚±rea en cmåª½: %s. Se usarã‚± ã‚±rea en pãƒ´xeles.", exc)
                        self._calibration_error_logged = True
                    area_val = None
                if area_val is not None and area_val > 0:
                    area_cm2_map[idx] = area_val

        cache["area_px"] = area_px_map
        cache["px_ready"] = True
        if require_cm2:
            cache["area_cm2"] = area_cm2_map
            cache["cm2_ready"] = True
        return cache["area_px"], cache["area_cm2"]

    def _set_rtsp_state(self, state: str, reason: str | None = None) -> None:
        state = state.strip().lower()
        if state != self._rtsp_state:
            self._rtsp_state = state
            now_mono = time.monotonic()
            now_wall = time.time()
            self._rtsp_last_state_change = now_mono
            self._rtsp_last_state_change_wall = now_wall
            if state == "transmitiendo":
                self._rtsp_fail_notified = False
        self._rtsp_state_reason = (reason or "").strip()

    def _heartbeat_payload(self) -> dict:
        now_wall = time.time()
        since_state = max(0.0, now_wall - self._rtsp_last_state_change_wall)
        since_frame = None
        if self._rtsp_last_frame_wall > 0.0:
            since_frame = max(0.0, now_wall - self._rtsp_last_frame_wall)
        fail_age = None
        if self._rtsp_state != "transmitiendo" and self._rtsp_last_state_change_wall > 0:
            fail_age = max(0.0, now_wall - self._rtsp_last_state_change_wall)
        return {
            "role": "detector",
            "state": self._rtsp_state,
            "state_reason": self._rtsp_state_reason,
            "state_since": since_state,
            "since_last_frame": since_frame,
            "fail_age": fail_age,
            "frames_sent": int(self._rtsp_frames_sent),
            "rtsp_queue_size": self._rtsp_q.qsize() if self._rtsp_q is not None else 0,
            "fail_notified": bool(self._rtsp_fail_notified),
            "ts": now_wall,
        }

    def _check_rtsp_watchdogs(self) -> None:
        if self._current_source_kind != "RTSP":
            return
        now_wall = time.time()
        if self._rtsp_state != "transmitiendo":
            elapsed = now_wall - self._rtsp_last_state_change_wall
            if elapsed >= RTSP_FAIL_ALERT_SEC and not self._rtsp_fail_notified:
                self._rtsp_fail_notified = True
                mensaje = f"Sin senal RTSP desde hace {int(elapsed)}s. Reintentando..."
                LOGGER.warning(mensaje)
                self._async_warning("RTSP", mensaje)
        elif self._rtsp_fail_notified:
            # Se recuperÃ³ la seÃ±al, limpiamos bandera para futuros avisos
            self._rtsp_fail_notified = False

    def _build_default_rtsp_url(self) -> str:
        host = self._local_ip or "127.0.0.1"
        return f"rtsp://{host}:554/mystream"

    def _normalize_rtsp_out_url(self) -> None:
        current = (self.rtsp_out_url.get() or "").strip()
        if not current:
            self.rtsp_out_url.set(self._default_rtsp_url)
            return

        parsed = urlparse(current)
        if parsed.scheme.lower() != "rtsp":
            return

        host = parsed.hostname
        if host in {"127.0.0.1", "localhost"} and self._local_ip not in {"127.0.0.1", "localhost"}:
            userinfo = ""
            if parsed.username:
                userinfo = parsed.username
                if parsed.password is not None:
                    userinfo += f":{parsed.password}"
                userinfo += "@"
            port = f":{parsed.port}" if parsed.port else ""
            new_netloc = f"{userinfo}{self._local_ip}{port}"
            rebuilt = parsed._replace(netloc=new_netloc)
            self.rtsp_out_url.set(urlunparse(rebuilt))
            LOGGER.info("RTSP URL normalizada a %s", urlunparse(rebuilt))

    def _rtsp_log(self, level: int, msg: str) -> None:
        try:
            LOGGER.log(level, msg)
        except Exception:
            print(f"[RTSP] {msg}", flush=True)

    def _rtsp_disable_toggle(self) -> None:
        try:
            self.root.after(0, lambda: self.rtsp_out_enable.set(False))
        except Exception:
            pass

    def _rtsp_set_codec(self, codec: str) -> None:
        try:
            self.root.after(0, lambda c=codec: self.rtsp_out_codec.set(c))
        except Exception:
            pass

    def _on_rtsp_transport_change(self) -> None:
        value = (self.rtsp_out_transport.get() or "").strip().upper()
        if value not in {"TCP", "UDP"}:
            value = "TCP"
            try:
                self.rtsp_out_transport.set(value)
            except Exception:
                pass

        current_url = self.rtsp_out_url.get().strip()
        updated = _ensure_rtsp_transport_param(current_url, value)
        if updated != current_url:
            try:
                self.rtsp_out_url.set(updated)
            except Exception:  # noqa: BLE001
                pass

        if self._loading_config:
            return

        self._rtsp_log(logging.INFO, f"Transporte RTSP de salida establecido a {value}")

        if self.rtsp_out_enable.get():
            self._rtsp_log(logging.INFO, "Reiniciando FFmpeg para aplicar el nuevo transporte")
            self._close_ffmpeg_process(update_indicator=False)
            self._rtsp_schedule_indicator(False)

    def _rtsp_reset_queue(self) -> None:
        self._rtsp_q = queue.Queue(maxsize=1)
        self._rtsp_idle_counter = 0
        self._rtsp_frames_sent = 0
        self._rtsp_last_stats = time.monotonic()
        self._rtsp_log(logging.DEBUG, "Cola RTSP reiniciada")

    def _close_ffmpeg_process(self, update_indicator: bool = True) -> None:
        if self._ffmpeg is None:
            return
        try:
            if self._ffmpeg.stdin:
                self._ffmpeg.stdin.close()
        except Exception:
            pass
        try:
            self._ffmpeg.wait(timeout=1.0)
        except Exception:
            try:
                self._ffmpeg.terminate()
            except Exception:
                pass
        self._ffmpeg = None
        if update_indicator:
            self._rtsp_schedule_indicator(False)
        self._rtsp_log(logging.INFO, "FFmpeg detenido")

    def _on_rtsp_toggle(self) -> None:
        self._normalize_rtsp_out_url()
        if bool(self.rtsp_out_enable.get()):
            self._rtsp_log(logging.INFO, f"Activando salida RTSP -> {self.rtsp_out_url.get().strip()}")
            self._rtsp_reset_queue()
            self._rtsp_schedule_indicator(False)
            if self._rtsp_thread is None or not self._rtsp_thread.is_alive():
                self._rtsp_thread = threading.Thread(target=self._rtsp_publisher_loop, daemon=True)
                self._rtsp_thread.start()
        else:
            self._rtsp_log(logging.INFO, "Desactivando salida RTSP")
            self._stop_rtsp_stream()

    def _stop_rtsp_stream(self) -> None:
        self._rtsp_fail_streak = 0
        self._drain_rtsp_queue()
        self._close_ffmpeg_process(update_indicator=True)
        self._rtsp_log(logging.INFO, "RTSP detenido")

    def _drain_rtsp_queue(self) -> None:
        try:
            while not self._rtsp_q.empty():
                self._rtsp_q.get_nowait()
        except Exception:
            pass

        # Ãrea de visualizaciÃ³n - ahora ocupa casi toda la ventana
        video_container = ttk.Frame(self.root, relief="sunken", borderwidth=2)
        video_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Configurar el grid del contenedor del vÃ­deo
        video_container.columnconfigure(0, weight=1)
        video_container.rowconfigure(0, weight=1)
        
        # Label para el vÃ­deo con fondo negro
        self.video_label = ttk.Label(video_container, background='black')
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Asegurarse de que el Ã¡rea de vÃ­deo se expanda
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

    # ------------------------- Callbacks -------------------------
    def _browse_model(self):
        path = filedialog.askopenfilename(title="Selecciona modelo", filetypes=[("Modelos", ".pt"), ("Todos", "*.*")])
        if path:
            if path.lower().endswith(".engine"):
                messagebox.showwarning("Modelo", "Los modelos .engine no estan soportados en esta version.")
                return
            self.model_path.set(path)
            self._save_config_debounced()
            self._publish_model_metadata()
            self._update_ui_model_names()
            self._restart_if_core_settings_changed()

    def _browse_model2(self):
        path = filedialog.askopenfilename(title="Selecciona modelo 2 (opcional)", filetypes=[("Modelos", ".pt"), ("Todos", "*.*")])
        if path:
            if path.lower().endswith(".engine"):
                messagebox.showwarning("Modelo", "Los modelos .engine no estan soportados en esta version.")
                return
            self.model_path2.set(path)
            self._save_config_debounced()
            self._publish_model_metadata()
            self._update_ui_model_names()
            self._restart_if_core_settings_changed()

    def _browse_video(self):
        path = filedialog.askopenfilename(title="Selecciona video", filetypes=[("Videos", ".mp4 .avi .mkv .mov"), ("Todos", "*.*")])
        if path:
            self.video_path.set(path)
            # Autocompletar salida sugerida
            self.out_path.set(self._default_out_from_video(path))
            self._save_config_debounced()

    def _browse_out(self):
        default_name = os.path.basename(self.video_path.get())
        if not default_name:
            default_name = "salida_anotada.mp4"
        elif not default_name.lower().endswith('.mp4'):
            default_name += "_anotado.mp4"
            
        path = filedialog.asksaveasfilename(
            title="Guardar salida",
            defaultextension=".mp4",
            filetypes=[("MP4", ".mp4")],
            initialfile=default_name
        )
        if path:
            self.out_path.set(path)
            self.save_out = True
            self._save_config_debounced()

    def _default_out_from_video(self, video_path: str) -> str:
        if not video_path:
            return ""
        base, _ = os.path.splitext(video_path)
        return base + "_manchas_annotated.mp4"

    # ------------------------- UI State Updates -------------------------
    def _update_ui_model_names(self) -> None:
        """Actualiza los nombres de modelos y cÃ¡mara en el dashboard."""
        # Nombre del modelo 1
        m1_path = self.model_path.get().strip()
        if m1_path:
            m1_name = os.path.basename(m1_path)
            # Truncar si es muy largo
            if len(m1_name) > 20:
                m1_name = m1_name[:17] + "..."
            self._ui_model1_name.set(m1_name)
        else:
            self._ui_model1_name.set("--")
        
        # Nombre del modelo 2
        m2_path = self.model_path2.get().strip()
        if m2_path:
            m2_name = os.path.basename(m2_path)
            if len(m2_name) > 20:
                m2_name = m2_name[:17] + "..."
            self._ui_model2_name.set(m2_name)
        else:
            self._ui_model2_name.set("--")
        
        # Nombre de la cÃ¡mara/fuente
        mode = self.source_mode.get()
        if mode == "RTSP":
            # Extraer host de la URL RTSP
            try:
                host = self.rtsp_host.get().strip()
                if host:
                    self._ui_camera_name.set(f"RTSP: {host}")
                else:
                    self._ui_camera_name.set("RTSP")
            except Exception:
                self._ui_camera_name.set("RTSP")
        else:
            video = self.video_path.get().strip()
            if video:
                name = os.path.basename(video)
                if len(name) > 25:
                    name = name[:22] + "..."
                self._ui_camera_name.set(name)
            else:
                self._ui_camera_name.set("--")

    # ------------------------- Perfiles globales -------------------------
    def _profiles_snapshot_current(self) -> dict[str, dict[str, object]]:
        return {
            "models": {
                "source_mode": self.source_mode.get(),
                "model_path": self.model_path.get(),
                "model_path2": self.model_path2.get(),
                "video_path": self.video_path.get(),
            },
            "rtsp_in": {
                "host": self.rtsp_host.get(),
                "port": self.rtsp_port.get(),
                "user": self.rtsp_user.get(),
                "password": self.rtsp_password.get(),
                "path": self.rtsp_path.get(),
                "manual_enabled": bool(self.rtsp_manual_enabled.get()),
                "manual_url": self.rtsp_manual_url.get(),
            },
            "rtsp_out": {
                "enable": bool(self.rtsp_out_enable.get()),
                "url": self.rtsp_out_url.get(),
                "codec": self.rtsp_out_codec.get(),
                "transport": self.rtsp_out_transport.get(),
            },
            "settings": {
                "conf": float(self.conf_var.get()),
                "iou": float(self.iou_var.get()),
                "imgsz": int(self.imgsz1_var.get()),
                "imgsz2": int(self.imgsz2_var.get()),
                "show_boxes": bool(self.show_boxes.get()),
                "show_names": bool(self.show_names.get()),
                "show_confidence": bool(self.show_confidence.get()),
                "show_masks": bool(self.show_masks.get()),
                "perf_mode": self.perf_mode.get(),
                "auto_skip": bool(self.auto_skip.get()),
                "target_fps": int(self.target_fps.get()),
                "det_stride": int(self.det_stride.get()),
                "filter_in_model": bool(self.filter_in_model.get()),
                "use_retina_masks": bool(self.use_retina_masks.get()),
                "masks_as_contours": bool(self.masks_as_contours.get()),
                "topk_draw": int(self.topk_draw.get()),
                "highlight_tiny": bool(self.highlight_tiny.get()),
                "agnostic_nms": bool(self.agnostic_nms.get()),
                "use_half": bool(self.use_half.get()),
                "line_thickness": int(self.line_thickness.get()),
                "font_scale": float(self.font_scale.get()),
                "area_text_scale": float(self.area_text_scale.get()),
                "enable_m1": bool(self.enable_m1.get()),
                "enable_m2": bool(self.enable_m2.get()),
                "ui_fps": int(self.ui_fps.get()),
                "area_label_mode": self._area_label_mode_flag,
                "class_cfg": self._serialize_class_cfg(),
                "snapshot_write_interval_ms": int(self.snapshot_write_interval_ms.get()),
                "snapshot_clean_interval_sec": float(self.snapshot_clean_interval_sec.get()),
                "restricciones_enabled": bool(self.restricciones_enabled.get()),
                "perf_tracked_class": self.perf_tracked_class.get(),
            },
        }

    def _profiles_apply_snapshot_to_settings(self, settings: dict[str, object], profile: ProfileData) -> None:
        models = profile.models
        rtsp_in = profile.rtsp_in
        rtsp_out = profile.rtsp_out
        profile_settings = profile.settings

        if "source_mode" in models:
            settings["source_mode"] = models["source_mode"]
        if "model_path" in models:
            settings["model_path"] = models["model_path"]
        if "model_path2" in models:
            settings["model_path2"] = models["model_path2"]
        if "video_path" in models:
            settings["video_path"] = models["video_path"]

        if "user" in rtsp_in:
            settings["rtsp_user"] = rtsp_in["user"]
        if "password" in rtsp_in:
            settings["rtsp_password"] = rtsp_in["password"]
        if "host" in rtsp_in:
            settings["rtsp_host"] = rtsp_in["host"]
        if "port" in rtsp_in:
            settings["rtsp_port"] = rtsp_in["port"]
        if "path" in rtsp_in:
            settings["rtsp_path"] = rtsp_in["path"]
        if "manual_enabled" in rtsp_in:
            settings["rtsp_manual_enabled"] = rtsp_in["manual_enabled"]
        if "manual_url" in rtsp_in:
            settings["rtsp_manual_url"] = rtsp_in["manual_url"]

        if "enable" in rtsp_out:
            settings["rtsp_out_enable"] = rtsp_out["enable"]
        if "url" in rtsp_out:
            settings["rtsp_out_url"] = rtsp_out["url"]
        if "codec" in rtsp_out:
            settings["rtsp_out_codec"] = rtsp_out["codec"]
        if "transport" in rtsp_out:
            settings["rtsp_out_transport"] = rtsp_out["transport"]

        if profile_settings:
            settings.update(profile_settings)

    def _profile_settings_for_profile(
        self,
        profile: ProfileData,
        base: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if base is None:
            settings = _profile_settings_sanitize(self._collect_config())
        else:
            settings = copy.deepcopy(base)
        self._profiles_apply_snapshot_to_settings(settings, profile)
        return settings

    def _profiles_snapshot_rtsp_out(self) -> dict[str, object]:
        return {
            "enable": bool(self.rtsp_out_enable.get()),
            "url": self.rtsp_out_url.get(),
            "codec": self.rtsp_out_codec.get(),
            "transport": self.rtsp_out_transport.get(),
        }

    def _profiles_commit_active_now(self) -> None:
        active_id = self.active_profile_id.get().strip()
        if not active_id:
            return
        profile = self._profiles_cache.get(active_id)
        if profile is None:
            return
        snapshot = self._profiles_snapshot_current()
        profile.models = snapshot["models"]
        profile.rtsp_in = snapshot["rtsp_in"]
        profile.rtsp_out = snapshot["rtsp_out"]
        profile.settings = snapshot["settings"]
        self._profiles_cache[active_id] = profile

    def _profiles_matches_current(self, profile: ProfileData, current: dict[str, dict[str, object]]) -> bool:
        return (
            str(profile.models.get("model_path", "")) == str(current["models"].get("model_path", ""))
            and str(profile.rtsp_in.get("host", "")) == str(current["rtsp_in"].get("host", ""))
            and str(profile.rtsp_out.get("url", "")) == str(current["rtsp_out"].get("url", ""))
        )

    def _profiles_sync_to_config(self) -> None:
        if not isinstance(self.config, dict):
            self.config = {}
        self.config[PROFILE_LIST_KEY] = {
            profile_id: profile.to_payload() for profile_id, profile in self._profiles_cache.items()
        }
        active_id = self.active_profile_id.get().strip()
        if active_id:
            self.config[PROFILE_ACTIVE_KEY] = active_id

    def _profiles_load_from_config(self) -> bool:
        """Carga perfiles desde self.config. Devuelve True si ajusta el activo o crea uno nuevo."""
        current_snapshot = self._profiles_snapshot_current()
        self._profiles_cache.clear()
        needs_save = False
        base_settings = _profile_settings_sanitize(self.config) if isinstance(self.config, dict) else {}

        profiles_section = self.config.get(PROFILE_LIST_KEY)
        if isinstance(profiles_section, dict):
            for profile_id, payload in profiles_section.items():
                if not isinstance(profile_id, str):
                    continue
                profile = ProfileData.from_payload(profile_id, payload, defaults=current_snapshot)
                if profile.settings is None:
                    profile.settings = self._profile_settings_for_profile(profile, base_settings)
                    needs_save = True
                self._profiles_cache[profile.profile_id] = profile

        if not self._profiles_cache:
            profile = ProfileData(
                profile_id=DEFAULT_PROFILE_ID,
                name=DEFAULT_PROFILE_NAME,
                models=current_snapshot["models"],
                rtsp_in=current_snapshot["rtsp_in"],
                rtsp_out=current_snapshot["rtsp_out"],
                settings=current_snapshot["settings"],
            )
            profile.settings = self._profile_settings_for_profile(profile, base_settings)
            self._profiles_cache[profile.profile_id] = profile
            self.active_profile_id.set(profile.profile_id)
            self._profiles_sync_to_config()
            return True

        active_id = str(self.config.get(PROFILE_ACTIVE_KEY) or "")
        if active_id not in self._profiles_cache:
            active_id = next(iter(self._profiles_cache.keys()))
            self.config[PROFILE_ACTIVE_KEY] = active_id
            needs_save = True
        if not active_id:
            active_id = next(iter(self._profiles_cache.keys()))
            self.config[PROFILE_ACTIVE_KEY] = active_id
            needs_save = True
        self.active_profile_id.set(active_id)
        return needs_save

    def _profiles_migrate_from_presets(self) -> bool:
        """Crea perfiles a partir de presets si no existen. Devuelve True si migra."""
        if PROFILE_LIST_KEY in self.config and isinstance(self.config.get(PROFILE_LIST_KEY), dict):
            return False

        self.config[PROFILE_LIST_KEY] = {}
        current_snapshot = self._profiles_snapshot_current()
        base_settings = _profile_settings_sanitize(self.config) if isinstance(self.config, dict) else {}
        presets_root = _safe_dict(self.config.get("presets"))
        p_models = _safe_dict(presets_root.get("models"))
        p_in = _safe_dict(presets_root.get("rtsp_in"))
        p_out = _safe_dict(presets_root.get("rtsp_out"))
        aliases = sorted({*p_models.keys(), *p_in.keys(), *p_out.keys()})
        profiles: dict[str, ProfileData] = {}

        for alias in aliases:
            name = str(alias)
            profile_id = _generate_profile_id(name, set(profiles.keys()))
            models = _merge_profile_section(current_snapshot["models"], _safe_dict(p_models.get(alias)))
            rtsp_in = _merge_profile_section(current_snapshot["rtsp_in"], _safe_dict(p_in.get(alias)))
            rtsp_out = _merge_profile_section(current_snapshot["rtsp_out"], _safe_dict(p_out.get(alias)))
            profile = ProfileData(
                profile_id=profile_id,
                name=name,
                models=models,
                rtsp_in=rtsp_in,
                rtsp_out=rtsp_out,
            )
            profile.settings = self._profile_settings_for_profile(profile, base_settings)
            profiles[profile_id] = profile

        if not profiles:
            profile = ProfileData(
                profile_id=DEFAULT_PROFILE_ID,
                name=DEFAULT_PROFILE_NAME,
                models=current_snapshot["models"],
                rtsp_in=current_snapshot["rtsp_in"],
                rtsp_out=current_snapshot["rtsp_out"],
                settings=current_snapshot["settings"],
            )
            profile.settings = self._profile_settings_for_profile(profile, base_settings)
            profiles[profile.profile_id] = profile

        active_id = ""
        for profile in profiles.values():
            if self._profiles_matches_current(profile, current_snapshot):
                active_id = profile.profile_id
                break
        if not active_id:
            ordered = sorted(profiles.values(), key=lambda item: item.name.lower())
            active_id = ordered[0].profile_id if ordered else DEFAULT_PROFILE_ID

        self.config[PROFILE_LIST_KEY] = {pid: profile.to_payload() for pid, profile in profiles.items()}
        self.config[PROFILE_ACTIVE_KEY] = active_id
        return True

    def _profiles_bootstrap_from_config(self) -> None:
        if not isinstance(self.config, dict):
            self.config = {}

        had_profiles = isinstance(self.config.get(PROFILE_LIST_KEY), dict)
        active_before = str(self.config.get(PROFILE_ACTIVE_KEY) or "")
        migrated = self._profiles_migrate_from_presets()
        adjusted = self._profiles_load_from_config()
        active_after = self.active_profile_id.get()

        if migrated or not had_profiles or adjusted or active_after != active_before:
            self._profiles_sync_to_config()
            self._save_config()

    def _profiles_refresh_ui(self) -> None:
        profiles_sorted = sorted(self._profiles_cache.values(), key=lambda item: item.name.lower())
        self._profiles_order = [profile.profile_id for profile in profiles_sorted]
        labels = [profile.name for profile in profiles_sorted]
        current_id = self.active_profile_id.get()
        if current_id not in self._profiles_order and self._profiles_order:
            current_id = self._profiles_order[0]
            self.active_profile_id.set(current_id)
            self.config[PROFILE_ACTIVE_KEY] = current_id

        idx = self._profiles_order.index(current_id) if current_id in self._profiles_order else -1
        self._profiles_updating_ui = True
        try:
            for combo_name in ("combo_profiles_main", "combo_profiles_settings"):
                combo = getattr(self, combo_name, None)
                if combo is None:
                    continue
                try:
                    if not combo.winfo_exists():
                        setattr(self, combo_name, None)
                        continue
                except tk.TclError:
                    setattr(self, combo_name, None)
                    continue
                try:
                    combo.configure(values=labels)
                    if idx >= 0:
                        combo.current(idx)
                    else:
                        combo.set("")
                except tk.TclError:
                    setattr(self, combo_name, None)
        finally:
            self._profiles_updating_ui = False

    def _profiles_apply(self, profile_id: str, *, persist_active: bool = True) -> None:
        profile = self._profiles_cache.get(profile_id)
        if profile is None:
            return
        if self._profiles_updating_ui:
            return

        rtsp_before = self._profiles_snapshot_rtsp_out()
        was_loading = self._loading_config
        self._loading_config = True
        try:
            payload = copy.deepcopy(self.config) if isinstance(self.config, dict) else {}
            if profile.settings is not None:
                payload.update(copy.deepcopy(profile.settings))
            self._profiles_apply_snapshot_to_settings(payload, profile)
            self._apply_config_payload(payload)
            self._update_ui_model_names()
        finally:
            self._loading_config = was_loading

        self.active_profile_id.set(profile_id)
        self.config[PROFILE_ACTIVE_KEY] = profile_id

        if getattr(self, "_sector_panel", None) is not None:
            try:
                if isinstance(self.config.get("sectores"), dict):
                    self._sector_panel.set_config(self.config["sectores"])
            except Exception:
                pass
            try:
                self._sector_panel.set_available_classes(list(self.class_cfg.keys()))
            except Exception:
                pass
            try:
                if hasattr(self, "_actualizar_diagrama_sectores"):
                    self._actualizar_diagrama_sectores()
            except Exception:
                pass

        self._apply_class_config()
        self._on_sector_config_change(save_config=False)
        self._apply_restricciones_to_sectorizador()
        try:
            self._update_rtsp_preview()
        except Exception:
            pass

        rtsp_after = self._profiles_snapshot_rtsp_out()
        if self.running and rtsp_before.get("enable"):
            if rtsp_before != rtsp_after:
                self._close_ffmpeg_process(update_indicator=False)
                self._rtsp_reset_queue()
                if rtsp_after.get("enable"):
                    self._rtsp_schedule_indicator(False)

        self._restart_if_core_settings_changed()
        self._profiles_refresh_ui()
        self._set_status(f"Perfil '{profile.name}' aplicado.")
        self._ui_status_message.set(f"Perfil '{profile.name}' aplicado.")

        if persist_active:
            self._profiles_sync_to_config()
            self._save_config()

    def _profiles_save_current_to(self, profile_id: str) -> None:
        profile = self._profiles_cache.get(profile_id)
        if profile is None:
            return
        snapshot = self._profiles_snapshot_current()
        profile.models = snapshot["models"]
        profile.rtsp_in = snapshot["rtsp_in"]
        profile.rtsp_out = snapshot["rtsp_out"]
        profile.settings = snapshot["settings"]
        self._profiles_cache[profile_id] = profile
        self._profiles_sync_to_config()
        self._save_config()
        self._set_status(f"Perfil '{profile.name}' guardado.")
        self._ui_status_message.set(f"Perfil '{profile.name}' guardado.")

    def _profiles_prompt_name(self, title: str, initial: str) -> str | None:
        name = simpledialog.askstring(title, "Nombre del perfil:", initialvalue=initial, parent=self.root)
        if name is None:
            return None
        return name.strip()

    def _profiles_create(self, name: str) -> ProfileData | None:
        safe_name = name.strip()
        if not safe_name:
            return None
        new_id = _generate_profile_id(safe_name, set(self._profiles_cache.keys()))
        snapshot = self._profiles_snapshot_current()
        settings = snapshot["settings"]
        profile = ProfileData(
            profile_id=new_id,
            name=safe_name,
            models=snapshot["models"],
            rtsp_in=snapshot["rtsp_in"],
            rtsp_out=snapshot["rtsp_out"],
            settings=settings,
        )
        self._profiles_cache[new_id] = profile
        self.active_profile_id.set(new_id)
        self._profiles_sync_to_config()
        self._save_config()
        self._profiles_refresh_ui()
        return profile

    def _profiles_duplicate_active(self) -> ProfileData | None:
        active_id = self.active_profile_id.get()
        active = self._profiles_cache.get(active_id)
        base_name = active.name if active else "Perfil"
        suggested = f"{base_name} copia"
        name = self._profiles_prompt_name("Duplicar perfil", suggested)
        if not name:
            return None
        return self._profiles_create(name)

    def _profiles_rename(self, profile_id: str, new_name: str) -> None:
        profile = self._profiles_cache.get(profile_id)
        if profile is None:
            return
        safe_name = new_name.strip()
        if not safe_name:
            return
        profile.name = safe_name
        self._profiles_cache[profile_id] = profile
        self._profiles_sync_to_config()
        self._save_config()
        self._profiles_refresh_ui()

    def _profiles_delete(self, profile_id: str) -> None:
        if profile_id not in self._profiles_cache:
            return
        if len(self._profiles_cache) <= 1:
            messagebox.showwarning("Perfiles", "No se puede eliminar el ultimo perfil.")
            return
        was_active = profile_id == self.active_profile_id.get()
        self._profiles_cache.pop(profile_id, None)
        if not self._profiles_cache:
            return
        if was_active:
            next_id = next(iter(self._profiles_cache.keys()))
            self._profiles_apply(next_id, persist_active=True)
            return
        self._profiles_sync_to_config()
        self._save_config()
        self._profiles_refresh_ui()

    def _profiles_on_selected(self, source: str) -> None:
        if self._profiles_updating_ui:
            return
        combo_name = "combo_profiles_main" if source == "main" else "combo_profiles_settings"
        combo = getattr(self, combo_name, None)
        if combo is None:
            return
        idx = combo.current()
        if idx < 0 or idx >= len(self._profiles_order):
            return
        profile_id = self._profiles_order[idx]
        if profile_id == self.active_profile_id.get():
            return
        self._cancel_pending_save()
        self._profiles_commit_active_now()
        self._profiles_apply(profile_id, persist_active=True)

    def _profiles_save_active(self) -> None:
        profile_id = self.active_profile_id.get()
        if not profile_id:
            return
        if profile_id == DEFAULT_PROFILE_ID and len(self._profiles_cache) <= 1:
            name = self._profiles_prompt_name("Guardar perfil", "Perfil nuevo")
            if not name:
                return
            profile = self._profiles_create(name)
            if profile is not None:
                self._set_status(f"Perfil '{profile.name}' guardado.")
                self._ui_status_message.set(f"Perfil '{profile.name}' guardado.")
            return
        self._profiles_save_current_to(profile_id)

    def _profiles_ui_new(self) -> None:
        name = self._profiles_prompt_name("Nuevo perfil", "Perfil nuevo")
        if not name:
            return
        profile = self._profiles_create(name)
        if profile is not None:
            self._set_status(f"Perfil '{profile.name}' creado.")
            self._ui_status_message.set(f"Perfil '{profile.name}' creado.")

    def _profiles_ui_duplicate(self) -> None:
        profile = self._profiles_duplicate_active()
        if profile is not None:
            self._set_status(f"Perfil '{profile.name}' creado.")
            self._ui_status_message.set(f"Perfil '{profile.name}' creado.")

    def _profiles_ui_rename(self) -> None:
        profile_id = self.active_profile_id.get()
        profile = self._profiles_cache.get(profile_id)
        if profile is None:
            return
        name = self._profiles_prompt_name("Renombrar perfil", profile.name)
        if not name:
            return
        self._profiles_rename(profile_id, name)
        self._set_status(f"Perfil '{name}' renombrado.")
        self._ui_status_message.set(f"Perfil '{name}' renombrado.")

    def _profiles_ui_delete(self) -> None:
        profile_id = self.active_profile_id.get()
        profile = self._profiles_cache.get(profile_id)
        if profile is None:
            return
        if not messagebox.askyesno("Eliminar perfil", f"Eliminar perfil '{profile.name}'?"):
            return
        self._profiles_delete(profile_id)

    def _update_ui_fps(self, fps_value: float) -> None:
        """Actualiza el valor de FPS en el dashboard."""
        self._ui_fps_value.set(f"{fps_value:.1f}")

    def _set_status(self, text: str) -> None:
        with self._ui_cache_lock:
            self._status_cache = text

    def _cache_fps(self, fps_value: float) -> None:
        with self._ui_cache_lock:
            self._fps_cache = float(fps_value)

    def _flush_ui_cache(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        with self._ui_cache_lock:
            status = self._status_cache
            fps_value = self._fps_cache

        if (
            status is not None
            and status != self._status_last_value
            and (now - self._status_last_ui) >= self._status_ui_interval_sec
        ):
            try:
                self.status_var.set(status)
            except Exception:
                pass
            try:
                self._ui_status_message.set(status)
            except Exception:
                pass
            self._status_last_ui = now
            self._status_last_value = status

        if fps_value is not None and fps_value != self._fps_last_value:
            try:
                self.fps_var.set(f"FPS: {fps_value:.1f}")
            except Exception:
                pass
            self._fps_last_value = fps_value

    def _update_runtime_cache(self) -> None:
        cfg = dict(self._runtime_cfg)
        try:
            cfg["perf_mode"] = str(self.perf_mode.get())
        except Exception:
            pass
        try:
            cfg["auto_skip"] = bool(self.auto_skip.get())
        except Exception:
            pass
        try:
            cfg["target_fps"] = int(self.target_fps.get())
        except Exception:
            pass
        try:
            cfg["det_stride"] = int(self.det_stride.get())
        except Exception:
            pass
        try:
            cfg["stride2"] = int(self.stride2_var.get())
        except Exception:
            pass
        try:
            cfg["imgsz1"] = int(self.imgsz1_var.get())
        except Exception:
            pass
        try:
            cfg["imgsz2"] = int(self.imgsz2_var.get())
        except Exception:
            pass
        try:
            cfg["max_det"] = int(self.max_det_var.get())
        except Exception:
            pass
        try:
            cfg["topk_draw"] = int(self.topk_draw.get())
        except Exception:
            pass
        try:
            cfg["filter_in_model"] = bool(self.filter_in_model.get())
        except Exception:
            pass
        try:
            cfg["agnostic_nms"] = bool(self.agnostic_nms.get())
        except Exception:
            pass
        try:
            cfg["use_half"] = bool(self.use_half.get())
        except Exception:
            pass
        try:
            cfg["enable_m1"] = bool(self.enable_m1.get())
        except Exception:
            pass
        try:
            cfg["enable_m2"] = bool(self.enable_m2.get())
        except Exception:
            pass
        try:
            cfg["show_boxes"] = bool(self.show_boxes.get())
        except Exception:
            pass
        try:
            cfg["show_names"] = bool(self.show_names.get())
        except Exception:
            pass
        try:
            cfg["show_confidence"] = bool(self.show_confidence.get())
        except Exception:
            pass
        try:
            cfg["show_masks"] = bool(self.show_masks.get())
        except Exception:
            pass
        try:
            cfg["masks_as_contours"] = bool(self.masks_as_contours.get())
        except Exception:
            pass
        try:
            cfg["use_retina_masks"] = bool(self.use_retina_masks.get())
        except Exception:
            pass
        try:
            cfg["highlight_tiny"] = bool(self.highlight_tiny.get())
        except Exception:
            pass
        try:
            cfg["line_thickness"] = int(self.line_thickness.get())
        except Exception:
            pass
        try:
            cfg["font_scale"] = float(self.font_scale.get())
        except Exception:
            pass
        try:
            cfg["area_text_scale"] = float(self.area_text_scale.get())
        except Exception:
            pass
        try:
            cfg["rtsp_out_enable"] = bool(self.rtsp_out_enable.get())
        except Exception:
            pass
        try:
            cfg["rtsp_out_url"] = str(self.rtsp_out_url.get())
        except Exception:
            pass
        try:
            cfg["rtsp_out_codec"] = str(self.rtsp_out_codec.get())
        except Exception:
            pass
        try:
            cfg["rtsp_out_transport"] = str(self.rtsp_out_transport.get())
        except Exception:
            pass
        try:
            cfg["sector_mostrar"] = bool(self.sector_mostrar.get())
        except Exception:
            pass
        try:
            cfg["ui_fps"] = int(self.ui_fps.get())
        except Exception:
            pass
        try:
            cfg["perf_tracked_class"] = str(self.perf_tracked_class.get())
        except Exception:
            pass
        self._runtime_cfg = cfg

    def _update_ui_chip(self, chip_name: str, text: str, status: str) -> None:
        """Actualiza un chip de estado en el dashboard."""
        chip_map = {
            "rtsp_in": self._chip_rtsp_in if hasattr(self, '_chip_rtsp_in') else None,
            "rtsp_out": self._chip_rtsp_out if hasattr(self, '_chip_rtsp_out') else None,
            "plc": self._chip_plc if hasattr(self, '_chip_plc') else None,
        }
        chip = chip_map.get(chip_name)
        if chip is None:
            return
        
        color_map = {
            "ok": (self.colors["chip_ok_bg"], self.colors["chip_ok_fg"]),
            "warning": (self.colors["chip_warn_bg"], self.colors["chip_warn_fg"]),
            "error": (self.colors["chip_error_bg"], self.colors["chip_error_fg"]),
        }
        bg, fg = color_map.get(status, color_map["ok"])
        try:
            chip.configure(text=text, bg=bg, fg=fg)
        except Exception:
            pass

    # ------------------------- Persistencia de configuraciÃ³n -------------------------
    def _setup_var_traces(self) -> None:
        tracked_vars: list[tk.Variable] = [
            self.model_path,
            self.model_path2,
            self.video_path,
            self.out_path,
            self.perf_mode,
            self.auto_skip,
            self.target_fps,
            self.det_stride,
            self.stride2_var,
            self.imgsz1_var,
            self.imgsz2_var,
            self.max_det_var,
            self.topk_draw,
            self.filter_in_model,
            self.agnostic_nms,
            self.use_half,
            self.enable_m1,
            self.enable_m2,
            self.show_boxes,
            self.show_names,
            self.show_confidence,
            self.show_masks,
            self.use_retina_masks,
            self.masks_as_contours,
            self.highlight_tiny,
            self.line_thickness,
            self.font_scale,
            self.area_text_scale,
            self.ui_fps,
            self.snapshot_write_interval_ms,
            self.snapshot_clean_interval_sec,
            self.sector_borde_sup,
            self.sector_borde_inf,
            self.sector_borde_izq,
            self.sector_borde_der,
            self.sector_modo,
            self.sector_num_vert,
            self.sector_num_horiz,
            self.sector_mostrar,
            self.sector_mostrar_etiquetas,
            self.sector_mostrar_borde_banda,
            self.sector_opacidad_lineas,
            self.sector_grosor_lineas,
            self.sector_use_perspective,
            self.sector_use_masks,
            self.sector_modo_delimitacion,
            self.sector_estabilidad,
            self.sector_comportamiento_fallo,
            self.sector_smooth_alpha,
            self.sector_max_jump,
            self.sector_inset,
            self.sector_debug_overlay,
            self.sector_curved_enabled,
            self.sector_curved_bins_vert,
            self.sector_curved_bins_horiz,
            self.sector_padding_top,
            self.sector_padding_bottom,
            self.sector_padding_left,
            self.sector_padding_right,
            self.restricciones_enabled,
            self.perf_tracked_class,
        ]
        for var in tracked_vars:
            self._trace_var(var)

        self._trace_var(self.area_label_mode_display)

        self._trace_var(self.model_path, self._on_model_path_change)
        self._trace_var(self.model_path2, self._on_model_path_change)
        self._trace_var(self.video_path)
        self._trace_var(self.out_path)
        self._trace_var(self.conf_var, self._on_conf_var_change)
        self._trace_var(self.iou_var, self._on_iou_var_change)
        self._trace_var(self.source_mode, self._on_source_mode_change)
        for rtsp_var in (
            self.rtsp_user,
            self.rtsp_password,
            self.rtsp_host,
            self.rtsp_port,
            self.rtsp_path,
            self.rtsp_manual_url,
        ):
            self._trace_var(rtsp_var, self._update_rtsp_preview)
        self._trace_var(self.rtsp_manual_enabled, self._update_rtsp_preview)
        self._trace_var(self.rtsp_out_enable, self._on_rtsp_toggle)
        self._trace_var(self.rtsp_out_url, self._refresh_heartbeat_targets)
        self._trace_var(self.rtsp_out_codec)
        self._trace_var(self.rtsp_out_transport, self._on_rtsp_transport_change)
        self._trace_var(self.snapshot_write_interval_ms, self._on_snapshot_timing_change)
        self._trace_var(self.snapshot_clean_interval_sec, self._on_snapshot_timing_change)

        # Flags accesibles desde hilos secundarios
        display_mode = self.area_label_mode_display.get()
        # Ser tolerantes con strings legacy o con variaciones (p.ej. "Sin etiqueta", "cm^2", etc).
        self._area_label_mode_flag = _coerce_area_mode(display_mode, getattr(self, "_area_label_mode_flag", AREA_MODE_CM2))
        if self._area_label_mode_flag == AREA_MODE_INHERIT:
            self._area_label_mode_flag = AREA_MODE_CM2
        self._recompute_area_mode_flags()
        self._update_runtime_cache()

    def _trace_var(self, var: tk.Variable, callback=None) -> None:
        def _handler(*_ignore):
            if callback is not None:
                callback()
            self._on_var_write(var)

        handle = var.trace_add("write", _handler)
        self._var_traces.append((var, handle))

    def _recompute_area_mode_flags(self) -> None:
        any_enabled = False

        for cfg in self.class_cfg.values():
            if not isinstance(cfg, dict):
                continue
            if self._class_area_enabled(cfg, "M1") or self._class_area_enabled(cfg, "M2"):
                any_enabled = True
                break

        self._area_any_enabled_flag = any_enabled

    def _area_any_enabled_for_model(self, model_tag: str | None) -> bool:
        any_enabled = bool(getattr(self, "_area_any_enabled_flag", False))
        if not any_enabled:
            return False
        if model_tag is None:
            return True
        cache = getattr(self, "_class_cache", {})
        cache_entry = cache.get("M1" if model_tag == "M1" else "M2", {})
        area_mode = cache_entry.get("area_mode")
        if isinstance(area_mode, np.ndarray):
            try:
                return bool(area_mode.any())
            except Exception:
                return False
        return False

    def _class_area_enabled(self, cfg: dict, model_tag: str) -> bool:
        if not isinstance(cfg, dict):
            return False
        key = "cm2_m1" if model_tag == "M1" else "cm2_m2"
        val = cfg.get(key)
        if val is not None:
            return _area_enabled_from_value(val)
        legacy_key = "area_mode_m1" if model_tag == "M1" else "area_mode_m2"
        legacy_val = cfg.get(legacy_key)
        if legacy_val is None:
            return False
        mode = _coerce_area_mode(legacy_val, AREA_MODE_INHERIT)
        if mode == AREA_MODE_INHERIT:
            global_mode = getattr(self, "_area_label_mode_flag", AREA_MODE_CM2)
            return global_mode != AREA_MODE_OFF
        return mode != AREA_MODE_OFF

    def _class_area_mode(self, cfg: dict, model_tag: str, global_mode: str) -> str:
        if not self._class_area_enabled(cfg, model_tag):
            return AREA_MODE_OFF
        if global_mode == AREA_MODE_OFF:
            return AREA_MODE_OFF
        return global_mode

    def _on_var_write(self, _var: tk.Variable) -> None:  # noqa: ARG002
        if self._loading_config:
            return
        if _var is self.area_label_mode_display:
            display = self.area_label_mode_display.get()
            self._area_label_mode_flag = _coerce_area_mode(display, getattr(self, "_area_label_mode_flag", AREA_MODE_CM2))
            if self._area_label_mode_flag == AREA_MODE_INHERIT:
                self._area_label_mode_flag = AREA_MODE_CM2
            self._recompute_area_mode_flags()
        self._update_runtime_cache()
        self._save_config_debounced()

    def _on_model_path_change(self) -> None:
        if self._loading_config:
            return
        self._ensure_default_class_cfg()
        self._publish_model_metadata()

    def _on_conf_var_change(self) -> None:
        try:
            self.conf = _clamp01(self.conf_var.get(), self.conf)
        except Exception:
            return
        if self._conf_value_label is not None:
            try:
                self._conf_value_label.configure(text=f"{self.conf*100:.0f}%")
            except tk.TclError:
                self._conf_value_label = None

    def _on_iou_var_change(self) -> None:
        try:
            self.iou = _clamp01(self.iou_var.get(), self.iou)
        except Exception:
            return
        if self._iou_value_label is not None:
            try:
                self._iou_value_label.configure(text=f"{self.iou*100:.0f}%")
            except tk.TclError:
                self._iou_value_label = None

    def _on_source_mode_change(self) -> None:
        mode = self.source_mode.get()
        if mode not in {"Archivo", "RTSP"}:
            mode = "Archivo"
            self.source_mode.set(mode)
        if mode == self._current_source_kind:
            return
        self._current_source_kind = mode
        self._update_source_widgets()
        self._update_rtsp_preview()

    def _update_source_widgets(self) -> None:
        if self._file_picker_frame is None or self._rtsp_summary_frame is None:
            return
        mode = self.source_mode.get()
        show_file = (mode == "Archivo")
        try:
            self._file_picker_frame.grid_remove() if not show_file else self._file_picker_frame.grid()
        except Exception:
            pass
        try:
            self._rtsp_summary_frame.grid_remove() if show_file else self._rtsp_summary_frame.grid()
        except Exception:
            pass

    def _update_rtsp_preview(self) -> None:
        if self.rtsp_manual_enabled.get():
            manual = self.rtsp_manual_url.get().strip()
            preview = manual if manual else "rtsp://<usuario>:<password>@<ip>:554/..."
            self._rtsp_preview_var.set(preview)
            return

        user = self.rtsp_user.get().strip()
        pwd = self.rtsp_password.get().strip()
        host = self.rtsp_host.get().strip()
        port = self.rtsp_port.get().strip() or "554"
        path = self.rtsp_path.get().strip().lstrip('/ ')

        if path.lower().startswith("rtsp://"):
            self._rtsp_preview_var.set(path)
            return

        preview = "rtsp://"
        if user:
            preview += user
            if pwd:
                preview += f":{pwd}"
            preview += "@"
        elif pwd:
            preview += f":{pwd}@"

        preview += host if host else "<ip>"
        if port:
            preview += f":{port}"
        if path:
            preview += "/" + path

        self._rtsp_preview_var.set(preview)

    def _open_rtsp_dialog(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Configuracion RTSP")

        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(0, weight=1)

        manual_chk = ttk.Checkbutton(
            frm,
            text="Introduce manualmente la URL completa",
            variable=self.rtsp_manual_enabled,
            command=self._update_rtsp_dialog_state,
        )
        manual_chk.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        entries: list[tuple[str, tk.StringVar, str]] = [
            ("Usuario", self.rtsp_user, "Nombre de usuario de la camara"),
            ("Contrasena", self.rtsp_password, "Contrasena asociada"),
            ("IP/Dominio", self.rtsp_host, "Direccion IP (ej. 192.168.1.10)"),
            ("Puerto", self.rtsp_port, "Puerto RTSP (por defecto 554)"),
            ("Ruta", self.rtsp_path, "Ruta final, sin la barra inicial"),
        ]

        base_row = 1
        self._rtsp_field_widgets: list[tuple[ttk.Label, ttk.Entry, ttk.Label]] = []
        for offset, (label, var, help_text) in enumerate(entries):
            idx = base_row + offset
            lbl = ttk.Label(frm, text=f"{label}:")
            lbl.grid(row=idx, column=0, sticky="w", pady=4)
            show = (label == "Contrasena")
            ent = ttk.Entry(frm, textvariable=var, show="*" if show else "")
            ent.grid(row=idx, column=1, sticky="we", padx=(6, 0))
            ent.configure(font=('Arial', 10))
            tip = ttk.Label(frm, text=help_text, foreground="#666666", font=('Arial', 9))
            tip.grid(row=idx, column=2, sticky="w", padx=(6, 0))
            self._rtsp_field_widgets.append((lbl, ent, tip))

        manual_row = base_row + len(entries)
        ttk.Label(frm, text="URL manual:").grid(row=manual_row, column=0, sticky="w", pady=(12, 4))
        self._rtsp_manual_entry = ttk.Entry(frm, textvariable=self.rtsp_manual_url, font=('Consolas', 10))
        self._rtsp_manual_entry.grid(row=manual_row, column=1, sticky="we", padx=(6, 0), pady=(12, 4))
        self._rtsp_manual_entry_tip = ttk.Label(
            frm,
            text="Pega la URL completa tal como la proporciona la camara",
            foreground="#666666",
            font=('Arial', 9),
        )
        self._rtsp_manual_entry_tip.grid(row=manual_row, column=2, sticky="w", padx=(6, 0), pady=(12, 4))

        frm.columnconfigure(1, weight=1)

        # Vista previa y botÃ³n de prueba
        preview_frame = ttk.LabelFrame(dlg, text="Vista previa", padding=10)
        preview_frame.grid(row=1, column=0, sticky="we", padx=10, pady=6)
        lbl_prev = ttk.Label(preview_frame, textvariable=self._rtsp_preview_var, font=('Consolas', 10))
        lbl_prev.grid(row=0, column=0, sticky="w")
        ttk.Button(preview_frame, text="Probar conexion", command=self._test_rtsp_connection).grid(row=0, column=1, padx=(10, 0))

        btns = ttk.Frame(dlg, padding=10)
        btns.grid(row=2, column=0, sticky="e")
        ttk.Button(btns, text="Cancelar", command=dlg.destroy).grid(row=0, column=0, padx=4)

        def save_and_close():
            self._update_rtsp_preview()
            self._restart_if_core_settings_changed()
            dlg.destroy()

        ttk.Button(btns, text="Guardar", command=save_and_close).grid(row=0, column=1, padx=4)

        self._update_rtsp_preview()
        self._update_rtsp_dialog_state()
        dlg.wait_window()

    def _update_rtsp_dialog_state(self) -> None:
        manual = self.rtsp_manual_enabled.get()
        state = "disabled" if manual else "normal"
        for lbl, ent, tip in getattr(self, "_rtsp_field_widgets", []):
            try:
                lbl.configure(state=state)
                ent.configure(state=state)
                tip.configure(state=state)
            except Exception:
                pass
        if hasattr(self, "_rtsp_manual_entry"):
            try:
                self._rtsp_manual_entry.configure(state="normal" if manual else "disabled")
                self._rtsp_manual_entry_tip.configure(state="normal" if manual else "disabled")
            except Exception:
                pass

    def _build_rtsp_url(self) -> str:
        if self.rtsp_manual_enabled.get():
            manual = self.rtsp_manual_url.get().strip()
            return manual

        user = self.rtsp_user.get().strip()
        pwd = self.rtsp_password.get().strip()
        host = self.rtsp_host.get().strip()
        port = self.rtsp_port.get().strip() or "554"
        path = self.rtsp_path.get().strip().lstrip('/ ')

        if path.lower().startswith("rtsp://"):
            return path

        if not host:
            return ""

        url = "rtsp://"
        if user:
            url += user
            if pwd:
                url += f":{pwd}"
            url += "@"
        elif pwd:
            url += f":{pwd}@"

        url += host
        if port:
            url += f":{port}"
        if path:
            url += f"/{path}"
        return url

    def _test_rtsp_connection(self) -> None:
        url = self._build_rtsp_url()
        if not url:
            messagebox.showwarning("RTSP", "Completa al menos la IP/Dominio para probar la conexion.")
            return
        url = _ensure_rtsp_transport_param(url, self.rtsp_out_transport.get())
        url = _add_low_latency_flags(url)
        cap = cv2.VideoCapture(url)
        try:
            if not cap.isOpened():
                messagebox.showerror("RTSP", "No se pudo abrir el stream. Revisa credenciales e IP.")
                return
            ok, _ = cap.read()
            if not ok:
                messagebox.showerror("RTSP", "Se conecto pero no se pudo leer ningun frame.")
                return
        finally:
            cap.release()
        messagebox.showinfo("RTSP", "Conexion exitosa. El stream responde correctamente.")

    def _save_config_debounced(self, delay_ms: int = 400) -> None:
        if self._loading_config:
            return
        self._cancel_pending_save()
        try:
            self._save_after_id = self.root.after(delay_ms, self._save_config)
        except Exception:
            self._save_config()

    def _cancel_pending_save(self) -> None:
        if self._save_after_id is None:
            return
        try:
            self.root.after_cancel(self._save_after_id)
        except Exception:
            pass
        self._save_after_id = None

    def _serialize_class_cfg(self) -> dict[str, dict]:
        class_cfg_serialized: dict[str, dict] = {}
        for name, cfg in self.class_cfg.items():
            if not isinstance(cfg, dict):
                continue
            entry = dict(cfg)
            color = entry.get("color")
            if isinstance(color, (list, tuple)) and len(color) == 3:
                entry["color"] = [int(color[0]), int(color[1]), int(color[2])]
            entry["thr_inherit"] = bool(entry.get("thr_inherit", True))
            entry["thr"] = float(_clamp01(entry.get("thr", float(self.conf)), float(self.conf)))
            global_mode = getattr(self, "_area_label_mode_flag", AREA_MODE_CM2)
            if "cm2_m1" in entry:
                cm2_m1 = _area_enabled_from_value(entry.get("cm2_m1"))
            else:
                legacy_mode = _coerce_area_mode(entry.get("area_mode_m1"), AREA_MODE_INHERIT)
                cm2_m1 = (global_mode != AREA_MODE_OFF) if legacy_mode == AREA_MODE_INHERIT else (legacy_mode != AREA_MODE_OFF)
            if "cm2_m2" in entry:
                cm2_m2 = _area_enabled_from_value(entry.get("cm2_m2"))
            else:
                legacy_mode = _coerce_area_mode(entry.get("area_mode_m2"), AREA_MODE_INHERIT)
                cm2_m2 = (global_mode != AREA_MODE_OFF) if legacy_mode == AREA_MODE_INHERIT else (legacy_mode != AREA_MODE_OFF)
            entry["cm2_m1"] = bool(cm2_m1)
            entry["cm2_m2"] = bool(cm2_m2)
            entry["area_mode_m1"] = AREA_MODE_INHERIT if entry["cm2_m1"] else AREA_MODE_OFF
            entry["area_mode_m2"] = AREA_MODE_INHERIT if entry["cm2_m2"] else AREA_MODE_OFF
            class_cfg_serialized[str(name)] = entry
        return class_cfg_serialized

    def _collect_config(self) -> dict:
        class_cfg_serialized = self._serialize_class_cfg()

        calibration_payload: dict[str, float]
        source_calibration = self._raw_calibration_config or self.calibration
        if source_calibration:
            calibration_payload = {}
            for key in CALIBRATION_REQUIRED_KEYS:
                value = source_calibration.get(key) if isinstance(source_calibration, dict) else None
                if value is None:
                    continue
                try:
                    calibration_payload[key] = float(value)
                except (TypeError, ValueError):
                    continue
            for key in CALIBRATION_REQUIRED_KEYS:
                if key not in calibration_payload:
                    calibration_payload[key] = float(DEFAULT_CALIBRATION[key])
        else:
            calibration_payload = dict(DEFAULT_CALIBRATION)

        # Empezar con la configuraciÃ³n cargada para preservar campos extras (presets)
        data = copy.deepcopy(self.config) if isinstance(self.config, dict) else {}

        data.update({
            "model_path": self.model_path.get(),
            "model_path2": self.model_path2.get(),
            "video_path": self.video_path.get(),
            "out_path": self.out_path.get(),
            "save_out": bool(self.save_out),
            "source_mode": self.source_mode.get(),
            "rtsp_user": self.rtsp_user.get(),
            "rtsp_password": self.rtsp_password.get(),
            "rtsp_host": self.rtsp_host.get(),
            "rtsp_port": self.rtsp_port.get(),
            "rtsp_path": self.rtsp_path.get(),
            "rtsp_manual_enabled": bool(self.rtsp_manual_enabled.get()),
            "rtsp_manual_url": self.rtsp_manual_url.get(),
            "conf": float(self.conf),
            "iou": float(self.iou),
            "show_boxes": bool(self.show_boxes.get()),
            "show_names": bool(self.show_names.get()),
            "show_confidence": bool(self.show_confidence.get()),
            "show_masks": bool(self.show_masks.get()),
            "area_label_mode": self._area_label_mode_flag,
            "masks_as_contours": bool(self.masks_as_contours.get()),
            "use_retina_masks": bool(self.use_retina_masks.get()),
            "highlight_tiny": bool(self.highlight_tiny.get()),
            "perf_mode": self.perf_mode.get(),
            "auto_skip": bool(self.auto_skip.get()),
            "target_fps": int(self.target_fps.get()),
            "det_stride": int(self.det_stride.get()),
            "stride2_var": int(self.stride2_var.get()),
            "imgsz1": int(self.imgsz1_var.get()),
            "imgsz2": int(self.imgsz2_var.get()),
            "max_det": int(self.max_det_var.get()),
            "topk_draw": int(self.topk_draw.get()),
            "filter_in_model": bool(self.filter_in_model.get()),
            "agnostic_nms": bool(self.agnostic_nms.get()),
            "use_half": bool(self.use_half.get()),
            "enable_m1": bool(self.enable_m1.get()),
            "enable_m2": bool(self.enable_m2.get()),
            "line_thickness": int(self.line_thickness.get()),
            "font_scale": float(self.font_scale.get()),
            "area_text_scale": float(self.area_text_scale.get()),
            "ui_fps": int(self.ui_fps.get()),
            "class_cfg": class_cfg_serialized,
            "rtsp_out_enable": bool(self.rtsp_out_enable.get()),
            "rtsp_out_url": self.rtsp_out_url.get(),
            "rtsp_out_codec": self.rtsp_out_codec.get(),
            "rtsp_out_transport": self.rtsp_out_transport.get(),
            "snapshot_write_interval_ms": int(self.snapshot_write_interval_ms.get()),
            "snapshot_clean_interval_sec": float(self.snapshot_clean_interval_sec.get()),
            "calibration": calibration_payload,
            "perf_tracked_class": self.perf_tracked_class.get(),
            "sectorizador": {
                "borde_superior": self.sector_borde_sup.get(),
                "borde_inferior": self.sector_borde_inf.get(),
                "borde_izquierdo": self.sector_borde_izq.get(),
                "borde_derecho": self.sector_borde_der.get(),
                "modo": self.sector_modo.get(),
                "num_verticales": int(self.sector_num_vert.get()),
                "num_horizontales": int(self.sector_num_horiz.get()),
                "mostrar": bool(self.sector_mostrar.get()),
                "mostrar_etiquetas": bool(self.sector_mostrar_etiquetas.get()),
                "mostrar_borde_banda": bool(self.sector_mostrar_borde_banda.get()),
                "opacidad_lineas": float(self.sector_opacidad_lineas.get()),
                "grosor_lineas": int(self.sector_grosor_lineas.get()),
                "use_perspective": bool(self.sector_use_perspective.get()),
                "use_border_masks": bool(self.sector_use_masks.get()),
                "modo_delimitacion": self.sector_modo_delimitacion.get(),
                "estabilidad": self.sector_estabilidad.get(),
                "comportamiento_fallo": self.sector_comportamiento_fallo.get(),
                "smooth_alpha": float(self.sector_smooth_alpha.get()),
                "max_corner_jump_px": float(self.sector_max_jump.get()),
                "inset_px": int(self.sector_inset.get()),
                "debug_overlay": bool(self.sector_debug_overlay.get()),
                # NUEVO: Bordes curvos
                "curved_edges_enabled": bool(self.sector_curved_enabled.get()),
                "curved_bins_vertical": int(self.sector_curved_bins_vert.get()),
                "curved_bins_horizontal": int(self.sector_curved_bins_horiz.get()),
                "curved_percentile_trim": 0.10,  # Valor fijo por ahora
                # NUEVO: Padding
                "padding_top_px": int(self.sector_padding_top.get()),
                "padding_bottom_px": int(self.sector_padding_bottom.get()),
                "padding_left_px": int(self.sector_padding_left.get()),
                "padding_right_px": int(self.sector_padding_right.get()),
                "roi_quant_step_px": int(self.sector_roi_quant_step.get()),
                "line_quant_step_px": int(self.sector_line_quant_step.get()),
                # NUEVO: Restricciones por clase
                "restricciones_enabled": bool(self.restricciones_enabled.get()),
                "restricciones_por_clase": self._restricciones_por_clase,
            },
        })
        
        # Persistir tambiÃ©n configuraciones de dashboard y FPS
        if hasattr(self, "fps_config"):
            data["fps_settings"] = self.fps_config
        
        if hasattr(self, "_profiles_cache") and self._profiles_cache:
            active_id = self.active_profile_id.get().strip()
            if active_id and active_id in self._profiles_cache:
                profile = self._profiles_cache[active_id]
                snapshot = self._profiles_snapshot_current()
                profile.models = snapshot["models"]
                profile.rtsp_in = snapshot["rtsp_in"]
                profile.rtsp_out = snapshot["rtsp_out"]
                profile.settings = _profile_settings_sanitize(data)
                self._profiles_cache[active_id] = profile
            data[PROFILE_LIST_KEY] = {
                profile_id: profile.to_payload() for profile_id, profile in self._profiles_cache.items()
            }
            if active_id:
                data[PROFILE_ACTIVE_KEY] = active_id
        
        return data

    def _sync_active_profile_settings(self, data: dict[str, object]) -> None:
        if not isinstance(getattr(self, "_profiles_cache", None), dict):
            return
        if not self._profiles_cache:
            return
        active_id = self.active_profile_id.get().strip()
        if not active_id or active_id not in self._profiles_cache:
            return
        profile = self._profiles_cache[active_id]
        profile.settings = _profile_settings_sanitize(data)
        self._profiles_cache[active_id] = profile
        data[PROFILE_LIST_KEY] = {
            profile_id: profile.to_payload() for profile_id, profile in self._profiles_cache.items()
        }
        data[PROFILE_ACTIVE_KEY] = active_id

    def _save_config(self) -> None:
        data = self._collect_config()
        self._sync_active_profile_settings(data)
        self._cancel_pending_save()
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            self.config = data
            print(f"[Config] Configuración guardada exitosamente en: {CONFIG_PATH}", flush=True)
        except Exception as exc:
            print(f"[Config] ERROR al guardar configuración: {exc}", flush=True)

    def _apply_config_payload(self, data: dict[str, object]) -> None:
        if not isinstance(data, dict):
            data = {}
        self.config = data

        def _set_bool(var: tk.BooleanVar, key: str) -> None:
            if key in data:
                try:
                    var.set(bool(data[key]))
                except Exception:
                    pass

        def _set_int(var: tk.IntVar, key: str) -> None:
            if key in data:
                try:
                    var.set(int(data[key]))
                except Exception:
                    pass

        def _set_float(var: tk.DoubleVar, key: str) -> None:
            if key in data:
                try:
                    var.set(float(data[key]))
                except Exception:
                    pass

        if "model_path" in data:
            self.model_path.set(str(data["model_path"]))
        if "model_path2" in data:
            self.model_path2.set(str(data["model_path2"]))
        if "video_path" in data:
            self.video_path.set(str(data["video_path"]))
        if "out_path" in data:
            self.out_path.set(str(data["out_path"]))
        if "source_mode" in data:
            try:
                self.source_mode.set(str(data["source_mode"]))
            except Exception:
                pass
        if "rtsp_user" in data:
            self.rtsp_user.set(str(data["rtsp_user"]))
        if "rtsp_password" in data:
            self.rtsp_password.set(str(data["rtsp_password"]))
        if "rtsp_host" in data:
            self.rtsp_host.set(str(data["rtsp_host"]))
        if "rtsp_port" in data:
            self.rtsp_port.set(str(data["rtsp_port"]))
        if "rtsp_path" in data:
            self.rtsp_path.set(str(data["rtsp_path"]))
        if "rtsp_manual_enabled" in data:
            try:
                self.rtsp_manual_enabled.set(bool(data["rtsp_manual_enabled"]))
            except Exception:
                self.rtsp_manual_enabled.set(False)
        if "rtsp_manual_url" in data:
            self.rtsp_manual_url.set(str(data["rtsp_manual_url"]))
        try:
            path_value = self.rtsp_path.get().strip()
            manual_url = self.rtsp_manual_url.get().strip()
            if path_value.lower().startswith("rtsp://") and not self.rtsp_manual_enabled.get():
                self.rtsp_manual_enabled.set(True)
                if not manual_url:
                    self.rtsp_manual_url.set(path_value)
                self.rtsp_path.set("")
        except Exception:
            pass
        if "rtsp_out_enable" in data:
            try:
                self.rtsp_out_enable.set(bool(data["rtsp_out_enable"]))
            except Exception:
                self.rtsp_out_enable.set(False)
        if "rtsp_out_url" in data:
            self.rtsp_out_url.set(str(data["rtsp_out_url"]))
        if "rtsp_out_codec" in data:
            self.rtsp_out_codec.set(str(data["rtsp_out_codec"]))
        if "rtsp_out_transport" in data:
            self.rtsp_out_transport.set(str(data["rtsp_out_transport"]))

        if "save_out" in data:
            try:
                self.save_out = bool(data["save_out"])
            except Exception:
                self.save_out = False

        if "show_masks" in data:
            try:
                self.show_masks.set(bool(data["show_masks"]))
            except Exception:
                self.show_masks.set(True)

        area_label_raw = data.get("area_label_mode")
        if area_label_raw is not None:
            global_mode = _coerce_area_mode(area_label_raw, AREA_MODE_CM2)
        else:
            # Compatibilidad con versiones anteriores basadas en booleanos
            show_cm2 = bool(data.get("show_area_cm2", True))
            global_mode = AREA_MODE_CM2 if show_cm2 else AREA_MODE_OFF
        if global_mode == AREA_MODE_INHERIT:
            global_mode = AREA_MODE_CM2
        self._area_label_mode_flag = global_mode
        self.area_label_mode_display.set(AREA_MODE_DISPLAY.get(global_mode, AREA_MODE_DISPLAY[AREA_MODE_CM2]))
        self._recompute_area_mode_flags()

        if "conf" in data:
            try:
                val = float(data["conf"])
                self.conf = val
                self.conf_var.set(val)
            except Exception:
                pass
        if "iou" in data:
            try:
                val = float(data["iou"])
                self.iou = val
                self.iou_var.set(val)
            except Exception:
                pass

        _set_bool(self.show_boxes, "show_boxes")
        _set_bool(self.show_names, "show_names")
        _set_bool(self.show_confidence, "show_confidence")
        _set_bool(self.show_masks, "show_masks")
        _set_bool(self.masks_as_contours, "masks_as_contours")
        _set_bool(self.use_retina_masks, "use_retina_masks")
        _set_bool(self.highlight_tiny, "highlight_tiny")
        _set_bool(self.auto_skip, "auto_skip")
        _set_bool(self.filter_in_model, "filter_in_model")
        _set_bool(self.agnostic_nms, "agnostic_nms")
        _set_bool(self.use_half, "use_half")
        _set_bool(self.enable_m1, "enable_m1")
        _set_bool(self.enable_m2, "enable_m2")

        if "perf_mode" in data:
            try:
                self.perf_mode.set(str(data["perf_mode"]))
            except Exception:
                pass

        _set_int(self.target_fps, "target_fps")
        _set_int(self.det_stride, "det_stride")
        _set_int(self.stride2_var, "stride2_var")
        _set_int(self.imgsz1_var, "imgsz1")
        _set_int(self.imgsz2_var, "imgsz2")
        _set_int(self.max_det_var, "max_det")
        _set_int(self.topk_draw, "topk_draw")
        _set_int(self.line_thickness, "line_thickness")
        _set_float(self.font_scale, "font_scale")
        _set_float(self.area_text_scale, "area_text_scale")
        _set_int(self.ui_fps, "ui_fps")
        _set_int(self.snapshot_write_interval_ms, "snapshot_write_interval_ms")
        _set_float(self.snapshot_clean_interval_sec, "snapshot_clean_interval_sec")

        if "calibration" in data and isinstance(data["calibration"], dict):
            self._raw_calibration_config = dict(data["calibration"])
        else:
            self._raw_calibration_config = None

        self._load_calibration()

        if "class_cfg" in data and isinstance(data["class_cfg"], dict):
            global_mode = getattr(self, "_area_label_mode_flag", AREA_MODE_CM2)
            parsed: dict[str, dict] = {}
            for name, cfg in data["class_cfg"].items():
                if not isinstance(cfg, dict):
                    continue
                entry = dict(cfg)
                color = entry.get("color")
                if isinstance(color, (list, tuple)) and len(color) == 3:
                    entry["color"] = (int(color[0]), int(color[1]), int(color[2]))
                entry["thr_inherit"] = bool(entry.get("thr_inherit", True))
                entry["thr"] = float(_clamp01(entry.get("thr", float(self.conf)), float(self.conf)))
                cm2_m1_raw = entry.get("cm2_m1")
                if cm2_m1_raw is not None:
                    entry["cm2_m1"] = _area_enabled_from_value(cm2_m1_raw)
                else:
                    legacy_mode = _coerce_area_mode(entry.get("area_mode_m1"), AREA_MODE_INHERIT)
                    entry["cm2_m1"] = (
                        global_mode != AREA_MODE_OFF
                        if legacy_mode == AREA_MODE_INHERIT
                        else legacy_mode != AREA_MODE_OFF
                    )
                cm2_m2_raw = entry.get("cm2_m2")
                if cm2_m2_raw is not None:
                    entry["cm2_m2"] = _area_enabled_from_value(cm2_m2_raw)
                else:
                    legacy_mode = _coerce_area_mode(entry.get("area_mode_m2"), AREA_MODE_INHERIT)
                    entry["cm2_m2"] = (
                        global_mode != AREA_MODE_OFF
                        if legacy_mode == AREA_MODE_INHERIT
                        else legacy_mode != AREA_MODE_OFF
                    )
                parsed[str(name)] = entry
            if parsed:
                self.class_cfg = parsed
                self._recompute_area_mode_flags()

        # Cargar configuraci?n del sectorizador
        if "sectorizador" in data and isinstance(data["sectorizador"], dict):
            sect_cfg = data["sectorizador"]
            if "borde_superior" in sect_cfg:
                self.sector_borde_sup.set(str(sect_cfg["borde_superior"]))
            if "borde_inferior" in sect_cfg:
                self.sector_borde_inf.set(str(sect_cfg["borde_inferior"]))
            if "borde_izquierdo" in sect_cfg:
                self.sector_borde_izq.set(str(sect_cfg["borde_izquierdo"]))
            if "borde_derecho" in sect_cfg:
                self.sector_borde_der.set(str(sect_cfg["borde_derecho"]))
            if "modo" in sect_cfg:
                self.sector_modo.set(str(sect_cfg["modo"]))
            if "num_verticales" in sect_cfg:
                try:
                    self.sector_num_vert.set(int(sect_cfg["num_verticales"]))
                except Exception:
                    pass
            if "num_horizontales" in sect_cfg:
                try:
                    self.sector_num_horiz.set(int(sect_cfg["num_horizontales"]))
                except Exception:
                    pass
            if "mostrar" in sect_cfg:
                try:
                    self.sector_mostrar.set(bool(sect_cfg["mostrar"]))
                except Exception:
                    pass
            if "mostrar_etiquetas" in sect_cfg:
                try:
                    self.sector_mostrar_etiquetas.set(bool(sect_cfg["mostrar_etiquetas"]))
                except Exception:
                    pass
            if "mostrar_borde_banda" in sect_cfg:
                try:
                    self.sector_mostrar_borde_banda.set(bool(sect_cfg["mostrar_borde_banda"]))
                except Exception:
                    pass
            if "opacidad_lineas" in sect_cfg:
                try:
                    self.sector_opacidad_lineas.set(float(sect_cfg["opacidad_lineas"]))
                except Exception:
                    pass
            if "grosor_lineas" in sect_cfg:
                try:
                    self.sector_grosor_lineas.set(int(sect_cfg["grosor_lineas"]))
                except Exception:
                    pass
            # ParÃŸmetros de perspectiva
            if "use_perspective" in sect_cfg:
                try:
                    self.sector_use_perspective.set(bool(sect_cfg["use_perspective"]))
                except Exception:
                    pass
            if "use_border_masks" in sect_cfg:
                try:
                    self.sector_use_masks.set(bool(sect_cfg["use_border_masks"]))
                except Exception:
                    pass
            if "modo_delimitacion" in sect_cfg:
                try:
                    self.sector_modo_delimitacion.set(str(sect_cfg["modo_delimitacion"]))
                except Exception:
                    pass
            if "estabilidad" in sect_cfg:
                try:
                    self.sector_estabilidad.set(str(sect_cfg["estabilidad"]))
                except Exception:
                    pass
            if "comportamiento_fallo" in sect_cfg:
                try:
                    self.sector_comportamiento_fallo.set(str(sect_cfg["comportamiento_fallo"]))
                except Exception:
                    pass
            if "smooth_alpha" in sect_cfg:
                try:
                    self.sector_smooth_alpha.set(float(sect_cfg["smooth_alpha"]))
                except Exception:
                    pass
            if "max_corner_jump_px" in sect_cfg:
                try:
                    self.sector_max_jump.set(float(sect_cfg["max_corner_jump_px"]))
                except Exception:
                    pass
            if "inset_px" in sect_cfg:
                try:
                    self.sector_inset.set(int(sect_cfg["inset_px"]))
                except Exception:
                    pass
            if "debug_overlay" in sect_cfg:
                try:
                    self.sector_debug_overlay.set(bool(sect_cfg["debug_overlay"]))
                except Exception:
                    pass
            # NUEVO: Bordes curvos
            if "curved_edges_enabled" in sect_cfg:
                try:
                    self.sector_curved_enabled.set(bool(sect_cfg["curved_edges_enabled"]))
                except Exception:
                    pass
            if "curved_bins_vertical" in sect_cfg:
                try:
                    self.sector_curved_bins_vert.set(int(sect_cfg["curved_bins_vertical"]))
                except Exception:
                    pass
            if "curved_bins_horizontal" in sect_cfg:
                try:
                    self.sector_curved_bins_horiz.set(int(sect_cfg["curved_bins_horizontal"]))
                except Exception:
                    pass
            # NUEVO: Padding
            if "padding_top_px" in sect_cfg:
                try:
                    self.sector_padding_top.set(int(sect_cfg["padding_top_px"]))
                except Exception:
                    pass
            if "padding_bottom_px" in sect_cfg:
                try:
                    self.sector_padding_bottom.set(int(sect_cfg["padding_bottom_px"]))
                except Exception:
                    pass
            if "padding_left_px" in sect_cfg:
                try:
                    self.sector_padding_left.set(int(sect_cfg["padding_left_px"]))
                except Exception:
                    pass
            if "padding_right_px" in sect_cfg:
                try:
                    self.sector_padding_right.set(int(sect_cfg["padding_right_px"]))
                except Exception:
                    pass
            if "roi_quant_step_px" in sect_cfg:
                try:
                    self.sector_roi_quant_step.set(int(sect_cfg["roi_quant_step_px"]))
                except Exception:
                    pass
            if "line_quant_step_px" in sect_cfg:
                try:
                    self.sector_line_quant_step.set(int(sect_cfg["line_quant_step_px"]))
                except Exception:
                    pass
            # NUEVO: Restricciones por clase
            if "restricciones_enabled" in sect_cfg:
                try:
                    self.restricciones_enabled.set(bool(sect_cfg["restricciones_enabled"]))
                except Exception:
                    pass
            if "restricciones_por_clase" in sect_cfg and isinstance(sect_cfg["restricciones_por_clase"], dict):
                self._restricciones_por_clase = dict(sect_cfg["restricciones_por_clase"])

        if isinstance(self.config.get("sectores"), dict):
            if "restricciones_clase" not in self.config["sectores"]:
                self.config["sectores"]["restricciones_clase"] = dict(self._restricciones_por_clase)
            if "sensibilidades" not in self.config["sectores"] and "ajustes_locales" in self.config["sectores"]:
                self.config["sectores"]["sensibilidades"] = dict(self.config["sectores"].get("ajustes_locales", {}))
        else:
            self.config["sectores"] = {"restricciones_clase": dict(self._restricciones_por_clase)}

        # Asegurar coherencia si no se pudieron cargar todos los campos
        self.conf = float(self.conf_var.get())
        self.iou = float(self.iou_var.get())

    def _snapshot_core_settings(self) -> dict[str, object]:
        return {
            "model_path": self.model_path.get().strip(),
            "model_path2": self.model_path2.get().strip(),
            "source_mode": self.source_mode.get(),
            "video_path": self.video_path.get().strip(),
            "rtsp_manual_enabled": bool(self.rtsp_manual_enabled.get()),
            "rtsp_manual_url": self.rtsp_manual_url.get().strip(),
            "rtsp_user": self.rtsp_user.get().strip(),
            "rtsp_password": self.rtsp_password.get().strip(),
            "rtsp_host": self.rtsp_host.get().strip(),
            "rtsp_port": self.rtsp_port.get().strip(),
            "rtsp_path": self.rtsp_path.get().strip(),
        }

    def _finish_restart(self) -> None:
        try:
            self.start()
        finally:
            self._restart_in_progress = False

    def _restart_if_core_settings_changed(self) -> None:
        current = self._snapshot_core_settings()
        if self._last_core_settings == current:
            return
        self._last_core_settings = current
        if not self.running:
            return
        if self._restart_in_progress:
            return
        self._restart_in_progress = True
        try:
            self._set_status("Reiniciando para aplicar ajustes...")
            self._ui_status_message.set("Reiniciando para aplicar ajustes...")
        except Exception:
            pass
        self.stop()
        self.root.after(200, self._finish_restart)

    def _load_config(self) -> None:
        print(f"[Config] Cargando configuración desde: {CONFIG_PATH}", flush=True)
        if not os.path.isfile(CONFIG_PATH):
            print(f"[Config] Archivo no encontrado: {CONFIG_PATH}", flush=True)
            self._apply_config_payload({})
            self._profiles_bootstrap_from_config()
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                print(f"[Config] Datos cargados: {len(data)} claves principales.", flush=True)
        except Exception as exc:
            print(f"[Config] Error crítico leyendo configuración: {exc}", flush=True)
            self._apply_config_payload({})
            self._profiles_bootstrap_from_config()
            return

        if not isinstance(data, dict):
            data = {}

        self._apply_config_payload(data)
        if self.sectorizador is not None:
            self._on_sector_config_change()
            self._apply_restricciones_to_sectorizador()
        self._profiles_bootstrap_from_config()
        self._update_runtime_cache()

    def _load_calibration(self) -> None:
        raw = self._raw_calibration_config
        if raw is None:
            raw = dict(DEFAULT_CALIBRATION)
            self._raw_calibration_config = dict(raw)

        normalized = _normalize_calibration(raw)
        self.calibration = None
        self._calibration_valid = False

        if normalized is None:
            if not self._calibration_missing_logged:
                LOGGER.warning("Falta calibraciÃ³n para cÃ¡lculo de cmÂ², se devolverÃ¡ Ã¡rea en pÃ­xeles.")
                self._calibration_missing_logged = True
            return

        if normalized["row_far_px"] <= normalized["row_near_px"]:
            if not self._calibration_error_logged:
                LOGGER.warning(
                    "CalibraciÃ³n invÃ¡lida: row_far_px (%.1f) debe ser mayor que row_near_px (%.1f). Se usarÃ¡ Ã¡rea en pÃ­xeles.",
                    normalized["row_far_px"],
                    normalized["row_near_px"],
                )
                self._calibration_error_logged = True
            return

        self.calibration = {key: float(normalized[key]) for key in CALIBRATION_REQUIRED_KEYS}
        self._raw_calibration_config = dict(self.calibration)
        self._calibration_valid = True
        self._calibration_missing_logged = False
        self._calibration_error_logged = False
        self._calibration_debug_logged = False

        LOGGER.info(
            "CalibraciÃ³n cargada (altura=%.1f cm, visible=%.1f cm, filas=%.0f-%.0f, A4=%.1f cm @ %.1f cm, %.1f px)",
            self.calibration["cam_height_cm"],
            self.calibration["visible_len_cm"],
            self.calibration["row_near_px"],
            self.calibration["row_far_px"],
            self.calibration["a4_real_cm"],
            self.calibration["a4_dist_cm"],
            self.calibration["a4_px"],
        )

    # ------------------------- Control ejecuciÃ³n -------------------------
    def start(self):
        if self.running:
            return
        model_path = self.model_path.get().strip()
        model_path2 = self.model_path2.get().strip()
        mode = self.source_mode.get()
        if mode == "RTSP":
            video_path = self._build_rtsp_url()
            if not video_path:
                messagebox.showerror("RTSP", "Configura la conexion RTSP antes de iniciar.")
                return
        else:
            video_path = self.video_path.get().strip()
        if not os.path.isfile(model_path):
            messagebox.showerror("Error", f"No existe el modelo:\n{model_path}")
            return
        if model_path2 and (not os.path.isfile(model_path2)):
            messagebox.showerror("Error", f"No existe el modelo 2:\n{model_path2}")
            return
        if mode != "RTSP" and not os.path.isfile(video_path):
            messagebox.showerror("Error", f"No existe el video:\n{video_path}")
            return

        self._current_source_kind = mode
        self._current_input_source = video_path
        self._last_core_settings = self._snapshot_core_settings()

        # Inicializar GPU y optimizaciones
        try:
            self.device = get_cuda_device_or_die()
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            # Reducir overhead de CPU en planificador/BLAS
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            try:
                cv2.setNumThreads(0)
            except Exception:
                pass
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except SystemExit as e:
            messagebox.showerror("CUDA", str(e))
            return

        # Lanzar hilo de trabajo
        self.running = True
        self.paused = False
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal", text="Pausar")
        self.btn_stop.config(state="normal")
        self._set_status("Cargando modelo...")
        self._apply_runtime_priority(True)
        
        # Actualizar estado UI moderno
        self._ui_main_state.set("En marcha")
        self._ui_status_message.set("Cargando modelo...")
        self._update_ui_model_names()

        self._schedule_snapshot_flush()
        self._write_initial_snapshot()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        mark_detector_started()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="Reanudar")
            self._ui_main_state.set("Pausado")
            self._ui_status_message.set("Deteccion pausada")
        else:
            self.btn_pause.config(text="Pausar")
            self._ui_main_state.set("En marcha")
            self._ui_status_message.set("Deteccion en curso")

            self._ui_status_message.set("Deteccion en curso")

    def _open_fps_config(self, event=None):
        """Abre el diÃ¡logo de configuraciÃ³n de FPS."""
        if FPSConfigDialog is None:
            return
        
        # DiÃ¡logo Modal
        dlg = FPSConfigDialog(self.root, self.fps_config)
        if dlg.result:
            self.fps_config.update(dlg.result)
            self.config["fps_settings"] = self.fps_config
            self._save_config()

    def _update_fps_ui_color(self, fps_val: float):
        """Actualiza el valor y color del FPS UI segÃºn configuraciÃ³n."""
        cfg = getattr(self, "fps_config", {})
        low = cfg.get("low_thresh", 15)
        med = cfg.get("med_thresh", 24)
        
        color = cfg.get("high_color", "#388e3c")
        if fps_val < low:
            color = cfg.get("low_color", "#d32f2f")
        elif fps_val < med:
            color = cfg.get("med_color", "#f57c00")
            
        if self._fps_display:
            try:
                self._ui_fps_value.set(f"{fps_val:.1f}")
                self._fps_display.config(foreground=color)
            except Exception:
                pass

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self._release_resources()
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled", text="Pausar")
        self.btn_stop.config(state="disabled")
        self._set_status("Detenido.")
        
        # Limpiar widgets
        if self._dashboard_widget_manager:
            try:
                self._dashboard_widget_manager.cleanup()
            except Exception:
                pass
        
        # Actualizar estado UI moderno
        self._ui_main_state.set("Detenido")
        self._ui_status_message.set("Listo para iniciar.")
        self._ui_fps_value.set("--")
        
        mark_detector_stopped()

    def on_close(self):
        try:
            self.stop()
            self._save_config()
            self._stop_snapshot_flush()
            self._stop_heartbeat()
            self._destroy_plc_window()
            self._stop_plc_service()
            if garbage_collector:
                garbage_collector.stop_garbage_collector()
        finally:
            self.root.destroy()

    # ------------------------- LÃ³gica de trabajo -------------------------
    
    # ===================== GATING DE INFERENCIA (NUEVO V2) =====================
    def _sanitize_sectores(self):
        """Obtiene nÃºmero de sectores, habilitados, excluidos y restricciones."""
        n = self.sectorizador.obtener_num_sectores() if self.sectorizador else 0
        all0 = set(range(n))
        
        sect_cfg = self.config.get("sectores", {})
        if not isinstance(sect_cfg, dict):
            sect_cfg = {}
        excl_1 = sect_cfg.get("excluidos", [])
        if not isinstance(excl_1, list):
            excl_1 = []
        excl_0 = {int(s) - 1 for s in excl_1 if isinstance(s, int) and 1 <= s <= n}
        
        enabled0 = all0 - excl_0
        restricciones = sect_cfg.get("restricciones_clase", {})
        if not isinstance(restricciones, dict):
            restricciones = {}
        return n, enabled0, excl_0, restricciones
    
    def _allowed_anywhere(self, class_name, enabled0, restricciones):
        """Retorna True si la clase puede aparecer en alguna zona habilitada."""
        cfg = restricciones.get(class_name)
        if not isinstance(cfg, dict):
            return True  # sin restricciÃ³n = permitida
        
        modo = cfg.get("modo", "sin_restriccion")
        sectores_cfg = cfg.get("sectores", [])
        if not isinstance(sectores_cfg, list):
            sectores_cfg = []
        sectores0 = {int(s) for s in sectores_cfg if isinstance(s, int)}
        
        if modo == "sin_restriccion":
            return True
        if modo == "solo_fuera_malla":
            return True  # siempre hay "fuera"
        if modo == "solo_malla":
            return len(enabled0) > 0  # si hay al menos 1 sector habilitado
        if modo == "solo_sectores":
            return len(sectores0 & enabled0) > 0  # si hay intersecciÃ³n
        return True
    
    def _effective_class_ids(self, sel_ids, model_names, enabled0, restricciones):
        """Filtra IDs de clases dejando solo las que pueden aparecer en alguna zona."""
        # Si sel_ids es None, consideramos TODAS las clases del modelo
        candidates = sel_ids if sel_ids is not None else list(model_names.keys())
        
        out = []
        for cid in candidates:
            # Asegurar que cid sea int
            try:
                cid_int = int(cid)
            except (ValueError, TypeError):
                continue
                
            name = model_names.get(cid_int, str(cid_int))
            if self._allowed_anywhere(name, enabled0, restricciones):
                out.append(cid_int)
        return out  # retorna lista vacÃ­a si ninguna clase es viable
    # ==========================================================================
    
    def _worker_loop(self):
        self._boost_current_thread_priority()
        model_path = self.model_path.get().strip()
        model_path2 = self.model_path2.get().strip()
        mode = self._current_source_kind or self.source_mode.get()
        video_path = self._current_input_source
        if not video_path:
            if mode == "RTSP":
                video_path = self._build_rtsp_url()
            else:
                video_path = self.video_path.get().strip()
        out_path = self.out_path.get().strip()
        save_out = self.save_out

        try:
            # Carga de modelos (solo .pt para ruta rÃ¡pida)
            self.using_trt = model_path.lower().endswith(".engine")
            self.model = YOLO(model_path)
            if model_path2:
                self.using_trt2 = model_path2.lower().endswith(".engine")
                self.model2 = YOLO(model_path2)
                self._set_status("Modelos cargados. Abriendo video...")
            else:
                self.model2 = None
                self._set_status("Modelo cargado. Abriendo video...")

            # === FAST TORCH BACKEND (dos streams) ===
            use_half = bool(self.use_half.get())
            self._use_half_active = use_half
            self.net1 = self.model.model.eval().to("cuda")
            self.net1 = self._set_model_precision(self.net1, use_half)
            if self.model2 is not None:
                self.net2 = self.model2.model.eval().to("cuda")
                self.net2 = self._set_model_precision(self.net2, use_half)
            else:
                self.net2 = None

            torch.backends.cudnn.benchmark = True
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream() if self.net2 is not None else None
            self.stream_preproc = torch.cuda.Stream()
            # ========================================

            # Warm-up rÃ¡pido para estabilizar kernels
            try:
                dummy = np.zeros((360, 640, 3), dtype=np.uint8)
                _ = self._infer_one_fast(
                    net=self.net1,
                    names=self.model.names,
                    frame_bgr=dummy,
                    imgsz=max(320, int(round(self.imgsz / 32) * 32)),
                    stream=self.stream1,
                    conf=0.01,
                    iou=0.5,
                    classes_ids=None,
                    agnostic=False,
                    max_det=1,
                )
                if self.net2 is not None:
                    _ = self._infer_one_fast(
                        net=self.net2,
                        names=self.model2.names,
                        frame_bgr=dummy,
                        imgsz=max(320, int(round(self.imgsz / 32) * 32)),
                        stream=self.stream2,
                        conf=0.01,
                        iou=0.5,
                        classes_ids=None,
                        agnostic=False,
                        max_det=1,
                    )
            except Exception:
                pass

            # Preparar mapeo de clases y configuraciÃ³n por defecto (si no existe)
            def _names_to_maps(m):
                try:
                    nm = m.names if isinstance(m.names, dict) else {}
                    # claves pueden ser int->str, invertimos a name->id
                    return {str(v): int(k) for k, v in nm.items()}
                except Exception:
                    return {}
            self.name_to_id_m1 = _names_to_maps(self.model)
            self.name_to_id_m2 = _names_to_maps(self.model2) if self.model2 is not None else {}
            self._ensure_default_class_cfg()
            self._apply_class_config()  # calcula sel_ids_m1/m2

            in_fps = 0.0
            in_w = 0
            in_h = 0

            if mode == "RTSP":
                url = _ensure_rtsp_transport_param(video_path, self.rtsp_out_transport.get())
                url = _add_low_latency_flags(url)
                self._current_input_url = url
                self.cap = cv2.VideoCapture(url)
                cap_ok = self.cap is not None and self.cap.isOpened()
                if not cap_ok:
                    LOGGER.warning(
                        "RTSP no disponible en el arranque (%s). El lector seguirÃ¡ reintentando en segundo plano.",
                        url,
                    )
                    self._set_rtsp_state("reintentando", "sin senal inicial")
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                    self.cap = None
                    try:
                        self._set_status("Esperando RTSP...")
                    except Exception:
                        pass
                else:
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    except Exception:
                        pass
                    in_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                    in_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    in_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._rtsp_last_frame_ts = time.monotonic()
                    self._rtsp_last_frame_wall = time.time()
                    self._set_rtsp_state("transmitiendo", "conexion inicial establecida")
            else:
                self._current_input_url = None
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError("No se pudo abrir el video o stream.")
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass
                in_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                in_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                in_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if in_w > 0 and in_h > 0:
                self._configure_stream_outputs(in_w, in_h, in_fps or 25.0)
            else:
                self._out_spec = None

            self._set_status("Procesando...")

            # Lanzar lector asÃ­ncrono de frames
            self.frame_q = queue.Queue(maxsize=2)
            self.read_stop = False
            self.read_thread = threading.Thread(target=self._reader_loop, args=(self.frame_q,), daemon=True)
            self.read_thread.start()

            # Cola y hilo de dibujo/escritura
            self.draw_queue = queue.Queue(maxsize=2)  # Aumentado de 1 a 2 para absorber picos
            self.draw_stop = False
            self._draw_perf_total = 0.0
            self._draw_perf_calls = 0
            self._draw_stride_counter = 0
            self.draw_thread = threading.Thread(target=self._draw_writer_loop, daemon=True)
            self.draw_thread.start()

            frame_count = 0
            t0 = time.time()
            perf_m1_time = 0.0
            perf_m2_time = 0.0
            perf_m1_calls = 0
            perf_m2_calls = 0

            # Inicializar serie FPS para grÃ¡fica
            try:
                with self.perf_lock:
                    self.fps_series.clear()
            except Exception:
                pass
            fps_sample_last_time = t0
            fps_sample_last_count = 0
            fps_ui_last_time = t0

            self._last_result1 = None
            self._last_result2 = None
            self._det_counter = 0
            self._dyn_skip = 0

            def _process_packet(packet: _FramePacket | None) -> None:
                nonlocal frame_count, fps_sample_last_time, fps_sample_last_count, fps_ui_last_time
                if packet is None or packet.frame is None:
                    return

                result1 = self._last_result1
                result2 = self._last_result2

                if packet.job1 is not None:
                    try:
                        result1 = self._pack_to_cpu(packet.job1, "M1")
                    except Exception as pack_err:
                        print(f"[Pack] Error M1: {pack_err}", flush=True)
                        result1 = None
                    self._last_result1 = result1

                if packet.job2 is not None:
                    try:
                        result2 = self._pack_to_cpu(packet.job2, "M2")
                    except Exception as pack_err:
                        print(f"[Pack] Error M2: {pack_err}", flush=True)
                        result2 = None
                    self._last_result2 = result2

                log_results = self._debug_result_samples < 3
                if self._debug_log_stride > 0 and (frame_count % self._debug_log_stride == 0):
                    log_results = True
                if log_results and LOGGER.isEnabledFor(logging.DEBUG):
                    n1 = int(result1.xyxy_np().shape[0]) if (result1 is not None and result1.xyxy_np() is not None) else 0
                    n2 = int(result2.xyxy_np().shape[0]) if (result2 is not None and result2.xyxy_np() is not None) else 0
                    LOGGER.debug(
                        "[Debug] results frame=%s job1=%s job2=%s dets_m1=%s dets_m2=%s",
                        frame_count,
                        packet.job1 is not None,
                        packet.job2 is not None,
                        n1,
                        n2,
                    )
                if log_results:
                    self._debug_result_samples += 1

                # Actualizar contadores para widgets del dashboard
                try:
                    current_dets = []
                    counts = {}
                    
                    for res in (result1, result2):
                        if res is not None:
                            cls_ids = res.cls_np()
                            if cls_ids is not None:
                                names_map = res.names
                                for c_id in cls_ids:
                                    c_id_int = int(c_id)
                                    name = names_map.get(c_id_int, str(c_id_int))
                                    counts[name] = counts.get(name, 0) + 1
                                    current_dets.append({"class": name})
                    
                    self._current_frame_detections = current_dets
                    self._last_detection_counts = counts
                except Exception:
                    pass

                payload = (packet.frame, result1, result2)
                if self.draw_queue is not None:
                    try:
                        self.draw_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            _ = self.draw_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.draw_queue.put_nowait(payload)
                        except Exception:
                            pass

                self._det_counter += 1

                # Actualizar estado con nÂº de detecciones por frame
                try:
                    def _count(result):
                        _, _, clss, _ = self._result_arrays(result)
                        return int(len(clss)) if clss is not None else 0

                    n1 = _count(result1)
                    n2 = _count(result2)
                    parts = []
                    if result2 is not None:
                        parts.append(f"M1:{n1}")
                        parts.append(f"M2:{n2}")
                    else:
                        parts.append(f"Dets:{n1}")

                    if frame_count - self._last_counts_frame >= 30:
                        self._last_counts_frame = frame_count
                        self._last_total1 = n1
                        self._last_total2 = n2
                        try:
                            import numpy as _np

                            def _class_counts(res):
                                out = {}
                                if res is None:
                                    return out
                                _, _, cls_ids, names = self._result_arrays(res)
                                if cls_ids is None or len(cls_ids) == 0:
                                    return out
                                uniq, counts = _np.unique(cls_ids, return_counts=True)
                                for cid, cnt in zip(uniq, counts):
                                    name = names.get(int(cid), str(int(cid))) if isinstance(names, dict) else str(int(cid))
                                    out[name] = int(cnt)
                                return out

                            self._last_cls_counts1 = _class_counts(result1)
                            self._last_cls_counts2 = _class_counts(result2) if result2 is not None else {}
                        except Exception:
                            pass

                    def _fmt(cc):
                        return ", ".join(f"{k}:{v}" for k, v in cc.items()) if cc else ""

                    extra = " | ".join(filter(None, [_fmt(self._last_cls_counts1), _fmt(self._last_cls_counts2) if result2 is not None else ""]))
                    msg = "Procesando... " + "  ".join(parts)
                    if extra:
                        msg += f" | {extra}"

                    try:
                        rt = self._runtime_cfg
                        if bool(rt.get("auto_skip", True)):
                            now = time.time()
                            inst_fps = frame_count / max(1e-6, (now - t0)) if frame_count else 0.0
                            self._snapshot_last_inst_fps = inst_fps
                            target = max(5, int(rt.get("target_fps", 25)))
                            if inst_fps < target * 0.85 and self._dyn_skip < 4:
                                self._dyn_skip += 1
                            elif inst_fps > target * 1.15 and self._dyn_skip > 0:
                                self._dyn_skip -= 1
                            
                            # NUEVO: Flush agresivo de cola post-rÃ¡faga
                            # Cuando hay pocas detecciones pero FPS sigue bajo = backlog pendiente
                            current_det_count = sum(self._last_detection_counts.values()) if self._last_detection_counts else 0
                            if current_det_count < 5 and inst_fps < target * 0.7:
                                try:
                                    flushed = 0
                                    while not self.frame_q.empty():
                                        self.frame_q.get_nowait()
                                        flushed += 1
                                    if flushed > 0:
                                        self._dyn_skip = 0  # Reset para procesar frames frescos
                                except Exception:
                                    pass
                            
                            msg += f" | skip:{self._dyn_skip}"
                    except Exception:
                        pass

                    self._set_status(msg)
                except Exception:
                    pass

                frame_count += 1
                # Muestreo de FPS para grÃ¡fica (cada ~0.5 s)
                now_time = time.time()
                if (now_time - fps_sample_last_time) >= 0.5:
                    fps_inst = (frame_count - fps_sample_last_count) / max(1e-6, (now_time - fps_sample_last_time))
                    
                    # Conteo de la clase rastreada (si hay alguna seleccionada)
                    tracked_count = 0.0
                    try:
                        tracked_val = self.perf_tracked_class.get()
                        if tracked_val:
                            tracked_count = float(self._last_detection_counts.get(tracked_val, 0))
                    except Exception:
                        pass

                    elapsed = now_time - t0
                    try:
                        with self.perf_lock:
                            self.fps_series.append((elapsed, fps_inst))
                            self.detection_series.append((elapsed, tracked_count))
                            cutoff = elapsed - PERF_HISTORY_SECONDS
                            while self.fps_series and self.fps_series[0][0] < cutoff:
                                self.fps_series.popleft()
                            while self.detection_series and self.detection_series[0][0] < cutoff:
                                self.detection_series.popleft()
                    except Exception:
                        pass
                    fps_sample_last_time = now_time
                    fps_sample_last_count = frame_count
                # ActualizaciÃ³n de FPS UI configurable
                now_ui = time.time()
                if (now_ui - fps_ui_last_time) >= self.fps_config.get("interval", 0.5):
                    fps = 0.0
                    with self.perf_lock:
                        if self.fps_series:
                            fps = self.fps_series[-1][1]
                        else:
                            dt = now_ui - t0
                            fps = frame_count / max(1e-6, dt)
                    
                    try:
                        self.root.after(0, lambda f=fps: self._update_fps_ui_color(f))
                    except Exception:
                        pass
                    fps_ui_last_time = now_ui
                    self._snapshot_last_avg_fps = fps


                if frame_count % 120 == 0 and frame_count:
                    avg_m1 = (perf_m1_time / perf_m1_calls) if perf_m1_calls else 0.0
                    avg_m2 = (perf_m2_time / perf_m2_calls) if perf_m2_calls else 0.0
                    with self._draw_perf_lock:
                        draw_total = self._draw_perf_total
                        draw_calls = self._draw_perf_calls
                    avg_draw = (draw_total / draw_calls) if draw_calls else 0.0

            while self.running:
                if self.paused:
                    time.sleep(0.05)
                    continue
                rt = self._runtime_cfg
                use_half = bool(rt.get("use_half", True))
                if use_half != self._use_half_active:
                    self._use_half_active = use_half
                    self.net1 = self._set_model_precision(self.net1, use_half)
                    if self.net2 is not None:
                        self.net2 = self._set_model_precision(self.net2, use_half)

                # Obtiene el siguiente frame del lector
                try:
                    t_wait_start = time.perf_counter()
                    frame = self.frame_q.get(timeout=0.5)
                    t_wait_end = time.perf_counter()
                    
                    if self.perf_trace_enabled.get():
                        q_sz = self.frame_q.qsize()
                        wait_ms = (t_wait_end - t_wait_start) * 1000.0
                        self._log_perftrace_event("worker_wait", {
                            "wait_ms": round(wait_ms, 2),
                            "q_size": q_sz,
                            "skip_level": self._dyn_skip
                        })

                    if frame is None:
                        break
                except Exception:
                    continue

                # Skip dinÃ¡mico de frames si estÃ¡ activo
                if bool(rt.get("auto_skip", True)) and self._dyn_skip > 0:
                    try:
                        # Drena frames de la cola sin procesarlos
                        drain_n = min(self._dyn_skip, 3)
                        for _ in range(drain_n):
                            self.frame_q.get_nowait()
                    except Exception:
                        pass

                # ParÃ¡metros fijos (interfaz simplificada)
                conf = float(self.conf)
                iou = float(self.iou)
                try:
                    imgsz1 = int(rt.get("imgsz1", self.imgsz))
                except Exception:
                    imgsz1 = int(self.imgsz)
                try:
                    imgsz2 = int(rt.get("imgsz2", imgsz1)) if self.net2 is not None else imgsz1
                except Exception:
                    imgsz2 = imgsz1
                try:
                    max_det_val = int(rt.get("max_det", self.max_det))
                except Exception:
                    max_det_val = int(self.max_det)
                imgsz1 = max(320, int(round(int(imgsz1) / 32) * 32))
                imgsz2 = max(320, int(round(int(imgsz2) / 32) * 32))

                det_stride = max(1, int(rt.get("det_stride", 1)))
                do_detect = (self._det_counter % det_stride) == 0
                if do_detect:
                    self._last_result1 = None
                    self._last_result2 = None

                # ================ GATING DE INFERENCIA (SIMPLIFICADO) ================
                # NOTA: El gating complejo basado en restricciones fue simplificado
                # porque aÃ±adÃ­a overhead significativo (~2-3ms/frame).
                # El filtrado real se hace despuÃ©s de la inferencia, en _draw_result_on.
                
                filter_in_model = bool(rt.get("filter_in_model", True))
                classes1 = self.sel_ids_m1 if (self.sel_ids_m1 and filter_in_model) else None
                classes2 = self.sel_ids_m2 if (self.sel_ids_m2 and filter_in_model) else None
                
                run_m1 = bool(rt.get("enable_m1", True)) and do_detect and (self.net1 is not None)
                try:
                    stride2 = max(1, int(rt.get("stride2", 1)))
                except Exception:
                    stride2 = 3
                run_m2 = bool(rt.get("enable_m2", True)) and (self.net2 is not None) and ((self._det_counter % stride2) == 0)
                # ============================================================
                perf_mode = str(rt.get("perf_mode", "auto")).strip().lower()
                if perf_mode not in {"auto", "secuencial", "paralelo"}:
                    perf_mode = "auto"
                use_parallel = False
                if perf_mode == "paralelo":
                    use_parallel = self.stream2 is not None
                elif perf_mode == "auto":
                    use_parallel = run_m1 and run_m2 and self.stream2 is not None
                stream_m1 = self.stream1
                stream_m2 = self.stream2 if use_parallel else self.stream1

                log_launch = self._debug_launch_samples < 3
                if self._debug_log_stride > 0 and (frame_count % self._debug_log_stride == 0):
                    log_launch = True
                if log_launch and LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "[Debug] loop frame=%s det_counter=%s do_detect=%s run_m1=%s run_m2=%s classes1=%s classes2=%s enable_m1=%s enable_m2=%s stride2=%s",
                        frame_count,
                        self._det_counter,
                        do_detect,
                        run_m1,
                        run_m2,
                        classes1,
                        classes2,
                        bool(rt.get("enable_m1", True)),
                        bool(rt.get("enable_m2", True)),
                        stride2,
                    )
                if log_launch:
                    self._debug_launch_samples += 1

                new_job1: _InferLaunch | None = None
                new_job2: _InferLaunch | None = None
                try:
                    if run_m1:
                        t_start = time.perf_counter()
                        new_job1 = self._launch_infer(
                            net=self.net1,
                            names=self.model.names,
                            model_tag="M1",
                            frame_bgr=frame,
                            imgsz=imgsz1,
                            stream=stream_m1,
                            conf=conf,
                            iou=iou,
                            classes_ids=classes1,
                            agnostic=bool(rt.get("agnostic_nms", False)),
                            max_det=max_det_val,
                        )
                        if new_job1 is None:
                            LOGGER.error(
                                "[Debug] _launch_infer devolviÃ³ None para M1 (frame=%s, do_detect=%s)",
                                frame_count,
                                do_detect,
                            )
                        perf_m1_time += time.perf_counter() - t_start
                        perf_m1_calls += 1

                    if run_m2:
                        t_start = time.perf_counter()
                        new_job2 = self._launch_infer(
                            net=self.net2,
                            names=self.model2.names,
                            model_tag="M2",
                            frame_bgr=frame,
                            imgsz=imgsz2,
                            stream=stream_m2,
                            conf=conf,
                            iou=iou,
                            classes_ids=classes2,
                            agnostic=bool(rt.get("agnostic_nms", False)),
                            max_det=max_det_val,
                        )
                        if new_job2 is None:
                            LOGGER.error(
                                "[Debug] _launch_infer devolviÃ³ None para M2 (frame=%s, det_counter=%s)",
                                frame_count,
                                self._det_counter,
                            )
                        perf_m2_time += time.perf_counter() - t_start
                        perf_m2_calls += 1
                except torch.cuda.OutOfMemoryError:
                    imgsz1 = max(640, imgsz1 - 128)
                    self.imgsz1_var.set(imgsz1)
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    self._set_status(f"Error inferencia: {e}")
                    LOGGER.exception("[Debug] ExcepciÃ³n lanzando inferencia")

                current_packet = _FramePacket(frame=frame, job1=new_job1, job2=new_job2)

                _process_packet(current_packet)
                if current_packet is not None:
                    self._snapshot_register_results(current_packet.frame, self._last_result1, self._last_result2)

            # Notificar hilo de dibujo que no habrÃ¡ mÃ¡s frames
            if self.draw_queue is not None:
                try:
                    self.draw_queue.put(None, timeout=0.1)
                except Exception:
                    pass

        except Exception as e:
            messagebox.showerror("Ejecucion", str(e))
        finally:
            self._revert_mmcss_for_thread()
            self.running = False
            self._release_resources()
            self.btn_start.config(state="normal")
            self.btn_pause.config(state="disabled", text="Pausar")
            self.btn_stop.config(state="disabled")
            self._set_status("Finalizado.")
            mark_detector_stopped()

    def _release_resources(self):
        # Parar lector de frames
        try:
            self.read_stop = True
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=1.0)
        except Exception:
            pass
        self.read_thread = None
        self.frame_q = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
        self._stop_rtsp_stream()
        if self.snapshot_writer is not None:
            try:
                self.snapshot_writer.flush(force=True)
            except Exception:
                pass
        if self._rtsp_thread and self._rtsp_thread.is_alive():
            try:
                self._rtsp_thread.join(timeout=1.0)
            except Exception:
                pass
        self._rtsp_thread = None
        self._update_rtsp_indicator(False)
        try:
            self._plc_push_stop.set()
        except Exception:
            pass
        if self._plc_push_thread and self._plc_push_thread.is_alive():
            try:
                self._plc_push_thread.join(timeout=1.0)
            except Exception:
                pass
        self._plc_push_thread = None
        self._plc_push_queue = None
        # Soltamos referencias a modelos para que liberen VRAM cuando sea posible
        try:
            self.model = None
            self.model2 = None
            self.net1 = None
            self.net2 = None
            self.stream1 = None
            self.stream2 = None
            self.stream_preproc = None
        except Exception:
            pass
        if self._current_source_kind == "RTSP":
            try:
                self._set_rtsp_state("detenido", "recursos liberados")
            except Exception:
                pass
        self._rtsp_last_frame_ts = 0.0
        self._rtsp_last_frame_wall = 0.0
        self._apply_runtime_priority(False)

    def _apply_runtime_priority(self, enable: bool) -> None:
        if os.name != "nt":
            return
        if enable:
            if self._win_priority_applied:
                self._apply_power_throttling(True)
                self._apply_timer_resolution(True)
                return
            try:
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                HIGH_PRIORITY_CLASS = 0x00000080
                get_proc = kernel32.GetCurrentProcess
                get_prio = kernel32.GetPriorityClass
                set_prio = kernel32.SetPriorityClass
                prev = get_prio(get_proc())
                if prev:
                    self._win_prev_priority = int(prev)
                    set_prio(get_proc(), HIGH_PRIORITY_CLASS)
                    self._win_priority_applied = True
            except Exception:
                self._win_prev_priority = None
                self._win_priority_applied = False
            self._apply_power_throttling(True)
            self._apply_timer_resolution(True)
            # Prevenir que Windows reduzca la prioridad automÃ¡ticamente
            try:
                import ctypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                # SetProcessPriorityBoost: Desactivar reducciÃ³n automÃ¡tica de prioridad
                kernel32.SetProcessPriorityBoost(kernel32.GetCurrentProcess(), False)
                LOGGER.info("SetProcessPriorityBoost desactivado - prioridad fija")
            except Exception as e:
                LOGGER.debug("SetProcessPriorityBoost no disponible: %s", e)
        else:
            if not self._win_priority_applied:
                self._apply_power_throttling(False)
                self._apply_timer_resolution(False)
                return
            try:
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                get_proc = kernel32.GetCurrentProcess
                set_prio = kernel32.SetPriorityClass
                prev = self._win_prev_priority
                if prev:
                    set_prio(get_proc(), int(prev))
            except Exception:
                pass
            self._win_prev_priority = None
            self._win_priority_applied = False
            self._apply_power_throttling(False)
            self._apply_timer_resolution(False)

    def _apply_power_throttling(self, disable: bool) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not hasattr(kernel32, "SetProcessInformation"):
                return

            class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                _fields_ = [
                    ("Version", ctypes.c_ulong),
                    ("ControlMask", ctypes.c_ulong),
                    ("StateMask", ctypes.c_ulong),
                ]

            PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x00000001
            PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x00000002
            ProcessPowerThrottling = 0x00000004

            state = PROCESS_POWER_THROTTLING_STATE()
            state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION

            hProcess = kernel32.GetCurrentProcess()
            success_count = 0

            # Llamada 1: Desactivar throttling de velocidad de ejecuciÃ³n
            state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            if disable:
                # Disable execution throttling and keep timer resolution in background.
                state.StateMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
            else:
                state.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            res1 = kernel32.SetProcessInformation(
                hProcess,
                ProcessPowerThrottling,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
            if res1:
                success_count += 1

            # Llamada 2: Forzar que Windows respete la resoluciÃ³n del timer
            state.ControlMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
            state.StateMask = 0 if disable else PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
            res2 = kernel32.SetProcessInformation(
                hProcess,
                ProcessPowerThrottling,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
            if res2:
                success_count += 1

            if success_count > 0:
                self._power_throttling_disabled = disable
                LOGGER.info(
                    "Power throttling %s (%d/2 APIs exitosas)",
                    "desactivado" if disable else "reactivado",
                    success_count,
                )
            else:
                LOGGER.warning("SetProcessInformation fallÃ³: error %d", ctypes.get_last_error())
        except Exception as e:
            LOGGER.warning("Error al configurar power throttling: %s", e)

    def _apply_timer_resolution(self, enable: bool) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            winmm = ctypes.WinDLL("winmm", use_last_error=True)
            time_begin = winmm.timeBeginPeriod
            time_end = winmm.timeEndPeriod
            if enable:
                if not self._timer_resolution_active:
                    if time_begin(1) == 0:
                        self._timer_resolution_active = True
            else:
                if self._timer_resolution_active:
                    time_end(1)
                    self._timer_resolution_active = False
        except Exception:
            pass

    def _set_mmcss_for_thread(self) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            avrt = ctypes.WinDLL("avrt", use_last_error=True)
            if not hasattr(avrt, "AvSetMmThreadCharacteristicsW"):
                return

            task_index = ctypes.c_ulong(0)
            task_name = "Games"
            handle = avrt.AvSetMmThreadCharacteristicsW(task_name, ctypes.byref(task_index))
            if handle:
                try:
                    if hasattr(avrt, "AvSetMmThreadPriority"):
                        avrt.AvSetMmThreadPriority(handle, 2)
                except Exception:
                    pass
                setattr(self._mmcss_local, "handle", handle)
        except Exception:
            pass

    def _revert_mmcss_for_thread(self) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            handle = getattr(self._mmcss_local, "handle", None)
            if not handle:
                return
            avrt = ctypes.WinDLL("avrt", use_last_error=True)
            if hasattr(avrt, "AvRevertMmThreadCharacteristics"):
                avrt.AvRevertMmThreadCharacteristics(handle)
            setattr(self._mmcss_local, "handle", None)
        except Exception:
            pass

    def _boost_current_thread_priority(self) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            THREAD_SET_INFORMATION = 0x0020
            THREAD_QUERY_INFORMATION = 0x0040
            THREAD_PRIORITY_ABOVE_NORMAL = 1
            THREAD_PRIORITY_HIGHEST = 2
            tid = None
            if hasattr(threading, "get_native_id"):
                tid = threading.get_native_id()
            else:
                cur = threading.current_thread()
                tid = getattr(cur, "native_id", None)
            if not tid:
                return
            handle = kernel32.OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, False, int(tid))
            if handle:
                prio = THREAD_PRIORITY_HIGHEST if self._win_priority_applied else THREAD_PRIORITY_ABOVE_NORMAL
                kernel32.SetThreadPriority(handle, prio)
                try:
                    if hasattr(kernel32, "SetThreadInformation"):
                        class THREAD_POWER_THROTTLING_STATE(ctypes.Structure):
                            _fields_ = [
                                ("Version", ctypes.c_ulong),
                                ("ControlMask", ctypes.c_ulong),
                                ("StateMask", ctypes.c_ulong),
                            ]

                        THREAD_POWER_THROTTLING_CURRENT_VERSION = 1
                        THREAD_POWER_THROTTLING_EXECUTION_SPEED = 0x00000001
                        THREAD_POWER_THROTTLING_EFFICIENCY_MODE = 0x00000002
                        ThreadPowerThrottling = 0x00000005

                        state = THREAD_POWER_THROTTLING_STATE()
                        state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION
                        state.ControlMask = (
                            THREAD_POWER_THROTTLING_EXECUTION_SPEED | THREAD_POWER_THROTTLING_EFFICIENCY_MODE
                        )
                        state.StateMask = 0
                        kernel32.SetThreadInformation(
                            handle,
                            ThreadPowerThrottling,
                            ctypes.byref(state),
                            ctypes.sizeof(state),
                        )
                except Exception:
                    pass
                kernel32.CloseHandle(handle)
            self._set_mmcss_for_thread()
        except Exception:
            pass

    # ------------------------- Render UI loop -------------------------
    def _refresh_image_loop(self):
        try:
            fps_value = max(5, int(self.ui_fps.get()))
            min_interval = 1.0 / float(fps_value)
            now = time.monotonic()
            try:
                state = self.root.state()
                visible = state not in ("iconic", "withdrawn")
            except Exception:
                visible = True
            self._ui_visible = bool(visible)
            if not self._ui_visible:
                self._last_ui_refresh = now
                return
            self._flush_ui_cache(now)

            frame_ref = None
            seq = self._last_ui_seq
            with self.frame_lock:
                current_seq = self._preview_seq
                if self.last_frame_preview is not None and (
                    current_seq != self._last_ui_seq or (now - self._last_ui_refresh) >= min_interval * 1.5
                ):
                    frame_ref = self.last_frame_preview
                    seq = current_seq

            if frame_ref is not None and self.video_label is not None:
                if (now - self._last_ui_refresh) >= min_interval:
                    t0_ui = time.perf_counter()
                    try:
                        disp = self._prepare_display_frame(frame_ref)
                    except MemoryError:
                        disp = None
                        self._last_ui_refresh = now
                    t1_ui = time.perf_counter()
                    
                    if disp is not None:
                        self.photo = ImageTk.PhotoImage(image=disp)
                        self.video_label.configure(image=self.photo)
                        self.video_label.image = self.photo  # Mantener referencia
                        self._last_ui_refresh = now
                        self._last_ui_seq = seq
                    
                    t2_ui = time.perf_counter()
                    if self.perf_trace_enabled.get():
                        self._log_perftrace_event("ui_perf", {
                            "prep_ms": round((t1_ui - t0_ui)*1000.0, 2),
                            "tk_ms": round((t2_ui - t1_ui)*1000.0, 2),
                            "total_ms": round((t2_ui - t0_ui)*1000.0, 2)
                        })
        finally:
            try:
                period = int(1000 / max(5, int(self.ui_fps.get())))
            except Exception:
                period = 100
            self.root.after(period, self._refresh_image_loop)

    def _rtsp_publisher_loop(self) -> None:
        self._boost_current_thread_priority()
        import subprocess as sp, shutil, time, queue

        fail_streak = 0
        idle_counter = 0

        def has_nvenc(codec: str) -> bool:
            return codec == "Auto (NVENC)"

        while self.running:
            rt = self._runtime_cfg
            if not bool(rt.get("rtsp_out_enable", False)):
                time.sleep(0.1)
                continue
            if not self._out_spec:
                time.sleep(0.05)
                continue

            try:
                frame = self._rtsp_q.get(timeout=0.5)
            except queue.Empty:
                idle_counter += 1
                if idle_counter % 6 == 0:
                    self._rtsp_log(logging.DEBUG, "RTSP publisher sin frames recientes (cola vacÃ­a)")
                continue
            idle_counter = 0

            w, h, fps = self._out_spec
            transport = str(rt.get("rtsp_out_transport", "")).strip().upper()
            url_base = str(rt.get("rtsp_out_url", "")).strip()

            # Reemplazo opcional de placeholders estÃ¡ndar
            # Nota: se eliminÃ³ el sufijo automÃ¡tico con PID porque generaba paths
            # distintos a los que esperan los clientes/MediaMTX (dejaba la ruta vacÃ­a).
            # Solo se reemplaza si el usuario lo pide explÃ­citamente.
            if "{pid}" in url_base:
                url_base = url_base.replace("{pid}", str(os.getpid()))
            if "{uid}" in url_base:
                url_base = url_base.replace("{uid}", str(uuid.uuid4())[:8])

            url = _ensure_rtsp_transport_param(url_base, transport)

            ffmpeg_path = _ffmpeg_binary("ffmpeg")
            if not os.path.isfile(ffmpeg_path):
                self._rtsp_disable_toggle()
                self._rtsp_schedule_indicator(False)
                self._async_warning("FFmpeg", f"No se encontrÃ³ ffmpeg en {ffmpeg_path}.")
                continue

            if self._ffmpeg is None or self._ffmpeg.poll() is not None:
                if self._ffmpeg is not None and self._ffmpeg.poll() is not None:
                    self._close_ffmpeg_process(update_indicator=False)

                common = [
                    ffmpeg_path,
                    "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{w}x{h}",
                    "-r", f"{fps:.2f}",
                    "-i", "-",
                    "-an",
                ]

                rtsp_codec = str(rt.get("rtsp_out_codec", ""))
                if has_nvenc(rtsp_codec) and shutil.which("nvidia-smi"):
                    enc = [
                        "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "ll",
                        "-b:v", "6M", "-maxrate", "6M", "-bufsize", "2M",
                        "-g", "15", "-bf", "0", "-pix_fmt", "yuv420p",
                    ]
                else:
                    enc = [
                        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                        "-g", "15", "-bf", "0", "-pix_fmt", "yuv420p",
                    ]

                if transport == "UDP":
                    outp = ["-f", "rtsp", "-rtsp_transport", "udp", "-fflags", "nobuffer", "-flags", "low_delay", "-max_delay", "0", url]
                else:
                    outp = ["-f", "rtsp", "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay", "-max_delay", "0", url]

                try:
                    self._ffmpeg = sp.Popen(common + enc + outp, stdin=sp.PIPE)
                    fail_streak = 0
                    self._rtsp_schedule_indicator(True)
                    self._rtsp_log(logging.INFO, f"FFmpeg iniciado ({w}x{h}@{fps:.2f})->{url}")
                except Exception as exc:
                    self._ffmpeg = None
                    fail_streak += 1
                    if has_nvenc(rtsp_codec) and fail_streak == 1:
                        self._rtsp_log(logging.WARNING, f"NVENC fallÃ³: {exc}. Cambiando a libx264")
                        self._rtsp_set_codec("libx264")
                        self._async_info("RTSP", "NVENC no disponible. Usando libx264.")
                    elif fail_streak >= 3:
                        self._rtsp_disable_toggle()
                        self._rtsp_log(logging.ERROR, f"No se pudo iniciar RTSP tras varios intentos: {exc}")
                        self._async_warning("RTSP", f"No se pudo iniciar RTSP: {exc}")
                    continue

            try:
                if self._ffmpeg and self._ffmpeg.stdin:
                    self._ffmpeg.stdin.write(frame.tobytes())
                    while not self._rtsp_q.empty():
                        nxt = self._rtsp_q.get_nowait()
                        self._ffmpeg.stdin.write(nxt.tobytes())
                    self._rtsp_frames_sent += 1
                    now = time.monotonic()
                    if now - self._rtsp_last_stats >= 5.0:
                        self._rtsp_log(logging.INFO, f"Frames RTSP enviados: {self._rtsp_frames_sent}")
                        self._rtsp_last_stats = now
            except Exception as exc:
                self._close_ffmpeg_process(update_indicator=True)
                fail_streak += 1
                if fail_streak >= 3:
                    self._rtsp_disable_toggle()
                    self._rtsp_log(logging.ERROR, f"Error enviando frames a FFmpeg: {exc}")
                    self._async_warning("RTSP", f"Salida RTSP desactivada tras fallos: {exc}")
                continue

        self._stop_rtsp_stream()
        self._revert_mmcss_for_thread()

    def _async_warning(self, title: str, msg: str) -> None:
        try:
            self.root.after(0, lambda: messagebox.showwarning(title, msg))
        except Exception:
            pass

    def _async_info(self, title: str, msg: str) -> None:
        try:
            self.root.after(0, lambda: messagebox.showinfo(title, msg))
        except Exception:
            pass

    # ------------------------- Controles de umbrales -------------------------
    def _on_conf_change(self, val):
        try:
            self.conf = _clamp01(val, self.conf)
            self.lbl_conf_val.configure(text=f"{self.conf*100:.0f}%")
            self.conf_var.set(self.conf)
        except Exception:
            pass

    def _on_iou_change(self, val):
        try:
            self.iou = _clamp01(val, self.iou)
            self.lbl_iou_val.configure(text=f"{self.iou*100:.0f}%")
            self.iou_var.set(self.iou)
        except Exception:
            pass

    def _prepare_display_frame(self, bgr: np.ndarray) -> Image.Image:
        try:
            # Obtener dimensiones del Ã¡rea de visualizaciÃ³n
            # Usar ancho/alto del contenedor padre para calcular el espacio disponible real
            master = self.video_label.master
            if master:
                # Si el label estÃ¡ en pack(side=top), su height puede ser el de la imagen previa,
                # mientras que el master tiene el height total disponible.
                # Queremos escalar hasta el ancho del master o el alto del master.
                label_w = max(100, master.winfo_width())
                label_h = max(100, master.winfo_height())
            else:
                 label_w = max(100, self.video_label.winfo_width() or 100)
                 label_h = max(100, self.video_label.winfo_height() or 100)
            
            # Dimensiones de la imagen de origen
            h, w = bgr.shape[:2]
            
            # Calcular escala para ajustar manteniendo relaciÃ³n de aspecto
            scale_w = label_w / w
            scale_h = label_h / h
            scale = min(scale_w, scale_h)
            
            # Aplicar redimensionamiento
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            
            # Redimensionar manteniendo relaciÃ³n de aspecto
            resized = cv2.resize(bgr, (new_w, new_h), 
                               interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
            
            # Convertir directamente la imagen redimensionada
            # ya no creamos un canvas negro debajo.
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            self._cleanup_overlay_messages()
            if self._overlay_messages:
                base_rgba = pil_img.convert("RGBA")
                overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                y_cursor = 12
                for msg in self._overlay_messages:
                    text = msg.text.strip()
                    if not text:
                        continue
                    alpha = int(max(0.0, min(1.0, msg.opacity)) * 255)
                    if alpha <= 0:
                        continue
                    rgb = (msg.color[2], msg.color[1], msg.color[0])
                    anchor = (12, y_cursor)
                    bbox = draw.textbbox(anchor, text, font=_OVERLAY_FONT)
                    padding = 6
                    rect = (
                        bbox[0] - padding,
                        bbox[1] - padding,
                        bbox[2] + padding,
                        bbox[3] + padding,
                    )
                    bg_alpha = max(60, int(alpha * 0.6))
                    draw.rectangle(rect, fill=(0, 0, 0, bg_alpha))
                    draw.text(anchor, text, font=_OVERLAY_FONT, fill=(*rgb, alpha))
                    y_cursor = rect[3] + 8
                pil_img = Image.alpha_composite(base_rgba, overlay).convert("RGB")

            return pil_img
            
        except Exception as e:
            # En caso de error, devolver una imagen negra con un mensaje
            error_img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Error: {str(e)[:30]}...", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return Image.fromarray(cv2.cvtColor(error_img, cv2.COLOR_BGR2RGB))

    # ------------------------- Prefetch de vÃ­deo -------------------------
    def _configure_stream_outputs(self, width: int, height: int, fps: float | int | None) -> None:
        if width is None or height is None:
            return
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return
        fps_val = float(fps or 25.0)
        self._out_spec = (width, height, fps_val)

        if self.save_out and self.writer is None:
            out_path = self.out_path.get().strip()
            if out_path:
                try:
                    self.writer = build_writer(out_path, width, height, fps_val)
                except Exception as e:
                    self.writer = None
                    self.root.after(0, lambda err=e: messagebox.showwarning("Salida", f"No se pudo crear el archivo de salida:\n{err}"))

        if bool(self._runtime_cfg.get("rtsp_out_enable", False)) and (self._rtsp_thread is None or not self._rtsp_thread.is_alive()):
            self._rtsp_thread = threading.Thread(target=self._rtsp_publisher_loop, daemon=True)
            self._rtsp_thread.start()

    def _try_open_rtsp_capture(self) -> cv2.VideoCapture | None:
        url = self._current_input_url
        if not url:
            return None
        LOGGER.info("Reconectando RTSP: %s", url)
        cap = cv2.VideoCapture(url)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._configure_stream_outputs(width, height, fps)
        self.cap = cap
        LOGGER.info("Conexion RTSP restablecida: %s", url)
        now_mono = time.monotonic()
        now_wall = time.time()
        self._rtsp_last_frame_ts = now_mono
        self._rtsp_last_frame_wall = now_wall
        self._set_rtsp_state("transmitiendo", "reconexion exitosa")
        return cap

    def _reader_loop(self, q: queue.Queue):
        self._boost_current_thread_priority()
        self._last_read_success_ts = 0.0 # PerfTrace
        reconnect_delay = 1.0
        recon_notice = False
        cap = self.cap
        try:
            while self.running and not self.read_stop:
                now_mono = time.monotonic()
                if (
                    self._current_source_kind == "RTSP"
                    and self._current_input_url
                    and cap is not None
                     and cap.isOpened()
                     and self._rtsp_last_frame_ts > 0.0
                     and (now_mono - self._rtsp_last_frame_ts) >= RTSP_WATCHDOG_TIMEOUT_SEC
                 ):
                    LOGGER.warning(
                        "Watchdog RTSP sin frames (%.1fs). Reiniciando captura (%s)",
                        now_mono - self._rtsp_last_frame_ts,
                        self._current_input_url,
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass
                    if self.cap is cap:
                        self.cap = None
                    cap = None
                    self._set_rtsp_state("reintentando", "watchdog sin frames")
                    if not recon_notice:
                        try:
                            self._set_status("Watchdog RTSP. Reintentando...")
                        except Exception:
                            pass
                    recon_notice = True
                    reconnect_delay = min(reconnect_delay * 1.5, 8.0)
                    continue

                if cap is None or not cap.isOpened():
                    if self._current_source_kind == "RTSP" and self._current_input_url:
                        if not recon_notice:
                            LOGGER.warning("RTSP desconectado. Reintentando (%s)", self._current_input_url)
                            try:
                                self._set_status("Reconectando RTSP...")
                            except Exception:
                                pass
                            recon_notice = True
                            self._set_rtsp_state("reintentando", "captura cerrada")
                        if self.read_stop:
                            break
                        new_cap = self._try_open_rtsp_capture()
                        if new_cap is None:
                            time.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 1.5, 8.0)
                            continue
                        cap = new_cap
                        reconnect_delay = 1.0
                        recon_notice = False
                        self._set_rtsp_state("transmitiendo", "reconexion exitosa")
                        try:
                            self._set_status("Senal RTSP recuperada.")
                        except Exception:
                            pass
                        continue
                    break

                # Si el consumidor va atrasado, en RTSP evitamos asignar arrays grandes
                # usando grab() (descarta el frame sin decodificar a np.ndarray).
                if self._current_source_kind == "RTSP":
                    try:
                        is_full = bool(q.full())
                    except Exception:
                        is_full = False
                    if is_full:
                        try:
                            ok_grab = bool(cap.grab())
                        except (MemoryError, SystemError, cv2.error):
                            ok_grab = False
                        except Exception:
                            ok_grab = False
                        if ok_grab:
                            self._rtsp_last_frame_ts = time.monotonic()
                            self._rtsp_last_frame_wall = time.time()
                            continue

                try:
                    t_read_start = time.perf_counter()
                    ok, fr = cap.read()
                except (MemoryError, SystemError, cv2.error) as exc:
                    ok, fr = False, None
                    if self._current_source_kind == "RTSP" and self._current_input_url:
                        LOGGER.exception("ExcepciÃ³n en cap.read(); reiniciando captura RTSP (%s)", self._current_input_url)
                    else:
                        LOGGER.exception("ExcepciÃ³n en cap.read(); deteniendo captura.")

                if (not ok) or (fr is None):
                    if self._current_source_kind == "RTSP" and self._current_input_url:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        if self.cap is cap:
                            self.cap = None
                        cap = None
                        if not recon_notice:
                            LOGGER.warning("Lectura RTSP fallida; reintentando (%s)", self._current_input_url)
                            try:
                                self._set_status("Sin senal RTSP. Reintentando...")
                            except Exception:
                                pass
                            recon_notice = True
                            self._set_rtsp_state("reintentando", "lectura fallida")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, 8.0)
                        continue
                    break

                reconnect_delay = 1.0
                recon_notice = False
                now_mono = time.monotonic()
                self._rtsp_last_frame_ts = now_mono
                self._rtsp_last_frame_wall = time.time()
                if self._current_source_kind == "RTSP" and self._current_input_url:
                    self._set_rtsp_state("transmitiendo", "frames recibidos")
                
                # PerfTrace: Reader
                if self.perf_trace_enabled.get():
                    t_read = (time.perf_counter() - t_read_start) * 1000.0
                    gap = (t_read_start - self._last_read_success_ts) * 1000.0 if self._last_read_success_ts > 0 else 0.0
                    self._last_read_success_ts = t_read_start
                    q_sz = q.qsize()
                    self._log_perftrace_event("reader_perf", {
                        "read_ms": round(t_read, 2),
                        "gap_ms": round(gap, 2),
                        "q_size": q_sz
                    })
                if self._current_source_kind == "RTSP":
                    try:
                        if q.full():
                            try:
                                _ = q.get_nowait()
                            except Exception:
                                pass
                        q.put_nowait(fr)
                    except Exception:
                        pass
                else:
                    try:
                        q.put(fr, timeout=0.1)
                    except Exception:
                        # Cola llena: descarta el mÃ¡s antiguo para mantener baja latencia
                        try:
                            _ = q.get_nowait()
                        except Exception:
                            pass
                        try:
                            q.put(fr, timeout=0.05)
                        except Exception:
                            pass
                if (self._out_spec is None or self._out_spec[0] <= 0 or self._out_spec[1] <= 0) and fr is not None:
                    try:
                        h, w = fr.shape[:2]
                        fps_val = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0.0
                        self._configure_stream_outputs(w, h, fps_val if fps_val and fps_val > 1e-3 else self._out_spec[2] if self._out_spec else 25.0)
                    except Exception:
                        pass
        finally:
            self._set_rtsp_state("detenido", "lector detenido")
            try:
                q.put(None, timeout=0.05)
            except Exception:
                pass
            self._revert_mmcss_for_thread()

    def _draw_writer_loop(self):
        self._boost_current_thread_priority()
        while self.running and not self.draw_stop:
            try:
                item = self.draw_queue.get(timeout=0.2) if self.draw_queue is not None else None
            except queue.Empty:
                continue
            if item is None:
                break

            frame, result1, result2 = item
            annotated = frame

            rt = self._runtime_cfg
            draw_start = time.perf_counter()
            t0 = time.perf_counter()
            
            self._draw_stride_counter += 1
            
            # Con V4.6 (Caching), podemos dibujar SIEMPRE.
            # El sectorizador internamente ya optimiza usando el overlay cacheado.
            do_draw = True
            
            t_draw_res = 0.0
            t_sector = 0.0
            t_write = 0.0
            t_resize = 0.0
            
            if do_draw:
                try:
                    # Dibujar resultados de detecciÃ³n
                    if result1 is not None:
                        self._draw_result_on(
                            annotated,
                            result1,
                            color=None,
                            name_prefix="M1",
                        )
                        try:
                            if bool(rt.get("highlight_tiny", True)):
                                annotated = self._highlight_small_stains(result1, annotated)
                        except Exception:
                            pass

                    if result2 is not None:
                        self._draw_result_on(
                            annotated,
                            result2,
                            color=None,
                            name_prefix="M2",
                        )
                        try:
                            if bool(rt.get("highlight_tiny", True)):
                                annotated = self._highlight_small_stains(result2, annotated)
                        except Exception:
                            pass
                    
                    t1 = time.perf_counter()
                    t_draw_res = (t1 - t0) * 1000.0
                    self._last_draw_ms = float(t_draw_res)

                    # Aplicar sectorizaciÃ³n (V4.7: Optimizado con stride para extracciÃ³n de detecciones)
                    sector_processed = False
                    if self.sectorizador is not None and bool(rt.get("sector_mostrar", True)):
                        try:
                            skip_sector = False
                            if getattr(self, "_sector_skip_frames", 0) > 0:
                                self._sector_skip_frames = max(0, int(self._sector_skip_frames) - 1)
                                skip_sector = True

                            if skip_sector:
                                try:
                                    if (
                                        hasattr(self.sectorizador, "_cached_overlay")
                                        and getattr(self.sectorizador, "_cached_overlay", None) is not None
                                        and not getattr(self.sectorizador, "_needs_redraw", True)
                                    ):
                                        self.sectorizador._aplicar_overlay_cached(annotated)
                                    else:
                                        skip_sector = False
                                except Exception:
                                    skip_sector = False

                            if not skip_sector:
                                # V4.7: Solo extraer detecciones cuando el sectorizador las necesita (cada N frames)
                                # El sectorizador hace fast path si no necesita recalcular
                                if not hasattr(self, '_sector_det_counter'):
                                    self._sector_det_counter = 0
                                self._sector_det_counter += 1
                                
                                # Sincronizar stride con el sectorizador (stride=10)
                                should_extract = (self._sector_det_counter % 10 == 1) or self._sector_det_counter <= 1
                                
                                if should_extract:
                                    detecciones = self._extraer_detecciones_para_sectorizador(result1, result2)
                                    # Guardar solo si hay detecciones nuevas (evita vaciar en rafagas)
                                    if detecciones:
                                        self._last_sector_detecciones = detecciones
                                    else:
                                        detecciones = getattr(self, "_last_sector_detecciones", detecciones)
                                else:
                                    # Usar detecciones cacheadas (o lista vacÃ­a si no hay)
                                    detecciones = getattr(self, '_last_sector_detecciones', [])
    
                                annotated = self.sectorizador.procesar_frame(
                                    detecciones,
                                    annotated,
                                    dibujar=True,
                                )
                                sector_processed = True
                        except Exception as sector_err:
                            LOGGER.debug(f"[Sectorizador] Error: {sector_err}")
                    
                    t2 = time.perf_counter()
                    t_sector = (t2 - t1) * 1000.0
                    if sector_processed:
                        self._last_sector_ms = float(t_sector)
                        if self._last_sector_ms > 45.0:
                            self._sector_skip_frames = min(self._sector_skip_frames + 2, 12)
                        elif self._last_sector_ms < 10.0:
                            # Muy rÃ¡pido = reset completo para recuperaciÃ³n inmediata
                            self._sector_skip_frames = 0
                        elif self._last_sector_ms < 20.0 and self._sector_skip_frames > 0:
                            # RecuperaciÃ³n mÃ¡s agresiva: -2 en lugar de -1
                            self._sector_skip_frames = max(0, self._sector_skip_frames - 2)

                except Exception as draw_err:
                    print(f"[Draw] Error: {draw_err}", flush=True)

            t3 = time.perf_counter()
            if self.writer is not None:
                try:
                    self.writer.write(annotated)
                except Exception:
                    pass
            t4 = time.perf_counter()
            t_write = (t4 - t3) * 1000.0

            frame_for_display = annotated
            preview_frame = None
            if self._ui_visible and frame_for_display is not None:
                preview_frame = frame_for_display
                try:
                    frame_h, frame_w = frame_for_display.shape[:2]
                    if frame_w > self._ui_preview_max_w or frame_h > self._ui_preview_max_h:
                        scale = min(self._ui_preview_max_w / max(frame_w, 1), self._ui_preview_max_h / max(frame_h, 1))
                        if scale < 1.0:
                            new_w = max(1, int(frame_w * scale))
                            new_h = max(1, int(frame_h * scale))
                            preview_frame = cv2.resize(frame_for_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    preview_frame = frame_for_display

            t5 = time.perf_counter()
            t_resize = (t5 - t4) * 1000.0
            
            # Print de debug si es lento (> 50ms)
            total_ms = (t5 - t0) * 1000.0
            if total_ms > 50:
                now_dbg = time.perf_counter()
                if (now_dbg - self._last_perf_debug_ts) >= 1.0:
                    self._last_perf_debug_ts = now_dbg
                    print(
                        f"[PerfDebug] Total={total_ms:.1f}ms | Draw={t_draw_res:.1f}ms | Sec={t_sector:.1f}ms | WRITER={t_write:.1f}ms | Resize={t_resize:.1f}ms",
                        flush=True,
                    )

            # PerfTrace: Log draw_frame event
            if self.perf_trace_enabled.get():
                det_count = 0
                try:
                    if result1 is not None and hasattr(result1, 'boxes') and result1.boxes is not None:
                        det_count += len(result1.boxes)
                    if result2 is not None and hasattr(result2, 'boxes') and result2.boxes is not None:
                        det_count += len(result2.boxes)
                except Exception:
                    pass
                q_len = self.draw_queue.qsize() if self.draw_queue is not None else 0
                self._log_perftrace_event("draw_frame", {
                    "total_ms": round(total_ms, 2),
                    "draw_ms": round(t_draw_res, 2),
                    "sec_ms": round(t_sector, 2),
                    "resize_ms": round(t_resize, 2),
                    "writer_ms": round(t_write, 2),
                    "det_count": det_count,
                    "q_len": q_len,
                })
            
            # PerfTrace V2: Marcar fin de ciclo
            self._last_cycle_end_ts = time.perf_counter()

            if bool(rt.get("rtsp_out_enable", False)):
                try:
                    if self._rtsp_q.full():
                        self._rtsp_q.get_nowait()
                    self._rtsp_q.put_nowait(frame_for_display)
                except Exception:
                    pass

            with self.frame_lock:
                self.last_frame_bgr = frame_for_display
                if self._ui_visible and preview_frame is not None:
                    self.last_frame_preview = preview_frame
                    self._preview_seq += 1

            with self._draw_perf_lock:
                self._draw_perf_total += time.perf_counter() - draw_start
                self._draw_perf_calls += 1

        # Vaciar posibles elementos restantes y asegurar que el hilo termina limpio
        if self.draw_queue is not None:
            while True:
                try:
                    item = self.draw_queue.get_nowait()
                    if item is None:
                        break
                except Exception:
                    break
        self._revert_mmcss_for_thread()

    def _set_model_precision(self, net, use_half: bool):
        if net is None:
            return None
        try:
            if use_half:
                net = net.half()
            else:
                net = net.float()
            try:
                net = net.to(memory_format=torch.channels_last)
            except Exception:
                pass
            net._precision = torch.float16 if use_half else torch.float32
        except Exception:
            pass
        return net

    def _launch_infer(
        self,
        net,
        names,
        model_tag: str | None,
        frame_bgr: np.ndarray,
        imgsz: int,
        stream: Optional[torch.cuda.Stream],
        conf: float,
        iou: float,
        classes_ids,
        agnostic: bool,
        max_det: int,
    ) -> _InferLaunch | None:
        if net is None:
            return None
        rt = self._runtime_cfg
        use_half = bool(rt.get("use_half", True))
        dtype = torch.float16 if use_half else torch.float32

        t_infer_start = time.perf_counter()
        t_preprocess = 0.0
        t_forward = 0.0
        t_nms = 0.0
        t_mask = 0.0
        n_dets_raw = 0
        mask_skipped = False
        mask_skip_reason = None

        h0, w0 = frame_bgr.shape[:2]
        t0_pre = time.perf_counter()
        x, ratio, (dw, dh), cpu_buf, gpu_buf, pre_event = _preprocess_to_cuda(
            frame_bgr,
            imgsz,
            stride=32,
            stream_preproc=self.stream_preproc,
            dtype=dtype,
        )
        if pre_event is not None:
            if stream is not None:
                stream.wait_event(pre_event)
            _PREPROCESS_CACHE.release_event(pre_event)
        t_preprocess = (time.perf_counter() - t0_pre) * 1000.0

        stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        masks_tensor = None
        dets = None

        with torch.inference_mode():
            with stream_ctx:
                t0_fwd = time.perf_counter()
                raw_out = net(x)
                # Sync to measure forward time accurately (only when tracing)
                if self.perf_trace_enabled.get():
                    torch.cuda.synchronize()
                t_forward = (time.perf_counter() - t0_fwd) * 1000.0

                proto = None
                preds = raw_out
                if isinstance(raw_out, (list, tuple)):
                    if (
                        len(raw_out) >= 1
                        and isinstance(raw_out[0], (list, tuple))
                        and len(raw_out[0]) >= 2
                    ):
                        preds = raw_out[0][0]
                        proto = raw_out[0][1]
                    else:
                        preds = raw_out[0]
                        if len(raw_out) > 1:
                            proto_candidate = raw_out[1]
                            if isinstance(proto_candidate, (list, tuple)):
                                proto_candidate = proto_candidate[-1]
                            proto = proto_candidate
                if proto is not None and not torch.is_tensor(proto):
                    proto = None

                nc = len(names) if isinstance(names, dict) else 0
                t0_nms = time.perf_counter()
                dets_list = non_max_suppression(
                    preds,
                    conf_thres=conf,
                    iou_thres=iou,
                    classes=classes_ids,
                    agnostic=agnostic,
                    max_det=max_det,
                    nc=nc,
                )
                dets = dets_list[0] if dets_list else None
                t_nms = (time.perf_counter() - t0_nms) * 1000.0
                n_dets_raw = len(dets) if dets is not None else 0

                if dets is not None and len(dets) > 0:
                    border_mask_required = False
                    if (
                        self.sectorizador is not None
                        and bool(self.sectorizador.config_sectores.use_border_masks)
                        and isinstance(names, dict)
                    ):
                        cb = self.sectorizador.config_bordes
                        border_names = {
                            cb.clase_superior,
                            cb.clase_inferior,
                            cb.clase_izquierdo,
                            cb.clase_derecho,
                        }
                        try:
                            name_to_id = {str(v): int(k) for k, v in names.items()}
                            border_ids = [name_to_id.get(cls) for cls in border_names if cls]
                        except Exception:
                            border_ids = []
                        if border_ids:
                            try:
                                cls_ids = dets[:, 5].to(torch.int64)
                                for border_id in border_ids:
                                    if border_id is None:
                                        continue
                                    if bool((cls_ids == border_id).any()):
                                        border_mask_required = True
                                        break
                            except Exception:
                                border_mask_required = True

                    show_masks = bool(rt.get("show_masks", True))
                    need_masks = (
                        show_masks
                        or self._area_any_enabled_for_model(model_tag)
                        or border_mask_required
                    )
                    mask_cap = 80
                    if len(dets) > mask_cap and not border_mask_required:
                        need_masks = False
                        mask_skipped = True
                        mask_skip_reason = "overflow"

                    if need_masks:
                        t0_mask = time.perf_counter()
                        retina_masks = bool(rt.get("use_retina_masks", False))
                        if proto is not None and dets.shape[1] > 6:
                            proto_img = proto
                            if isinstance(proto_img, (list, tuple)):
                                proto_img = proto_img[-1]
                            if proto_img.dim() == 4:
                                proto_img = proto_img[0]
                            mask_coeffs = dets[:, 6:].clone()
                            mask_dim = proto_img.shape[0] if proto_img is not None else mask_coeffs.shape[1]
                            if mask_coeffs.shape[1] > mask_dim:
                                mask_coeffs = mask_coeffs[:, :mask_dim]
                            boxes_in = dets[:, :4].clone()
                            if retina_masks:
                                scaled_boxes = scale_boxes((x.shape[2], x.shape[3]), boxes_in, (h0, w0))
                                masks_tensor = process_mask_native(proto_img, mask_coeffs, scaled_boxes, (h0, w0))
                                dets[:, :4] = scaled_boxes
                            else:
                                masks_tensor = process_mask(proto_img, mask_coeffs, boxes_in, (x.shape[2], x.shape[3]), upsample=True)
                                masks_tensor = scale_masks(masks_tensor.unsqueeze(0), (h0, w0))[0]
                                dets[:, :4] = scale_boxes((x.shape[2], x.shape[3]), boxes_in, (h0, w0))
                            masks_tensor = masks_tensor.gt_(0.0)
                            t_mask = (time.perf_counter() - t0_mask) * 1000.0
                        else:
                            dets[:, :4] = scale_boxes((x.shape[2], x.shape[3]), dets[:, :4], (h0, w0))
                    else:
                        dets[:, :4] = scale_boxes((x.shape[2], x.shape[3]), dets[:, :4], (h0, w0))

                    dets[:, :4].round_()

        record_stream = stream if stream is not None else torch.cuda.current_stream()
        done_event = torch.cuda.Event()
        done_event.record(record_stream)

        # PerfTrace: Log infer_perf event
        if self.perf_trace_enabled.get():
            t_total = (time.perf_counter() - t_infer_start) * 1000.0
            self._log_perftrace_event("infer_perf", {
                "total_ms": round(t_total, 2),
                "preproc_ms": round(t_preprocess, 2),
                "forward_ms": round(t_forward, 2),
                "nms_ms": round(t_nms, 2),
                "mask_ms": round(t_mask, 2),
                "mask_skipped": mask_skipped,
                "mask_skip_reason": mask_skip_reason,
                "n_dets": n_dets_raw,
                "imgsz": imgsz,
            })

        return _InferLaunch(
            done_event=done_event,
            dets=dets,
            masks=masks_tensor,
            names=names,
            frame_shape=(h0, w0),
            cpu_buf=cpu_buf,
            gpu_buf=gpu_buf,
            pre_event=pre_event,
        )


    def _pack_to_cpu(self, job: _InferLaunch | None, model_tag: str | None = None) -> _ResultAdapter | None:
        if job is None:
            return None

        t_pack_start = time.perf_counter()
        t_sync = 0.0
        t_det_cpu = 0.0
        t_mask_cpu = 0.0
        t_area = 0.0
        t_contours = 0.0
        n_dets = 0
        mask_shape = []
        sum_bbox_pixels = 0
        need_area_px = self._area_any_enabled_for_model(model_tag)
        need_contours = bool(self._runtime_cfg.get("show_masks", True))

        try:
            if job.done_event is not None:
                job.done_event.synchronize()
        except Exception:
            torch.cuda.synchronize()
        t_sync = (time.perf_counter() - t_pack_start) * 1000.0

        dets = job.dets
        masks_tensor = job.masks
        names = job.names

        if dets is None or dets.numel() == 0:
            _PREPROCESS_CACHE.release_cpu(job.cpu_buf)
            _PREPROCESS_CACHE.release_gpu(job.gpu_buf)
            empty_xyxy = np.empty((0, 4), dtype=np.float32)
            empty_conf = np.empty((0,), dtype=np.float32)
            empty_cls = np.empty((0,), dtype=np.int64)
            empty_boxes = _BoxesAdapter(xyxy=empty_xyxy, conf=empty_conf, cls=empty_cls)
            return _ResultAdapter(empty_boxes, names, masks=None)

        t0_det = time.perf_counter()
        dets_cpu = dets[:, :6].detach().to(dtype=torch.float32).cpu().contiguous()
        xyxy_np = dets_cpu[:, :4].contiguous().numpy()
        conf_np = dets_cpu[:, 4].contiguous().numpy()
        cls_np = dets_cpu[:, 5].to(torch.int64).contiguous().numpy()
        t_det_cpu = (time.perf_counter() - t0_det) * 1000.0
        n_dets = len(xyxy_np)

        boxes = _BoxesAdapter(
            xyxy=xyxy_np,
            conf=conf_np,
            cls=cls_np,
        )

        masks_adapter: _MasksAdapter | None = None
        if masks_tensor is not None:
            t0_mask = time.perf_counter()
            masks_cpu = masks_tensor.detach().to(torch.uint8).cpu().contiguous()
            masks_np = masks_cpu.numpy()
            t_mask_cpu = (time.perf_counter() - t0_mask) * 1000.0
            mask_shape = list(masks_np.shape) if masks_np is not None else []
            
            masks_area_px: np.ndarray | None = None
            bbox_crops = None
            if need_area_px or need_contours:
                t0_area = time.perf_counter()
                try:
                    if masks_np is not None and masks_np.ndim == 3:
                        h_m, w_m = masks_np.shape[1], masks_np.shape[2]
                        bbox_crops = []
                        if need_area_px:
                            masks_area_px = np.zeros((masks_np.shape[0],), dtype=np.float32)
                        pad = 2
                        for i in range(masks_np.shape[0]):
                            x1p = 0
                            y1p = 0
                            x2p = w_m
                            y2p = h_m
                            if xyxy_np is not None and i < len(xyxy_np):
                                try:
                                    x1f, y1f, x2f, y2f = [float(v) for v in xyxy_np[i]]
                                    if math.isfinite(x1f) and math.isfinite(y1f) and math.isfinite(x2f) and math.isfinite(y2f):
                                        x1p = max(0, int(math.floor(x1f)) - pad)
                                        y1p = max(0, int(math.floor(y1f)) - pad)
                                        x2p = min(w_m, int(math.ceil(x2f)) + pad)
                                        y2p = min(h_m, int(math.ceil(y2f)) + pad)
                                        if x2p <= x1p or y2p <= y1p:
                                            x1p, y1p, x2p, y2p = 0, 0, w_m, h_m
                                except Exception:
                                    x1p, y1p, x2p, y2p = 0, 0, w_m, h_m
                            bbox_crops.append((x1p, y1p, x2p, y2p))
                            if need_area_px:
                                sum_bbox_pixels += max(0, x2p - x1p) * max(0, y2p - y1p)
                                m_crop = masks_np[i][y1p:y2p, x1p:x2p]
                                if m_crop.size == 0:
                                    masks_area_px[i] = 0.0
                                else:
                                    try:
                                        masks_area_px[i] = float(np.count_nonzero(m_crop))
                                    except Exception:
                                        masks_area_px[i] = float(m_crop.sum())
                except Exception:
                    masks_area_px = None
                    bbox_crops = None
                    sum_bbox_pixels = 0
                t_area = (time.perf_counter() - t0_area) * 1000.0
            
            # Pre-calcular contornos para evitar hacerlo en cada frame de dibujo (gran cuello de botella)
            # Esto se hace una vez por inferencia en lugar de N veces por display
            masks_xy = None
            if need_contours:
                t0_contours = time.perf_counter()
                try:
                    # Asumimos que masks_np es (N, H, W)
                    if masks_np is not None and masks_np.ndim == 3 and masks_np.shape[0] <= 25:
                        masks_xy = []
                        for i in range(masks_np.shape[0]):
                            m = masks_np[i]
                            if bbox_crops is not None and i < len(bbox_crops):
                                x1p, y1p, x2p, y2p = bbox_crops[i]
                                m_crop = m[y1p:y2p, x1p:x2p]
                            else:
                                x1p, y1p = 0, 0
                                m_crop = m
                            if m_crop.size == 0:
                                masks_xy.append(None)
                                continue
                            mask_uint8 = m_crop if m_crop.dtype == np.uint8 else m_crop.astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                # Tomar el contorno mas grande
                                c = max(contours, key=cv2.contourArea)
                                if x1p or y1p:
                                    try:
                                        c = c + np.array([[[x1p, y1p]]], dtype=c.dtype)
                                    except Exception:
                                        pass
                                masks_xy.append(c)
                            else:
                                masks_xy.append(None)
                except Exception:
                    # Si falla, se dejara como None y el dibujo hara fallback (lento)
                    masks_xy = None
                t_contours = (time.perf_counter() - t0_contours) * 1000.0

            masks_adapter = _MasksAdapter(data_np=masks_np, xy=masks_xy, area_px=masks_area_px)

        _PREPROCESS_CACHE.release_cpu(job.cpu_buf)
        _PREPROCESS_CACHE.release_gpu(job.gpu_buf)

        # PerfTrace: Log pack_perf event
        if self.perf_trace_enabled.get():
            t_total = (time.perf_counter() - t_pack_start) * 1000.0
            self._log_perftrace_event("pack_perf", {
                "total_ms": round(t_total, 2),
                "sync_ms": round(t_sync, 2),
                "det_cpu_ms": round(t_det_cpu, 2),
                "mask_cpu_ms": round(t_mask_cpu, 2),
                "area_ms": round(t_area, 2),
                "contours_ms": round(t_contours, 2),
                "n_dets": n_dets,
                "mask_shape": mask_shape,
                "sum_bbox_pixels": int(sum_bbox_pixels),
            })
            
            # PerfTrace V2: Cycle Gap (tiempo perdido entre frames)
            if self._last_cycle_end_ts > 0:
                cycle_gap = (t_pack_start - self._last_cycle_end_ts) * 1000.0
                # Si el gap es muy grande (>2000ms), es probablemente un pause/resume, no lo logueamos como gap
                if cycle_gap < 2000:
                   self._log_perftrace_event("cycle_perf", {
                       "gap_ms": round(cycle_gap, 2)
                   })

        return _ResultAdapter(boxes, names, masks=masks_adapter)

    def _highlight_small_stains(self, result, img: np.ndarray) -> np.ndarray:
        """Dibuja un recuadro grueso y un punto rojo en el centro de detecciones
        de clase 'Mancha' que sean muy pequeÃ±as, para que no pasen desapercibidas.
        No altera el filtrado del modelo, solo la visualizaciÃ³n.
        """
        names = {}
        try:
            if hasattr(result, "names") and isinstance(result.names, dict):
                names = result.names
            elif hasattr(self.model, "names"):
                names = getattr(self.model, "names", {})
        except Exception:
            names = {}

        mancha_cid = None
        try:
            for cid, name in names.items():
                if isinstance(name, str) and name.lower().startswith("mancha"):
                    mancha_cid = int(cid)
                    break
        except Exception:
            pass

        if mancha_cid is None:
            return img

        try:
            h, w = img.shape[:2]
            img_area = float(w * h)
            tiny_thr = 0.0005 * img_area

            boxes_xyxy = result.xyxy_np()
            boxes_cls = result.cls_np()
            
            if boxes_xyxy is None or len(boxes_xyxy) == 0:
                return img
                
            # VectorizaciÃ³n: calcular Ã¡reas y filtrar
            widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
            areas = widths * heights
            
            # MÃ¡scara booleana: es clase 'Mancha' Y es pequeÃ±a
            is_mancha = (boxes_cls.astype(int) == int(mancha_cid))
            is_tiny = (areas <= tiny_thr)
            mask = is_mancha & is_tiny
            
            if not np.any(mask):
                return img
            
            # Obtener Ã­ndices que cumplen condiciÃ³n
            idxs = np.where(mask)[0]
            
            thickness = 3 if max(w, h) < 1400 else 4
            color = (255, 0, 255)
            red = (0, 0, 255)
            
            boxes_conf = result.conf_np()
            
            for i in idxs:
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int) # type: ignore
                # Clampear coordenadas
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 5, red, -1)
                
                if boxes_conf is not None:
                    rt = getattr(self, "_runtime_cfg", {}) or {}
                    if bool(rt.get("show_names", True)):
                        show_confidence = bool(rt.get("show_confidence", True))
                        cf = float(boxes_conf[i])
                        label = f"Mancha {cf:.2f}" if show_confidence else "Mancha"
                        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1, cv2.LINE_AA)
        except Exception:
            return img
        return img

    def _draw_result_on(
        self,
        img: np.ndarray,
        result,
        color=None,
        name_prefix: str = "M1",
        *,
        force_show_names: bool | None = None,
        force_show_masks: bool | None = None,
        force_show_area: bool | None = None,
        force_topk: int | None = None,
    ) -> None:
        """Dibuja sobre `img` las detecciones de `result`.
        Si `color` es None, usa el color configurado por clase en `self.class_cfg`.
        Respeta las opciones `show_boxes`/`show_names` y las casillas M1/M2 de cada clase.
        """
        rt = self._runtime_cfg
        show_boxes = bool(rt.get("show_boxes", True)) if hasattr(self, "show_boxes") else True
        show_names = (
            force_show_names if force_show_names is not None
            else (bool(rt.get("show_names", True)) if hasattr(self, "show_names") else True)
        )
        show_confidence = bool(rt.get("show_confidence", True)) if hasattr(self, "show_confidence") else True
        show_masks = (
            force_show_masks if force_show_masks is not None
            else (bool(rt.get("show_masks", True)) if hasattr(self, "show_masks") else True)
        )
        masks_as_contours = bool(rt.get("masks_as_contours", False))
        line_thickness = int(rt.get("line_thickness", 2))
        font_scale = float(rt.get("font_scale", 0.5))
        area_text_scale = float(rt.get("area_text_scale", 0.4))
        topk_draw = int(rt.get("topk_draw", 0))
        use_m1 = (name_prefix == "M1")

        # Lee predicciones (ya en NumPy)
        xyxy = result.xyxy_np()
        confs = result.conf_np()
        clss = result.cls_np()
        xyxy_i = None
        if xyxy is not None:
            try:
                xyxy_i = xyxy.astype(np.int32, copy=False)
            except Exception:
                xyxy_i = None
        names = {}
        try:
            if hasattr(result, "names") and isinstance(result.names, dict):
                names = result.names
            elif hasattr(self.model, "names"):
                names = getattr(self.model, "names", {})
        except Exception:
            pass

        class_cache = getattr(self, "_class_cache", {})
        cache_entry = class_cache.get("M1" if use_m1 else "M2", {})
        enabled_by_cid = cache_entry.get("enabled")
        thr_by_cid = cache_entry.get("thr")
        color_by_cid = cache_entry.get("color")
        area_enabled_by_cid = cache_entry.get("area_mode")

        mask_data_npy = result.masks_np()
        masks_xy = result.masks_xy()
        masks_area_px = result.masks_area_px() if hasattr(result, "masks_area_px") else None

        global_area_mode = getattr(self, "_area_label_mode_flag", AREA_MODE_CM2)
        area_any_enabled = self._area_any_enabled_for_model(name_prefix)
        show_area = global_area_mode != AREA_MODE_OFF and area_any_enabled
        if force_show_area is False:
            show_area = False
        need_area_any = show_area
        need_area_cm2 = (
            show_area
            and global_area_mode in {AREA_MODE_CM2, AREA_MODE_BOTH}
            and self._calibration_valid
            and self.calibration is not None
        )

        # SelecciÃ³n previa de Ã­ndices a dibujar (por clase habilitada + Top-K por conf)
        sel_idx = None
        if xyxy is not None and len(xyxy) > 0:
            idx_all = np.arange(len(xyxy), dtype=int)
            cls_idx = clss.astype(int, copy=False)
            enabled_mask = np.ones_like(idx_all, dtype=bool)
            if enabled_by_cid is not None and enabled_by_cid.size > 0:
                within = cls_idx < enabled_by_cid.size
                enabled_mask &= within & enabled_by_cid[cls_idx.clip(max=enabled_by_cid.size - 1)]
            if confs is not None:
                thr_values = np.full_like(cls_idx, float(self.conf), dtype=np.float32)
                if thr_by_cid is not None and thr_by_cid.size > 0:
                    thr_values = thr_by_cid[np.clip(cls_idx, 0, thr_by_cid.size - 1)]
                conf_mask = confs >= thr_values
                enabled_mask &= conf_mask
            enabled_idx = idx_all[enabled_mask]
            
            # OPTIMIZADO: Filtrar por sectores excluidos y restricciones de clase
            # Solo ejecutar si hay restricciones activas o sectores excluidos
            if self.sectorizador is not None and len(enabled_idx) > 0:
                # Asegurar cachÃ© actualizada
                if not hasattr(self, "_cached_excluded_sectors_0based"):
                    self._update_sector_filter_cache()
                
                # Chequeo rÃ¡pido de si necesitamos filtrar algo
                has_exclusions = len(self._cached_excluded_sectors_0based) > 0
                has_restrictions = self.sectorizador.config_restricciones.enabled
                
                if has_exclusions or has_restrictions:
                    allowed_mask = np.ones(len(enabled_idx), dtype=bool)

                    for i, k in enumerate(enabled_idx):
                        cid = int(clss[k])
                        class_name = names.get(cid, str(cid))
                        box = xyxy[k].tolist()
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        sector_id = self.sectorizador.obtener_sector_para_punto(cx, cy)
                        
                        det = {
                            "class_name": class_name,
                            "cls": class_name,
                            "bbox": box,
                            "bbox_xyxy": box,
                            "sector": sector_id
                        }
                        
                        if not self._is_detection_allowed(det):
                            allowed_mask[i] = False
                    
                    enabled_idx = enabled_idx[allowed_mask]
            
            try:
                kmax = int(force_topk) if force_topk is not None else topk_draw
            except Exception:
                kmax = 0
            if kmax and confs is not None and enabled_idx.size > kmax:
                sub_scores = confs[enabled_idx]
                order = np.argpartition(-sub_scores, kmax - 1)[:kmax]
                sel_idx = enabled_idx[order]
            else:
                sel_idx = enabled_idx

        # Dibujo de mÃ¡scaras (por detecciÃ³n, con color por clase)
        try:
            if show_masks and (mask_data_npy is not None or masks_xy is not None):
                # Si se solicita dibujo por contornos, intentamos usar polÃ­gonos nativos
                if masks_xy is not None and masks_as_contours:
                    polys = masks_xy
                    try:
                        N = len(polys)
                    except Exception:
                        N = 0
                    indices = sel_idx if sel_idx is not None else list(range(N))
                    
                    # Calcular epsilon una vez fuera del bucle
                    eps_factor = 0.002 * (img.shape[0] + img.shape[1])
                    eps = max(1.5, eps_factor)
                    line_thick = line_thickness
                    
                    for k in indices:
                        if k >= N:
                            continue
                        cid = int(clss[k]) if clss is not None and k < len(clss) else None
                        if cid is None:
                            continue
                        if enabled_by_cid is not None and cid < enabled_by_cid.size and not enabled_by_cid[cid]:
                            continue
                        
                        poly = polys[k]
                        if poly is None:
                            continue

                        base_color = color_by_cid[cid] if (color_by_cid is not None and cid < color_by_cid.shape[0]) else np.array((0, 255, 255), dtype=np.uint8)
                        bgr = tuple(color if color is not None else base_color.tolist())
                        
                        try:
                            # Si es un contorno de OpenCV (N, 1, 2) o similar
                            pts = poly
                            if isinstance(pts, np.ndarray):
                                # Asegurar formato float32
                                if pts.dtype != np.float32:
                                    pts = pts.astype(np.float32)
                                
                                if pts.ndim == 3 and pts.shape[1] == 1:
                                    # Contorno standard de findContours (N, 1, 2)
                                    approx = cv2.approxPolyDP(pts, epsilon=eps, closed=True)
                                    cv2.polylines(img, [approx.astype(np.int32)], True, bgr, line_thick)
                                elif pts.ndim == 2:
                                    # (N, 2)
                                    approx = cv2.approxPolyDP(pts, epsilon=eps, closed=True)
                                    cv2.polylines(img, [approx.astype(np.int32)], True, bgr, line_thick)
                            else:
                                # Fallback para listas
                                pts = np.asarray(poly, dtype=np.float32)
                                approx = cv2.approxPolyDP(pts, epsilon=eps, closed=True)
                                cv2.polylines(img, [approx.astype(np.int32)], True, bgr, line_thick)
                        except Exception:
                            continue
                else:
                    filled_with_polys = False
                    if masks_xy is not None:
                        polys = masks_xy
                        try:
                            N = len(polys)
                        except Exception:
                            N = 0
                        if N > 0:
                            indices = sel_idx if sel_idx is not None else list(range(N))
                            if self._overlay_buf is None or self._overlay_buf.shape != img.shape:
                                self._overlay_buf = np.zeros_like(img, dtype=np.uint8)
                            else:
                                self._overlay_buf[...] = 0
                            overlay = self._overlay_buf

                            eps_factor = 0.002 * (img.shape[0] + img.shape[1])
                            eps = max(1.5, eps_factor)
                            any_poly = False
                            for k in indices:
                                if k >= N:
                                    continue
                                cid = int(clss[k]) if clss is not None and k < len(clss) else None
                                if cid is None:
                                    continue
                                if enabled_by_cid is not None and cid < enabled_by_cid.size and not enabled_by_cid[cid]:
                                    continue

                                poly = polys[k]
                                if poly is None:
                                    continue

                                base_color = color_by_cid[cid] if (color_by_cid is not None and cid < color_by_cid.shape[0]) else np.array((0, 255, 255), dtype=np.uint8)
                                bgr = tuple(color if color is not None else base_color.tolist())

                                try:
                                    pts = np.asarray(poly, dtype=np.float32)
                                    if pts.ndim == 3 and pts.shape[1] == 1:
                                        curve = pts
                                    elif pts.ndim == 2:
                                        curve = pts.reshape(-1, 1, 2)
                                    else:
                                        continue
                                    if curve.shape[0] < 3:
                                        continue
                                    approx = cv2.approxPolyDP(curve, epsilon=eps, closed=True)
                                    cv2.fillPoly(overlay, [approx.astype(np.int32)], bgr)
                                    any_poly = True
                                except Exception:
                                    continue

                            if any_poly:
                                cv2.addWeighted(overlay, 0.30, img, 1.0, 0.0, dst=img)
                                filled_with_polys = True

                    if (not filled_with_polys) and mask_data_npy is not None and mask_data_npy.ndim == 3:
                        N = mask_data_npy.shape[0]
                        indices = sel_idx if sel_idx is not None else list(range(N))
                        
                        # NOTA: El clipping global de mÃ¡scaras fue desactivado porque eliminaba
                        # clases "solo_fuera_malla" que por definiciÃ³n estÃ¡n fuera del ROI.
                        # Para un clipping correcto, se necesitarÃ­a lÃ³gica por clase.
                        roi_mask_clipping = None  # Desactivado temporalmente
                        
                        if masks_as_contours:
                            # Dibujar contornos a partir de raster
                            for k in indices:
                                if k >= N:
                                    continue
                                cid = int(clss[k]) if clss is not None and k < len(clss) else None
                                if cid is None:
                                    continue
                                if enabled_by_cid is not None and cid < enabled_by_cid.size and not enabled_by_cid[cid]:
                                    continue
                                base_color = color_by_cid[cid] if (color_by_cid is not None and cid < color_by_cid.shape[0]) else np.array((0, 255, 255), dtype=np.uint8)
                                bgr = tuple(color if color is not None else base_color.tolist())
                                m = mask_data_npy[k]
                                
                                # CLIPPING
                                if roi_mask_clipping is not None and m.shape == roi_mask_clipping.shape:
                                    m = m & roi_mask_clipping
                                
                                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if not contours:
                                    continue
                                cv2.polylines(img, contours, isClosed=True, color=bgr, thickness=line_thickness)
                        else:
                            # Overlay coloreado
                            if self._overlay_buf is None or self._overlay_buf.shape != img.shape:
                                self._overlay_buf = np.zeros_like(img, dtype=np.uint8)
                            else:
                                self._overlay_buf[...] = 0
                            overlay = self._overlay_buf
                            fast_union_done = False
                            if clss is not None:
                                try:
                                    idx_arr = np.asarray(indices, dtype=int)
                                    if idx_arr.size >= 6:
                                        cls_sel = clss[idx_arr].astype(int, copy=False)
                                        if cls_sel.size and np.all(cls_sel == cls_sel[0]):
                                            cid = int(cls_sel[0])
                                            if enabled_by_cid is None or (cid < enabled_by_cid.size and enabled_by_cid[cid]):
                                                base_color = color_by_cid[cid] if (color_by_cid is not None and cid < color_by_cid.shape[0]) else np.array((0, 255, 255), dtype=np.uint8)
                                                bgr = tuple(color if color is not None else base_color.tolist())
                                                mask_union = None
                                                for k in idx_arr:
                                                    if k >= N:
                                                        continue
                                                    m = mask_data_npy[k]
                                                    if mask_union is None:
                                                        mask_union = (m != 0)
                                                    else:
                                                        mask_union |= (m != 0)
                                                if mask_union is not None:
                                                    if roi_mask_clipping is not None and mask_union.shape == roi_mask_clipping.shape:
                                                        mask_union &= roi_mask_clipping
                                                    overlay[mask_union] = bgr
                                                    fast_union_done = True
                                except Exception:
                                    fast_union_done = False

                            if not fast_union_done:
                                for k in indices:
                                    if k >= N:
                                        continue
                                    cid = int(clss[k]) if clss is not None and k < len(clss) else None
                                    if cid is None:
                                        continue
                                    if enabled_by_cid is not None and cid < enabled_by_cid.size and not enabled_by_cid[cid]:
                                        continue
                                    base_color = color_by_cid[cid] if (color_by_cid is not None and cid < color_by_cid.shape[0]) else np.array((0, 255, 255), dtype=np.uint8)
                                    bgr = tuple(color if color is not None else base_color.tolist())
                                    m = mask_data_npy[k]
                                    
                                    # CLIPPING
                                    if roi_mask_clipping is not None and m.shape == roi_mask_clipping.shape:
                                        m = m & roi_mask_clipping
                                    
                                    overlay[m != 0] = bgr
                            cv2.addWeighted(overlay, 0.30, img, 1.0, 0.0, dst=img)
        except Exception:
            pass

        area_map_px, area_map_cm2 = {}, {}
        if need_area_any:
            require_px = global_area_mode in {AREA_MODE_PX, AREA_MODE_BOTH}
            area_map_px, area_map_cm2 = self._get_area_maps_for_result(
                result,
                name_prefix,
                (img.shape[0], img.shape[1]),
                require_px=require_px,
                require_cm2=need_area_cm2,
            )

        try:
            if xyxy is not None:
                indices = sel_idx if sel_idx is not None else range(len(xyxy))
                label_positions: list[int] = []

                for i in indices:
                    box = xyxy_i[i] if xyxy_i is not None else xyxy[i]
                    if xyxy_i is not None:
                        x1, y1, x2, y2 = box
                    else:
                        x1, y1, x2, y2 = [int(v) for v in box]
                    cid = int(clss[i]) if clss is not None and i < len(clss) else None
                    if cid is None:
                        continue
                    if enabled_by_cid is not None and cid < enabled_by_cid.size and not enabled_by_cid[cid]:
                        continue

                    # Cajas opcionales
                    base_color = color_by_cid[cid] if (color_by_cid is not None and cid < color_by_cid.shape[0]) else np.array((255, 255, 0), dtype=np.uint8)
                    bgr_box = tuple(color if color is not None else base_color.tolist())
                    if show_boxes:
                        cv2.rectangle(img, (x1, y1), (x2, y2), bgr_box, line_thickness)

                    # === FIX: ConstrucciÃ³n correcta de etiqueta (Clase + Conf + Area) ===
                    # Se permite entrar si hay nombres habilitados O confianza habilitada O si el modo global de Ã¡rea no es OFF
                    if not show_names and not show_confidence and global_area_mode == AREA_MODE_OFF:
                        continue

                    # Determinar escala de fuente
                    base_fs = font_scale
                    try:
                        area_scale = area_text_scale
                    except Exception:
                        area_scale = 0.4
                    area_scale = max(0.1, area_scale)
                    
                    # Decidir quÃ© mostrar de Ã¡rea
                    class_mode = AREA_MODE_OFF
                    cname = names.get(cid, str(cid)) if cid is not None else ""
                    if global_area_mode != AREA_MODE_OFF:
                        cfg = self.class_cfg.get(str(cname), {})
                        class_mode = self._class_area_mode(cfg, name_prefix, global_area_mode)
                    
                    # Construir partes del texto
                    parts = []
                    
                    # 1. Nombre y Confianza
                    if show_names or show_confidence:
                        text_parts = []
                        if show_names:
                            text_parts.append(cname)
                        if show_confidence:
                            conf_val = confs[i] if confs is not None else 0.0
                            text_parts.append(f"/[Conf: {conf_val:.2f}]")
                        
                        if text_parts:
                            parts.append(" ".join(text_parts))

                    # 2. Ãrea (si corresponde)
                    show_cm2 = class_mode in {AREA_MODE_CM2, AREA_MODE_BOTH}
                    show_px = class_mode in {AREA_MODE_PX, AREA_MODE_BOTH}
                    area_enabled_flag = (
                        area_enabled_by_cid is not None
                        and cid < area_enabled_by_cid.size
                        and area_enabled_by_cid[cid]
                    )
                    if not area_enabled_flag:
                        show_cm2 = False
                        show_px = False

                    if show_cm2:
                        area_cm2 = area_map_cm2.get(int(i))
                        if area_cm2 is not None and area_cm2 > 0:
                            parts.append(_format_area_cm_value(area_cm2))
                        elif not getattr(self, "_calibration_valid", False):
                            parts.append("ERR:CAL")
                        elif getattr(self, "calibration", None) is None:
                            parts.append("ERR:NoCAL")
                        else:
                            parts.append("ERR:Calc")

                    if show_px:
                        area_px = area_map_px.get(int(i))
                        if area_px is not None and area_px > 0:
                            parts.append(f"/[{area_px:.0f} px^2]")

                    if not parts:
                        continue

                    text_line = " ".join(parts)
                    # Ajustar escala si solo mostramos Ã¡rea (hacerla un poco mÃ¡s grande si se desea, o mantener consistencia)
                    # Si 'show_names' es False, quizÃ¡s queramos usar el tamaÃ±o de fuente base, no el reducido de Ã¡rea.
                    fs = base_fs if show_names else (base_fs * area_scale)
                    fs = max(0.1, fs)
                    size, _ = cv2.getTextSize(text_line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                    text_width, text_height = size
                    cx = int((x1 + x2) / 2)
                    pos_x = cx - text_width // 2
                    pos_y = y1 - 6
                    if pos_y < 10:
                        pos_y = y2 + text_height + int(6 * fs)

                    line_spacing = max(14, int(24 * fs))
                    y_top = pos_y
                    attempts = 0
                    while any(abs(y_top - existing) < line_spacing for existing in label_positions):
                        y_top += line_spacing
                        attempts += 1
                        if y_top > img.shape[0] - 10:
                            y_top = pos_y
                            while any(abs(y_top - existing) < line_spacing for existing in label_positions):
                                y_top -= line_spacing
                            break
                    y_top = int(max(10, min(img.shape[0] - 10, y_top)))
                    label_positions.append(y_top)

                    pos_x = int(max(5, min(img.shape[1] - text_width - 5, pos_x)))
                    base_thickness = max(1, line_thickness)
                    thickness_fg = max(1, int(round(base_thickness * area_scale * 0.6)))
                    thickness_bg = thickness_fg + 2

                    cv2.putText(
                        img,
                        text_line,
                        (pos_x, y_top),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs,
                        (0, 0, 0),
                        thickness_bg,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        text_line,
                        (pos_x, y_top),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs,
                        (255, 255, 255),
                        thickness_fg,
                        cv2.LINE_AA,
                    )
        except Exception:
            pass

    # ------------------------- ConfiguraciÃ³n de clases/colores -------------------------
    def _ensure_default_class_cfg(self):
        # Construye un conjunto de clases visto en ambos modelos
        names = set()
        if isinstance(self.name_to_id_m1, dict):
            names.update(self.name_to_id_m1.keys())
        if isinstance(self.name_to_id_m2, dict):
            names.update(self.name_to_id_m2.keys())

        if not names:
            # Intento anticipado leyendo directamente los modelos configurados
            names.update(self._peek_model_names(1))
            names.update(self._peek_model_names(2))

        # NUEVO: Limpiar clases que ya no existen en los modelos actuales
        # if names:
            # stale = [n for n in self.class_cfg.keys() if n not in names]
            # for n in stale:
                # del self.class_cfg[n]

        # Paleta de colores (BGR)
        palette = [
            (0, 0, 255),    # rojo
            (255, 0, 0),    # azul
            (0, 255, 0),    # verde
            (0, 255, 255),  # amarillo
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
            (0, 140, 255),  # naranja
            (147, 20, 255), # pÃºrpura
            (203, 192, 255),# rosa
            (50, 205, 50),  # lima
        ]

        if not names:
            # No hay informaciÃ³n de modelos todavÃ­a: conserva la configuraciÃ³n que exista
            for cfg in self.class_cfg.values():
                cfg.setdefault('color', palette[0])
                cfg.setdefault('thr', float(self.conf))
                cfg.setdefault('cm2_m1', False)
                cfg.setdefault('cm2_m2', False)
            return

        new_color_idx = 0
        for n in sorted(names):
            has_m1 = isinstance(self.name_to_id_m1, dict) and n in self.name_to_id_m1
            has_m2 = isinstance(self.name_to_id_m2, dict) and n in self.name_to_id_m2
            if n not in self.class_cfg:
                self.class_cfg[n] = {
                    'color': palette[new_color_idx % len(palette)],
                    'm1': bool(has_m1),
                    'm2': bool(has_m2),
                    'thr': float(self.conf),
                    'cm2_m1': False,
                    'cm2_m2': False,
                }
                new_color_idx += 1
            else:
                cfg = self.class_cfg[n]
                cfg.setdefault('color', palette[new_color_idx % len(palette)])
                cfg.setdefault('thr', float(self.conf))
                cfg.setdefault('cm2_m1', False)
                cfg.setdefault('cm2_m2', False)
                cfg['m1'] = bool(has_m1 and cfg.get('m1', True))
                cfg['m2'] = bool(has_m2 and cfg.get('m2', True))
                if not has_m1:
                    cfg['cm2_m1'] = False
                if not has_m2:
                    cfg['cm2_m2'] = False

    def _peek_model_names(self, model_idx: int) -> set[str]:
        path = self.model_path.get().strip() if model_idx == 1 else self.model_path2.get().strip()
        if not path:
            return set()
        try:
            return self._load_model_names(path)
        except Exception:
            return set()

    def _apply_class_config(self):
        # Calcula listas de IDs seleccionadas por modelo para usar en predict(classes=...)
        self.sel_ids_m1 = []
        self.sel_ids_m2 = []
        self._class_cache: dict[str, dict[str, np.ndarray]] = {}
        for name, cfg in self.class_cfg.items():
            if cfg.get('m1', True) and isinstance(self.name_to_id_m1, dict) and name in self.name_to_id_m1:
                self.sel_ids_m1.append(self.name_to_id_m1[name])
            if cfg.get('m2', True) and isinstance(self.name_to_id_m2, dict) and name in self.name_to_id_m2:
                self.sel_ids_m2.append(self.name_to_id_m2[name])
        self._publish_model_metadata()
        if not self.sel_ids_m1:
            self.sel_ids_m1 = None
        if not self.sel_ids_m2:
            self.sel_ids_m2 = None
        self._rebuild_class_cache()
        self._recompute_area_mode_flags()

    def _rebuild_class_cache(self) -> None:
        cache: dict[str, dict[str, np.ndarray]] = {"M1": {}, "M2": {}}
        for model_key, name_to_id in (("M1", self.name_to_id_m1), ("M2", self.name_to_id_m2)):
            if not isinstance(name_to_id, dict):
                cache[model_key]["enabled"] = np.zeros(0, dtype=bool)
                cache[model_key]["thr"] = np.zeros(0, dtype=np.float32)
                cache[model_key]["color"] = np.zeros((0, 3), dtype=np.uint8)
                cache[model_key]["area_mode"] = np.zeros(0, dtype=np.uint8)
                continue
            max_cid = max(name_to_id.values()) if name_to_id else -1
            size = max_cid + 1
            default_thr = float(self.conf)
            default_color = np.array((255, 255, 0), dtype=np.uint8)
            enabled = np.ones(size, dtype=bool)
            thr = np.full(size, default_thr, dtype=np.float32)
            color = np.tile(default_color, (size, 1))
            area_mode = np.zeros(size, dtype=np.uint8)
            for class_name, cid in name_to_id.items():
                cfg = self.class_cfg.get(class_name, {})
                try:
                    enabled_val = cfg.get('m1' if model_key == "M1" else 'm2', True)
                    enabled[cid] = bool(enabled_val)
                except Exception:
                    enabled[cid] = True
                try:
                    inherit = bool(cfg.get('thr_inherit', True))
                    thr_val = default_thr if inherit else cfg.get('thr', default_thr)
                    thr[cid] = float(_clamp01(thr_val, default_thr))
                except Exception:
                    thr[cid] = default_thr
                color_val = cfg.get('color') if isinstance(cfg, dict) else None
                if color_val is not None:
                    try:
                        color[cid] = np.array(color_val, dtype=np.uint8)
                    except Exception:
                        color[cid] = default_color
                else:
                    color[cid] = default_color
                try:
                    area_mode[cid] = int(self._class_area_enabled(cfg, "M1" if model_key == "M1" else "M2"))
                except Exception:
                    area_mode[cid] = 0
            if size > 0 and not enabled.any():
                enabled[:] = True
            cache[model_key]["enabled"] = enabled
            cache[model_key]["thr"] = thr
            cache[model_key]["color"] = color
            cache[model_key]["area_mode"] = area_mode
        self._class_cache = cache

    def _bgr_to_hex(self, bgr):
        try:
            b, g, r = bgr
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#ffffff"

    def _hex_to_bgr(self, hx: str):
        try:
            hx = hx.lstrip('#')
            r = int(hx[0:2], 16)
            g = int(hx[2:4], 16)
            b = int(hx[4:6], 16)
            return (b, g, r)
        except Exception:
            return (255, 255, 255)

    def _open_class_config(self):
        # DiÃ¡logo para configurar clases y colores
        self._ensure_default_class_cfg()
        dlg = tk.Toplevel(self.root)
        dlg.title("Clases y Colores")
        dlg.geometry("520x420")
        frm = ttk.Frame(dlg, padding=6)
        frm.pack(fill="both", expand=True)

        # Encabezados
        hdr = ttk.Frame(frm)
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Clase", width=22).grid(row=0, column=0, sticky="w")
        ttk.Label(hdr, text="M1", width=6).grid(row=0, column=1)
        ttk.Label(hdr, text="M2", width=6).grid(row=0, column=2)
        ttk.Label(hdr, text="Color", width=10).grid(row=0, column=3)

        canvas = tk.Canvas(frm)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Variables de ediciÃ³n
        vars_m1 = {}
        vars_m2 = {}
        vars_cm2_m1 = {}
        vars_cm2_m2 = {}
        vars_inherit = {}
        color_labels = {}

        def update_color_label(name):
            bgr = self.class_cfg[name]['color']
            hx = self._bgr_to_hex(bgr)
            color_labels[name].configure(background=hx)

        def choose_color(name):
            curr = self.class_cfg[name]['color']
            hx_init = self._bgr_to_hex(curr)
            c = colorchooser.askcolor(title=f"Color para {name}", initialcolor=hx_init)
            if c and c[1]:
                self.class_cfg[name]['color'] = self._hex_to_bgr(c[1])
                update_color_label(name)

        # Rellenar filas
        for r, name in enumerate(sorted(self.class_cfg.keys())):
            cfg = self.class_cfg[name]
            ttk.Label(inner, text=name, width=20).grid(row=r, column=0, sticky="w", padx=2, pady=2)
            v1 = tk.BooleanVar(value=bool(cfg.get('m1', True)))
            v2 = tk.BooleanVar(value=bool(cfg.get('m2', True)))
            cm1 = tk.BooleanVar(value=bool(cfg.get('cm2_m1', True)))
            cm2 = tk.BooleanVar(value=bool(cfg.get('cm2_m2', True)))
            vars_m1[name] = v1
            vars_m2[name] = v2
            vars_cm2_m1[name] = cm1
            vars_cm2_m2[name] = cm2
            ttk.Checkbutton(inner, variable=v1).grid(row=r, column=1)
            ttk.Checkbutton(inner, variable=v2).grid(row=r, column=2)
            ttk.Checkbutton(inner, variable=cm1).grid(row=r, column=3)
            ttk.Checkbutton(inner, variable=cm2).grid(row=r, column=4)
            sw = tk.Label(inner, text="      ", relief="groove")
            color_labels[name] = sw
            sw.grid(row=r, column=5, padx=4)
            update_color_label(name)
            ttk.Button(inner, text="Cambiar...", command=lambda n=name: choose_color(n)).grid(row=r, column=6, padx=4, sticky="w")
            # Slider de confianza
            row2 = r
            sf = ttk.Frame(inner)
            sf.grid(row=row2, column=7, padx=6, pady=2, sticky="we")
            lblv = ttk.Label(sf, text=f"{tvar.get():.2f}")
            lblv.pack(side="right")
            def make_scale(tv, lab):
                def _on_change(val):
                    try:
                        tv.set(round(float(val), 2))
                        lab.configure(text=f"{tv.get():.2f}")
                    except Exception:
                        pass
                return _on_change
            s = ttk.Scale(sf, from_=0.0, to=1.0, orient="horizontal", command=make_scale(tvar, lblv))
            s.set(float(tvar.get()))
            s.pack(fill="x", expand=True, side="left", padx=(0, 6))
        inner.grid_columnconfigure(4, weight=1)

        # Botones de acciÃ³n
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=6)
        def select_all(model_idx: int, val: bool, *, area: bool = False):
            for name in self.class_cfg.keys():
                if model_idx == 1:
                    target_dict = vars_cm2_m1 if area else vars_m1
                else:
                    target_dict = vars_cm2_m2 if area else vars_m2
                target_dict[name].set(val)

        quick_lbl = ttk.Label(btns, text="Seleccion rapida:")
        quick_lbl.pack(side="left", padx=(0, 8))

        display_frame = ttk.Frame(btns)
        display_frame.pack(side="left", padx=(0, 16))
        ttk.Button(display_frame, text="M1: Mostrar todo", command=lambda: select_all(1, True)).pack(side="left", padx=2)
        ttk.Button(display_frame, text="M1: Ocultar todo", command=lambda: select_all(1, False)).pack(side="left", padx=2)
        ttk.Button(display_frame, text="M2: Mostrar todo", command=lambda: select_all(2, True)).pack(side="left", padx=6)
        ttk.Button(display_frame, text="M2: Ocultar todo", command=lambda: select_all(2, False)).pack(side="left", padx=2)

        area_frame = ttk.Frame(btns)
        area_frame.pack(side="left")
        ttk.Button(area_frame, text="M1: Area ON", command=lambda: select_all(1, True, area=True)).pack(side="left", padx=2)
        ttk.Button(area_frame, text="M1: Area OFF", command=lambda: select_all(1, False, area=True)).pack(side="left", padx=2)
        ttk.Button(area_frame, text="M2: Area ON", command=lambda: select_all(2, True, area=True)).pack(side="left", padx=6)
        ttk.Button(area_frame, text="M2: Area OFF", command=lambda: select_all(2, False, area=True)).pack(side="left", padx=2)

        def apply_and_close(close=True):
            # Persiste cambios y recalcula ids
            for name in self.class_cfg.keys():
                self.class_cfg[name]['m1'] = bool(vars_m1[name].get())
                self.class_cfg[name]['m2'] = bool(vars_m2[name].get())
                self.class_cfg[name]['cm2_m1'] = bool(vars_cm2_m1[name].get())
                self.class_cfg[name]['cm2_m2'] = bool(vars_cm2_m2[name].get())
            self._apply_class_config()
            if close:
                dlg.destroy()

        ttk.Button(btns, text="Aplicar", command=lambda: apply_and_close(False)).pack(side="right", padx=4)
        ttk.Button(btns, text="Cerrar", command=dlg.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Guardar", command=apply_and_close).pack(side="right", padx=4)


    def _open_settings_dialog(self, tab_name: str | None = None):
        # DiÃ¡logo de ajustes que incluye clases/colores y parÃ¡metros avanzados
        self._ensure_default_class_cfg()
        dlg = tk.Toplevel(self.root)
        dlg.title("Ajustes")
        dlg.transient(self.root)
        dlg.geometry("1150x720")
        dlg.minsize(980, 640)
        dlg.resizable(True, True)
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(1, weight=1)
        def _on_settings_destroy(event):
            if event.widget is dlg:
                if getattr(self, "_settings_traces", None):
                    for var, tid in self._settings_traces:
                        try: var.trace_remove("write", tid)
                        except: pass
                    self._settings_traces = []
                if getattr(self, "combo_profiles_settings", None) is not None:
                    self.combo_profiles_settings = None
        dlg.bind("<Destroy>", _on_settings_destroy)
        self._settings_traces = []
        nb = ttk.Notebook(dlg)
        nb.grid(row=1, column=0, sticky="nsew")

        # --- Tab General ---
        tab_general = ttk.Frame(nb)
        nb.add(tab_general, text="⚙ General")
        if tab_name == "General":
            nb.select(tab_general)
        gf = ttk.Frame(tab_general, padding=10)
        gf.pack(fill="both", expand=True)
        gf.columnconfigure(0, weight=1)
        gf.columnconfigure(1, weight=1)
        gf.rowconfigure(2, weight=1)

        # Bloque de perfil global
        profile_frame = ttk.LabelFrame(gf, text="Perfil", padding=8)
        profile_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        profile_frame.columnconfigure(1, weight=1)

        self._label_with_info(
            profile_frame,
            "Perfil:",
            "detect_manchas.profiles.active",
            row=0,
            column=0,
            sticky="w",
            pady=2,
        )
        self.combo_profiles_settings = ttk.Combobox(profile_frame, state="readonly", values=(), width=28)
        self.combo_profiles_settings.grid(row=0, column=1, sticky="we", pady=2, padx=(0, 4))
        self.combo_profiles_settings.bind("<<ComboboxSelected>>", lambda *_: self._profiles_on_selected("settings"))

        ttk.Button(
            profile_frame,
            text="Guardar perfil",
            command=self._profiles_save_active,
        ).grid(row=0, column=2, sticky="e", pady=2)

        profile_actions = ttk.Frame(profile_frame)
        profile_actions.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Button(profile_actions, text="Nuevo", command=self._profiles_ui_new).pack(side="left", padx=2)
        ttk.Button(profile_actions, text="Duplicar", command=self._profiles_ui_duplicate).pack(side="left", padx=2)
        ttk.Button(profile_actions, text="Renombrar", command=self._profiles_ui_rename).pack(side="left", padx=2)
        ttk.Button(profile_actions, text="Eliminar", command=self._profiles_ui_delete).pack(side="left", padx=2)
        self._profiles_refresh_ui()

        # ------------------------------------------------------------------
        # COLUMNA IZQUIERDA: Origen y Modelos
        # ------------------------------------------------------------------
        left_gen = ttk.LabelFrame(gf, text="Origen y Modelos", padding=8)
        left_gen.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(0, 6), pady=(0, 6))
        left_gen.columnconfigure(1, weight=1)

        row = 0
        # Preset Modelos
        self._label_with_info(
            left_gen,
            "⭐ Favoritos:",
            "detect_manchas.models.preset",
            row=row,
            column=0,
            sticky="w",
            pady=2,
            font=("Segoe UI", 9, "italic"),
        )
        conf_dict = getattr(self, "config", {})
        if not isinstance(conf_dict, dict): conf_dict = {}
        p_root = conf_dict.get("presets", {})
        if not isinstance(p_root, dict): p_root = {}
        p_models = p_root.get("models", {})
        if not isinstance(p_models, dict): p_models = {}
        self.combo_p_models = ttk.Combobox(left_gen, textvariable=self.preset_models_var, values=sorted(p_models.keys()) if p_models else [])
        self.combo_p_models.grid(row=row, column=1, sticky="we", pady=2, padx=(0, 4))
        btn_fm = ttk.Frame(left_gen)
        btn_fm.grid(row=row, column=2, sticky="e")
        ttk.Button(btn_fm, text="Cargar", command=lambda: self._apply_preset("models", self.preset_models_var)).pack(side="left", padx=2)
        ttk.Button(btn_fm, text="Guardar", command=lambda: self._save_preset_ui("models", self.preset_models_var)).pack(side="left", padx=2)
        row += 1

        # Origen
        self._label_with_info(
            left_gen,
            "🌐 Origen:",
            "detect_manchas.source.mode",
            row=row,
            column=0,
            sticky="w",
            pady=4,
        )
        ttk.Combobox(left_gen, textvariable=self.source_mode, state="readonly", width=12,
                     values=["Archivo", "RTSP"]).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        # Modelo 1
        self._label_with_info(
            left_gen,
            "🧠 Modelo 1:",
            "detect_manchas.models.model1_path",
            row=row,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Entry(left_gen, textvariable=self.model_path).grid(row=row, column=1, sticky="we", pady=3, padx=(0, 4))
        ttk.Button(left_gen, text="...", width=3, command=self._browse_model).grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # Modelo 2
        self._label_with_info(
            left_gen,
            "🧠 Modelo 2:",
            "detect_manchas.models.model2_path",
            row=row,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Entry(left_gen, textvariable=self.model_path2).grid(row=row, column=1, sticky="we", pady=3, padx=(0, 4))
        ttk.Button(left_gen, text="...", width=3, command=self._browse_model2).grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # Archivo de video
        self._label_with_info(
            left_gen,
            "🎞 Archivo video:",
            "detect_manchas.source.video_path",
            row=row,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Entry(left_gen, textvariable=self.video_path).grid(row=row, column=1, sticky="we", pady=3, padx=(0, 4))
        ttk.Button(left_gen, text="...", width=3, command=self._browse_video).grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # ------------------------------------------------------------------
        # COLUMNA DERECHA: RTSP Entrada
        # ------------------------------------------------------------------
        right_gen = ttk.LabelFrame(gf, text="RTSP Entrada", padding=8)
        right_gen.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        right_gen.columnconfigure(1, weight=1)

        row_r = 0
        self._label_with_info(
            right_gen,
            "🌐 Host:",
            "detect_manchas.rtsp_in.host",
            row=row_r,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Entry(right_gen, textvariable=self.rtsp_host).grid(row=row_r, column=1, columnspan=2, sticky="we", pady=2)
        row_r += 1

        self._label_with_info(
            right_gen,
            "🔌 Puerto:",
            "detect_manchas.rtsp_in.port",
            row=row_r,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Entry(right_gen, textvariable=self.rtsp_port, width=8).grid(row=row_r, column=1, sticky="w", pady=2)
        row_r += 1

        self._label_with_info(
            right_gen,
            "👤 Usuario:",
            "detect_manchas.rtsp_in.user",
            row=row_r,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Entry(right_gen, textvariable=self.rtsp_user).grid(row=row_r, column=1, columnspan=2, sticky="we", pady=2)
        row_r += 1

        self._label_with_info(
            right_gen,
            "🔒 Contrasena:",
            "detect_manchas.rtsp_in.password",
            row=row_r,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Entry(right_gen, textvariable=self.rtsp_password, show="*").grid(row=row_r, column=1, columnspan=2, sticky="we", pady=2)
        row_r += 1

        self._label_with_info(
            right_gen,
            "🛣 Path:",
            "detect_manchas.rtsp_in.path",
            row=row_r,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Entry(right_gen, textvariable=self.rtsp_path).grid(row=row_r, column=1, columnspan=2, sticky="we", pady=2)
        row_r += 1

        # Preset RTSP Entrada
        self._label_with_info(
            right_gen,
            "⭐ Favoritos:",
            "detect_manchas.rtsp_in.preset",
            row=row_r,
            column=0,
            sticky="w",
            pady=2,
            font=("Segoe UI", 9, "italic"),
        )
        conf_dict = getattr(self, "config", {})
        if not isinstance(conf_dict, dict): conf_dict = {}
        p_root = conf_dict.get("presets", {})
        if not isinstance(p_root, dict): p_root = {}
        p_in = p_root.get("rtsp_in", {})
        if not isinstance(p_in, dict): p_in = {}
        self.combo_p_in = ttk.Combobox(right_gen, textvariable=self.preset_rtsp_in_var, values=sorted(p_in.keys()) if p_in else [])
        self.combo_p_in.grid(row=row_r, column=1, sticky="we", pady=2, padx=(0, 4))
        btn_fi = ttk.Frame(right_gen)
        btn_fi.grid(row=row_r, column=2, sticky="e")
        ttk.Button(btn_fi, text="Cargar", command=lambda: self._apply_preset("rtsp_in", self.preset_rtsp_in_var)).pack(side="left", padx=2)
        ttk.Button(btn_fi, text="Guardar", command=lambda: self._save_preset_ui("rtsp_in", self.preset_rtsp_in_var)).pack(side="left", padx=2)
        row_r += 1

        rtsp_btns = ttk.Frame(right_gen)
        rtsp_btns.grid(row=row_r, column=0, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Button(rtsp_btns, text="🛠 Configurar RTSP...", command=self._open_rtsp_dialog).pack(side="left", padx=(0, 6))
        ttk.Button(rtsp_btns, text="✅ Probar conexion", command=self._test_rtsp_connection).pack(side="left")
        row_r += 1

        # ------------------------------------------------------------------
        # COLUMNA DERECHA (fila 1): RTSP Salida
        # ------------------------------------------------------------------
        out_gen = ttk.LabelFrame(gf, text="Streaming RTSP Salida", padding=8)
        out_gen.grid(row=2, column=1, sticky="nsew", padx=(6, 0), pady=(6, 6))
        out_gen.columnconfigure(1, weight=1)

        row_o = 0
        self._check_with_info(
            out_gen,
            "🚀 Activar salida RTSP",
            self.rtsp_out_enable,
            "detect_manchas.rtsp_out.enable",
            row=row_o,
            column=0,
            columnspan=3,
            sticky="w",
            pady=2,
            command=self._on_rtsp_toggle,
        )
        row_o += 1

        self._label_with_info(
            out_gen,
            "🔗 URL:",
            "detect_manchas.rtsp_out.url",
            row=row_o,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Entry(out_gen, textvariable=self.rtsp_out_url).grid(row=row_o, column=1, columnspan=2, sticky="we", pady=2)
        row_o += 1

        self._label_with_info(
            out_gen,
            "🎧 Codec:",
            "detect_manchas.rtsp_out.codec",
            row=row_o,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Combobox(out_gen, textvariable=self.rtsp_out_codec, state="readonly",
                     values=["Auto (NVENC)", "libx264"], width=14).grid(row=row_o, column=1, sticky="w", pady=2)
        row_o += 1

        self._label_with_info(
            out_gen,
            "🚚 Transporte:",
            "detect_manchas.rtsp_out.transport",
            row=row_o,
            column=0,
            sticky="w",
            pady=2,
        )
        ttk.Combobox(out_gen, textvariable=self.rtsp_out_transport, state="readonly",
                     values=["TCP", "UDP"], width=14).grid(row=row_o, column=1, sticky="w", pady=2)
        row_o += 1

        # Preset RTSP Salida
        self._label_with_info(
            out_gen,
            "⭐ Favoritos:",
            "detect_manchas.rtsp_out.preset",
            row=row_o,
            column=0,
            sticky="w",
            pady=2,
            font=("Segoe UI", 9, "italic"),
        )
        conf_dict = getattr(self, "config", {})
        if not isinstance(conf_dict, dict): conf_dict = {}
        p_root = conf_dict.get("presets") if isinstance(conf_dict.get("presets"), dict) else {}
        p_out = p_root.get("rtsp_out", {}) if isinstance(p_root.get("rtsp_out"), dict) else {}
        self.combo_p_out = ttk.Combobox(out_gen, textvariable=self.preset_rtsp_out_var, values=sorted(p_out.keys()) if p_out else [])
        self.combo_p_out.grid(row=row_o, column=1, sticky="we", pady=2, padx=(0, 4))
        btn_fo = ttk.Frame(out_gen)
        btn_fo.grid(row=row_o, column=2, sticky="e")
        ttk.Button(btn_fo, text="Cargar", command=lambda: self._apply_preset("rtsp_out", self.preset_rtsp_out_var)).pack(side="left", padx=2)
        ttk.Button(btn_fo, text="Guardar", command=lambda: self._save_preset_ui("rtsp_out", self.preset_rtsp_out_var)).pack(side="left", padx=2)
        row_o += 1

        # --- Tab VisualizaciÃ³n ---
        tab_vis = ttk.Frame(nb)
        nb.add(tab_vis, text="📺 Visualizacion")
        vf = ttk.Frame(tab_vis, padding=10)
        vf.pack(fill="both", expand=True)
        vf.columnconfigure(0, weight=1)
        vf.columnconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # COLUMNA IZQUIERDA: Overlays en vÃ­deo
        # ------------------------------------------------------------------
        left_vis = ttk.LabelFrame(vf, text="Overlays en video", padding=8)
        left_vis.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))

        self._check_with_info(
            left_vis,
            "Mostrar cajas",
            self.show_boxes,
            "detect_manchas.visualization.show_boxes",
            row=0,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            left_vis,
            "Mostrar nombres",
            self.show_names,
            "detect_manchas.visualization.show_names",
            row=1,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            left_vis,
            "Mostrar confianza",
            self.show_confidence,
            "detect_manchas.visualization.show_confidence",
            row=2,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            left_vis,
            "Mostrar mascaras",
            self.show_masks,
            "detect_manchas.visualization.show_masks",
            row=3,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            left_vis,
            "Mascaras contorno",
            self.masks_as_contours,
            "detect_manchas.visualization.masks_contours",
            row=4,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            left_vis,
            "Retina masks",
            self.use_retina_masks,
            "detect_manchas.visualization.retina_masks",
            row=5,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            left_vis,
            "Resaltar manchas pequenas",
            self.highlight_tiny,
            "detect_manchas.visualization.highlight_tiny",
            row=6,
            column=0,
            sticky="w",
            pady=3,
        )

        ttk.Separator(left_vis, orient="horizontal").grid(row=7, column=0, sticky="we", pady=8)

        area_row = ttk.Frame(left_vis)
        area_row.grid(row=8, column=0, sticky="w", pady=3)
        ttk.Label(area_row, text="Etiqueta de area:").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(area_row, "detect_manchas.visualization.area_label").pack(side="left", padx=(6, 0))
        ttk.Combobox(area_row, textvariable=self.area_label_mode_display,
                     values=AREA_MODE_CHOICES_GLOBAL, state="readonly", width=14).pack(side="left", padx=(8, 0))

        # ------------------------------------------------------------------
        # COLUMNA DERECHA: Modelos activos + info
        # ------------------------------------------------------------------
        right_vis = ttk.LabelFrame(vf, text="Modelos activos", padding=8)
        right_vis.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))

        self._check_with_info(
            right_vis,
            "Habilitar Modelo 1",
            self.enable_m1,
            "detect_manchas.models.enable_m1",
            row=0,
            column=0,
            sticky="w",
            pady=3,
        )
        self._check_with_info(
            right_vis,
            "Habilitar Modelo 2",
            self.enable_m2,
            "detect_manchas.models.enable_m2",
            row=1,
            column=0,
            sticky="w",
            pady=3,
        )

        ttk.Separator(right_vis, orient="horizontal").grid(row=2, column=0, sticky="we", pady=8)

        ttk.Label(right_vis, text="Tip: Desactiva un modelo para\nahorrar recursos si no lo usas.",
                  font=("Segoe UI", 8), foreground="#666", justify="left").grid(row=3, column=0, sticky="w", pady=3)
        
        # --- Tab Avanzado ---
        tab_advanced = ttk.Frame(nb)
        nb.add(tab_advanced, text="🧰 Avanzado")
        af = ttk.Frame(tab_advanced, padding=10)
        af.pack(fill="both", expand=True)
        af.columnconfigure(0, weight=1)
        af.columnconfigure(1, weight=2)
        af.rowconfigure(0, weight=1)

        # ------------------------------------------------------------------
        # COLUMNA IZQUIERDA: Snapshot + Herramientas
        # ------------------------------------------------------------------
        left_adv = ttk.Frame(af)
        left_adv.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_adv.rowconfigure(1, weight=1)

        # Snapshot JSON
        snap_frame = ttk.LabelFrame(left_adv, text="Snapshot JSON", padding=8)
        snap_frame.pack(fill="x", pady=(0, 6))
        snap_frame.columnconfigure(1, weight=1)

        self._label_with_info(
            snap_frame,
            "Intervalo escritura (ms):",
            "detect_manchas.settings.advanced.snapshot_write_interval",
            row=0,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(snap_frame, from_=50, to=5000, increment=50, textvariable=self.snapshot_write_interval_ms, width=8).grid(
            row=0, column=1, sticky="w", pady=3)

        self._label_with_info(
            snap_frame,
            "Purgar historial (s):",
            "detect_manchas.settings.advanced.snapshot_purge_interval",
            row=1,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(snap_frame, from_=0, to=3600, increment=30, textvariable=self.snapshot_clean_interval_sec, width=8).grid(
            row=1, column=1, sticky="w", pady=3)

        # Herramientas
        tools_frame = ttk.LabelFrame(left_adv, text="Herramientas", padding=8)
        tools_frame.pack(fill="x", pady=(6, 0))

        ttk.Button(tools_frame, text="Entrenamiento...", command=self._open_trainer_gui).pack(anchor="w", pady=2, fill="x")
        ttk.Button(tools_frame, text="Capturas...", command=self._open_captures_folder).pack(anchor="w", pady=2, fill="x")
        ttk.Button(tools_frame, text="Manual", command=self._open_documentation_folder).pack(anchor="w", pady=2, fill="x")

        # ------------------------------------------------------------------
        # COLUMNA DERECHA: DiagnÃ³stico Heartbeat
        # ------------------------------------------------------------------
        hb_frame = ttk.LabelFrame(af, text="Diagnostico Heartbeat", padding=8)
        hb_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        hb_frame.columnconfigure(0, weight=1)
        hb_frame.rowconfigure(1, weight=1)
        hb_frame.rowconfigure(3, weight=1)

        self._label_with_info(
            hb_frame,
            "Enviado (Detector):",
            "detect_manchas.settings.advanced.heartbeat_sent",
            row=0,
            column=0,
            sticky="w",
            pady=(0, 2),
        )
        hb_local = tk.Text(hb_frame, height=6, state="disabled", wrap="word")
        hb_local.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self._hb_local_box = hb_local

        self._label_with_info(
            hb_frame,
            "Recibido (Visor):",
            "detect_manchas.settings.advanced.heartbeat_received",
            row=2,
            column=0,
            sticky="w",
            pady=(0, 2),
        )
        hb_remote = tk.Text(hb_frame, height=6, state="disabled", wrap="word")
        hb_remote.grid(row=3, column=0, sticky="nsew")
        self._hb_remote_box = hb_remote


        # --- Tab Clases ---
        tab_classes = ttk.Frame(nb)
        nb.add(tab_classes, text="🏷 Clases")

        line_color_v = "#7b859b"
        line_color_h = "#8591a6"

        ttk.Label(
            tab_classes,
            text=(
                "Activa para cada clase lo que debe mostrarse sobre el video:\n"
                "- 'Mostrar deteccion' habilita la visualizacion y filtrado de la clase en ese modelo.\n"
                "- 'Area' habilita el calculo de area de esa clase (para PLC/estadisticas). La visualizacion se controla en Visualizacion > Etiqueta de area."
            ),
            justify="left",
            wraplength=940,
        ).pack(fill="x", padx=10, pady=(8, 8))

        canvas = tk.Canvas(tab_classes, highlightthickness=0)
        vsb = ttk.Scrollbar(tab_classes, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=6)
        vsb.pack(side="right", fill="y", padx=(0, 10), pady=6)

        vars_m1 = {}
        vars_m2 = {}
        vars_thr = {}
        vars_inherit = {}
        vars_cm2_m1 = {}
        vars_cm2_m2 = {}
        color_labels = {}

        preview_m1 = set(self.name_to_id_m1.keys()) if isinstance(self.name_to_id_m1, dict) else self._peek_model_names(1)
        preview_m2 = set(self.name_to_id_m2.keys()) if isinstance(self.name_to_id_m2, dict) else self._peek_model_names(2)

        def _model_has_class(class_name: str, model_tag: str) -> bool:
            if model_tag == "M1":
                return class_name in preview_m1
            return class_name in preview_m2

        def _draw_placeholder(cell: tk.Frame, text: str) -> None:
            canvas = tk.Canvas(cell, height=56, bg="#fde7e7", highlightthickness=0)
            canvas.pack(fill="both", expand=True)

            def _paint(_event=None):
                canvas.delete("pattern")
                canvas.delete("label")
                w = max(1, canvas.winfo_width())
                h = max(1, canvas.winfo_height())
                step = 14
                for offset in range(-h, w, step):
                    canvas.create_line(offset, 0, offset + h, h, fill="#e05a5a", width=1, dash=(4, 4), tags="pattern")
                canvas.create_text(
                    w / 2,
                    h / 2,
                    text=text,
                    fill="#a32626",
                    font=("Segoe UI", 9, "italic"),
                    tags="label",
                    justify="center",
                )

            canvas.bind("<Configure>", _paint)
            _paint()

        inner.grid_columnconfigure(0, weight=3)
        inner.grid_columnconfigure(2, weight=3)
        inner.grid_columnconfigure(4, weight=3)
        inner.grid_columnconfigure(6, weight=2)
        inner.grid_columnconfigure(8, weight=2)

        # Fila de cabecera dentro de la tabla (scrollable)
        ttk.Label(inner, text="Clase", anchor="w").grid(row=0, column=0, padx=6, pady=(4, 4), sticky="w")
        tk.Frame(inner, width=2, bg=line_color_v).grid(row=0, column=1, sticky="ns", padx=2)
        ttk.Label(inner, text="Opciones Modelo M1", anchor="center").grid(row=0, column=2, padx=6, pady=(4, 4))
        tk.Frame(inner, width=2, bg=line_color_v).grid(row=0, column=3, sticky="ns", padx=2)
        ttk.Label(inner, text="Opciones Modelo M2", anchor="center").grid(row=0, column=4, padx=6, pady=(4, 4))
        tk.Frame(inner, width=2, bg=line_color_v).grid(row=0, column=5, sticky="ns", padx=2)
        ttk.Label(inner, text="Color", anchor="center").grid(row=0, column=6, padx=6, pady=(4, 4))
        tk.Frame(inner, width=2, bg=line_color_v).grid(row=0, column=7, sticky="ns", padx=2)
        ttk.Label(inner, text="Confianza minima", anchor="center").grid(row=0, column=8, padx=6, pady=(4, 4))
        tk.Frame(inner, height=3, bg=line_color_h).grid(row=1, column=0, columnspan=9, sticky="we")

        def update_color_label(name):
            bgr = self.class_cfg[name]['color']
            hx = self._bgr_to_hex(bgr)
            color_labels[name].configure(background=hx)

        def choose_color(name):
            curr = self.class_cfg[name]['color']
            hx_init = self._bgr_to_hex(curr)
            c = colorchooser.askcolor(title=f"Color para {name}", initialcolor=hx_init)
            if c and c[1]:
                self.class_cfg[name]['color'] = self._hex_to_bgr(c[1])
                update_color_label(name)

        def _ensure_class_defaults(name: str, cfg: dict) -> dict:
            cfg = cfg if isinstance(cfg, dict) else {}
            if "color" not in cfg or not (isinstance(cfg.get("color"), (list, tuple)) and len(cfg.get("color")) == 3):
                cfg["color"] = (255, 255, 0)
            if "m1" not in cfg:
                cfg["m1"] = False
            if "m2" not in cfg:
                cfg["m2"] = False
            cfg["thr_inherit"] = bool(cfg.get("thr_inherit", True))
            cfg["thr"] = float(_clamp01(cfg.get("thr", float(self.conf)), float(self.conf)))
            cfg["cm2_m1"] = bool(cfg.get("cm2_m1", False))
            cfg["cm2_m2"] = bool(cfg.get("cm2_m2", False))
            self.class_cfg[name] = cfg
            return cfg

        available_names = preview_m1.union(preview_m2)
        sorted_names = list(sorted(available_names))
        conf_trace_ids: list[str] = []
        for r, name in enumerate(sorted_names):
            cfg = _ensure_class_defaults(name, self.class_cfg.get(name, {}))
            row_base = 2 + r * 2

            class_cell = tk.Frame(inner, padx=8, pady=6)
            class_cell.grid(row=row_base, column=0, sticky="nsew")
            ttk.Label(class_cell, text=name, anchor="w").pack(anchor="w")

            tk.Frame(inner, width=2, bg=line_color_v).grid(row=row_base, column=1, sticky="ns")

            has_m1 = _model_has_class(name, "M1")
            has_m2 = _model_has_class(name, "M2")

            m1_cell = tk.Frame(inner, padx=10, pady=6)
            m1_cell.grid(row=row_base, column=2, sticky="nsew")
            if has_m1:
                v1 = tk.BooleanVar(value=bool(cfg.get('m1', True)))
                cm1 = tk.BooleanVar(value=bool(cfg.get('cm2_m1', True)))
                vars_m1[name] = v1
                vars_cm2_m1[name] = cm1
                ttk.Checkbutton(m1_cell, text="Mostrar deteccion", variable=v1).pack(anchor="w", pady=2)
                ttk.Checkbutton(m1_cell, text="Area", variable=cm1).pack(anchor="w", pady=2)
            else:
                vars_m1[name] = None
                vars_cm2_m1[name] = None
                _draw_placeholder(m1_cell, "No disponible\nModelo 1")

            tk.Frame(inner, width=2, bg=line_color_v).grid(row=row_base, column=3, sticky="ns")

            m2_cell = tk.Frame(inner, padx=10, pady=6)
            m2_cell.grid(row=row_base, column=4, sticky="nsew")
            if has_m2:
                v2 = tk.BooleanVar(value=bool(cfg.get('m2', True)))
                cm2 = tk.BooleanVar(value=bool(cfg.get('cm2_m2', True)))
                vars_m2[name] = v2
                vars_cm2_m2[name] = cm2
                ttk.Checkbutton(m2_cell, text="Mostrar deteccion", variable=v2).pack(anchor="w", pady=2)
                ttk.Checkbutton(m2_cell, text="Area", variable=cm2).pack(anchor="w", pady=2)
            else:
                vars_m2[name] = None
                vars_cm2_m2[name] = None
                _draw_placeholder(m2_cell, "No disponible\nModelo 2")

            tk.Frame(inner, width=2, bg=line_color_v).grid(row=row_base, column=5, sticky="ns")

            color_cell = tk.Frame(inner, padx=10, pady=6)
            color_cell.grid(row=row_base, column=6, sticky="nsew")
            sw = tk.Label(color_cell, text="      ", relief="groove")
            color_labels[name] = sw
            sw.pack(side="left", padx=(0, 6))
            ttk.Button(color_cell, text="Cambiar color...", command=lambda n=name: choose_color(n)).pack(side="left")
            update_color_label(name)

            tk.Frame(inner, width=2, bg=line_color_v).grid(row=row_base, column=7, sticky="ns")

            inherit_var = tk.BooleanVar(value=bool(cfg.get('thr_inherit', True)))
            vars_inherit[name] = inherit_var
            tvar = tk.DoubleVar(value=float(cfg.get('thr', float(self.conf))))
            vars_thr[name] = tvar
            conf_cell = tk.Frame(inner, padx=10, pady=6)
            conf_cell.grid(row=row_base, column=8, sticky="nsew")

            inherit_chk = tk.Checkbutton(conf_cell, text="Heredar", variable=inherit_var, onvalue=True, offvalue=False)
            inherit_chk.pack(side="top", anchor="w", pady=(0, 4))

            lblv = ttk.Label(conf_cell, text=f"{tvar.get():.2f}")
            lblv.pack(side="right", anchor="e")
            eff_lbl = ttk.Label(conf_cell, text="", foreground="#555")
            eff_lbl.pack(side="right", anchor="e", padx=(0, 8))

            def make_scale(tv, lab):
                def _on_change(val):
                    try:
                        tv.set(round(float(val), 2))
                        lab.configure(text=f"{tv.get():.2f}")
                    except Exception:
                        pass
                return _on_change

            s = ttk.Scale(conf_cell, from_=0.0, to=1.0, orient="horizontal", command=make_scale(tvar, lblv))
            s.set(float(tvar.get()))
            s.pack(fill="x", expand=True, side="left", padx=(0, 10))

            def _toggle_scale(var=inherit_var, sc=s, tv=tvar, lab=lblv, eff=eff_lbl):
                if var.get():
                    sc.state(["disabled"])
                    tv.set(round(float(self.conf), 2))
                    eff.configure(text=f"Ef: {tv.get():.2f} (global)")
                else:
                    sc.state(["!disabled"])
                    eff.configure(text=f"Ef: {tv.get():.2f} (custom)")
                lab.configure(text=f"{tv.get():.2f}")

            inherit_var.trace_add("write", lambda *_a, v=inherit_var, sc=s, tv=tvar, lab=lblv, eff=eff_lbl: _toggle_scale(v, sc, tv, lab, eff))
            _toggle_scale(inherit_var, s, tvar, lblv)

            def _on_global_conf_change(*_args, v=inherit_var, sc=s, tv=tvar, lab=lblv, eff=eff_lbl):
                try:
                    if not (lab.winfo_exists() and eff.winfo_exists()):
                        return
                except Exception:
                    return
                if v.get():
                    tv.set(round(float(self.conf), 2))
                    lab.configure(text=f"{tv.get():.2f}")
                    eff.configure(text=f"Ef: {tv.get():.2f} (global)")

            trace_id = self.conf_var.trace_add("write", _on_global_conf_change)
            conf_trace_ids.append(trace_id)

            tk.Frame(inner, height=3, bg=line_color_h).grid(row=row_base + 1, column=0, columnspan=9, sticky="we")

        if sorted_names:
            tk.Frame(inner, height=3, bg=line_color_h).grid(row=2 + len(sorted_names) * 2, column=0, columnspan=9, sticky="we")

        # Sincronizar ancho del canvas con el tab (sin limitar altura)
        def _sync_canvas_width(event=None):
            canvas.itemconfig(canvas.find_withtag("all")[0], width=canvas.winfo_width() - 4)
        canvas.bind("<Configure>", lambda e: (canvas.configure(scrollregion=canvas.bbox("all")), 
                                               canvas.itemconfig(canvas.find_withtag("all")[0], width=max(1, canvas.winfo_width() - 20)) if canvas.find_withtag("all") else None))

        def _cleanup_conf_traces():
            for tid in list(conf_trace_ids):
                try:
                    self.conf_var.trace_remove("write", tid)
                except Exception:
                    pass
            conf_trace_ids.clear()

        def _safe_close_dialog():
            _cleanup_conf_traces()
            try:
                dlg.destroy()
            except Exception:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _safe_close_dialog)

        # --- Tab Rendimiento ---
        tab_perf = ttk.Frame(nb)
        nb.add(tab_perf, text="📈 Rendimiento")
        pf = ttk.Frame(tab_perf, padding=10)
        pf.pack(fill="both", expand=True)
        pf.columnconfigure(0, weight=1)
        pf.columnconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # COLUMNA IZQUIERDA: Estrategia + Inferencia
        # ------------------------------------------------------------------
        left_perf = ttk.Frame(pf)
        left_perf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Estrategia
        strat_frame = ttk.LabelFrame(left_perf, text="Estrategia", padding=8)
        strat_frame.pack(fill="x", pady=(0, 6))
        strat_frame.columnconfigure(1, weight=1)

        self._label_with_info(
            strat_frame,
            "Modo:",
            "detect_manchas.settings.performance.mode",
            row=0,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Combobox(strat_frame, textvariable=self.perf_mode, values=["Auto", "Secuencial", "Paralelo"], width=12, state="readonly").grid(row=0, column=1, sticky="w", pady=3)

        self._check_with_info(
            strat_frame,
            "Auto-skip frames",
            self.auto_skip,
            "detect_manchas.settings.performance.auto_skip_frames",
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=3,
        )

        fps_row = ttk.Frame(strat_frame)
        fps_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(fps_row, text="FPS objetivo:").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(fps_row, "detect_manchas.settings.performance.target_fps").pack(side="left", padx=(6, 0))
        ttk.Spinbox(fps_row, from_=5, to=60, textvariable=self.target_fps, width=5).pack(side="left", padx=(6, 0))

        # Inferencia
        inf_frame = ttk.LabelFrame(left_perf, text="Inferencia", padding=8)
        inf_frame.pack(fill="x", pady=(6, 0))

        self._check_with_info(
            inf_frame,
            "Filtrar en inferencia (clases)",
            self.filter_in_model,
            "detect_manchas.settings.performance.filter_inference",
            row=0,
            column=0,
            sticky="w",
            pady=2,
        )
        self._check_with_info(
            inf_frame,
            "Agnostic NMS",
            self.agnostic_nms,
            "detect_manchas.settings.performance.agnostic_nms",
            row=1,
            column=0,
            sticky="w",
            pady=2,
        )
        self._check_with_info(
            inf_frame,
            "Half precision (FP16)",
            self.use_half,
            "detect_manchas.settings.performance.half_precision",
            row=2,
            column=0,
            sticky="w",
            pady=2,
        )
        self._check_with_info(
            inf_frame,
            "Habilitar Modelo 1",
            self.enable_m1,
            "detect_manchas.settings.performance.enable_m1",
            row=3,
            column=0,
            sticky="w",
            pady=2,
        )
        self._check_with_info(
            inf_frame,
            "Habilitar Modelo 2",
            self.enable_m2,
            "detect_manchas.settings.performance.enable_m2",
            row=4,
            column=0,
            sticky="w",
            pady=2,
        )

        # ------------------------------------------------------------------
        # COLUMNA DERECHA: TamaÃ±os + Herramientas
        # ------------------------------------------------------------------
        right_perf = ttk.Frame(pf)
        right_perf.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # TamaÃ±os y lÃ­mites
        size_frame = ttk.LabelFrame(right_perf, text="Tamanos y limites", padding=8)
        size_frame.pack(fill="x", pady=(0, 6))
        size_frame.columnconfigure(1, weight=1)

        self._label_with_info(
            size_frame,
            "Stride global:",
            "detect_manchas.settings.performance.stride_global",
            row=0,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(size_frame, from_=1, to=6, textvariable=self.det_stride, width=5).grid(row=0, column=1, sticky="w", pady=3)

        self._label_with_info(
            size_frame,
            "Stride M2:",
            "detect_manchas.settings.performance.stride_m2",
            row=1,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(size_frame, from_=1, to=10, textvariable=self.stride2_var, width=5).grid(row=1, column=1, sticky="w", pady=3)

        self._label_with_info(
            size_frame,
            "imgsz M1:",
            "detect_manchas.settings.performance.imgsz_m1",
            row=2,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(size_frame, from_=480, to=1600, increment=32, textvariable=self.imgsz1_var, width=6).grid(row=2, column=1, sticky="w", pady=3)

        self._label_with_info(
            size_frame,
            "imgsz M2:",
            "detect_manchas.settings.performance.imgsz_m2",
            row=3,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(size_frame, from_=480, to=1600, increment=32, textvariable=self.imgsz2_var, width=6).grid(row=3, column=1, sticky="w", pady=3)

        self._label_with_info(
            size_frame,
            "max_det:",
            "detect_manchas.settings.performance.max_det",
            row=4,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(size_frame, from_=1, to=2000, increment=10, textvariable=self.max_det_var, width=6).grid(row=4, column=1, sticky="w", pady=3)

        self._label_with_info(
            size_frame,
            "TopK dibujar:",
            "detect_manchas.settings.performance.topk_draw",
            row=5,
            column=0,
            sticky="w",
            pady=3,
        )
        ttk.Spinbox(size_frame, from_=0, to=200, textvariable=self.topk_draw, width=6).grid(row=5, column=1, sticky="w", pady=3)

        # Herramientas
        tool_perf = ttk.LabelFrame(right_perf, text="Herramientas", padding=8)
        tool_perf.pack(fill="both", expand=True, pady=(6, 0))
        tool_chart_row = ttk.Frame(tool_perf)
        tool_chart_row.pack(fill="x")
        ttk.Button(tool_chart_row, text="Grafica rendimiento", command=self._open_perf_plot_window).pack(side="left", fill="x", expand=True)
        if InfoIcon is not None:
            InfoIcon(tool_chart_row, "detect_manchas.settings.performance.chart").pack(side="left", padx=(6, 0))
        
        # Separador
        ttk.Separator(tool_perf, orient="horizontal").pack(fill="x", pady=10)
        
        # --- PerfTrace: Logs de rendimiento ---
        ttk.Label(tool_perf, text="Diagnostico Avanzado", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        
        # Frame para toggle + badge
        pt_toggle_row = ttk.Frame(tool_perf)
        pt_toggle_row.pack(fill="x", pady=(6, 4))
        
        # Cargar estado desde config
        pt_cfg = self.config.get("perf_trace", {})
        initial_enabled = bool(pt_cfg.get("enabled", False)) if isinstance(pt_cfg, dict) else False
        self.perf_trace_enabled.set(initial_enabled)
        
        # Toggle checkbox
        def _on_perftrace_toggle():
            enabled = self.perf_trace_enabled.get()
            self._set_perftrace_enabled(enabled)
            _update_perftrace_ui()
        
        pt_check_frame = ttk.Frame(pt_toggle_row)
        pt_check_frame.pack(side="left")
        pt_check = ttk.Checkbutton(
            pt_check_frame,
            text="PerfTrace: Logs de rendimiento",
            variable=self.perf_trace_enabled,
            command=_on_perftrace_toggle
        )
        pt_check.pack(side="left")
        if InfoIcon is not None:
            InfoIcon(pt_check_frame, "detect_manchas.settings.performance.perftrace_enabled").pack(side="left", padx=(6, 0))
        
        # Badge de estado (circulo de color + texto)
        pt_badge_frame = tk.Frame(pt_toggle_row)
        pt_badge_frame.pack(side="left", padx=(12, 0))
        
        pt_indicator = tk.Canvas(pt_badge_frame, width=12, height=12, highlightthickness=0)
        pt_indicator.pack(side="left")
        pt_indicator.create_oval(2, 2, 10, 10, fill="#c62828", outline="", tags="dot")
        
        pt_badge_label = tk.Label(pt_badge_frame, text="DESACTIVADO", fg="#c62828", font=("Segoe UI", 8, "bold"))
        pt_badge_label.pack(side="left", padx=(4, 0))
        
        # Frame para ruta
        pt_path_row = ttk.Frame(tool_perf)
        pt_path_row.pack(fill="x", pady=(2, 6))
        pt_path_label = ttk.Frame(pt_path_row)
        pt_path_label.pack(anchor="w")
        ttk.Label(pt_path_label, text="Archivo actual:", font=("Segoe UI", 8)).pack(side="left")
        if InfoIcon is not None:
            InfoIcon(pt_path_label, "detect_manchas.settings.performance.perftrace_path").pack(side="left", padx=(6, 0))
        
        pt_path_var = tk.StringVar(value="(no activo)")
        pt_path_entry = ttk.Entry(pt_path_row, textvariable=pt_path_var, state="readonly", font=("Segoe UI", 8))
        pt_path_entry.pack(fill="x", pady=(2, 0))
        
        # Frame para botones
        pt_btns_row = ttk.Frame(tool_perf)
        pt_btns_row.pack(fill="x")
        
        pt_btn_folder = ttk.Button(pt_btns_row, text="Abrir carpeta", command=self._open_perftrace_folder, state="disabled")
        pt_btn_folder.pack(side="left", padx=(0, 4))
        
        pt_btn_copy = ttk.Button(pt_btns_row, text="Copiar ruta", command=self._copy_perftrace_path, state="disabled")
        pt_btn_copy.pack(side="left")
        
        # FunciÃ³n para actualizar UI segÃºn estado
        def _update_perftrace_ui():
            enabled = self.perf_trace_enabled.get()
            if enabled:
                pt_indicator.itemconfig("dot", fill="#2e7d32")
                pt_badge_label.config(text="ACTIVADO", fg="#2e7d32")
                path = self._perftrace_log_path or "(creando...)"
                pt_path_var.set(path)
                pt_btn_folder.config(state="normal")
                pt_btn_copy.config(state="normal")
            else:
                pt_indicator.itemconfig("dot", fill="#c62828")
                pt_badge_label.config(text="DESACTIVADO", fg="#c62828")
                last_path = self._perftrace_log_path if self._perftrace_log_path else "(no activo)"
                pt_path_var.set(last_path if self._perftrace_log_path else "(no activo)")
                pt_btn_folder.config(state="disabled")
                pt_btn_copy.config(state="disabled")
        
        # Si ya estaba habilitado al abrir, activar ahora
        if initial_enabled and self._perftrace_handler is None:
            self._set_perftrace_enabled(True)
        _update_perftrace_ui()


        # --- Tab Apariencia ---
        tab_look = ttk.Frame(nb)
        nb.add(tab_look, text="🎨 Apariencia")
        lf = ttk.Frame(tab_look, padding=10)
        lf.pack(fill="both", expand=True)
        lf.columnconfigure(0, weight=1)
        lf.columnconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # COLUMNA IZQUIERDA: Dibujo
        # ------------------------------------------------------------------
        draw_frame = ttk.LabelFrame(lf, text="Dibujo", padding=8)
        draw_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        draw_frame.columnconfigure(1, weight=1)

        self._label_with_info(
            draw_frame,
            "Grosor linea:",
            "detect_manchas.settings.appearance.line_thickness",
            row=0,
            column=0,
            sticky="w",
            pady=4,
        )
        ttk.Spinbox(draw_frame, from_=1, to=6, textvariable=self.line_thickness, width=6).grid(row=0, column=1, sticky="w", pady=4)

        self._label_with_info(
            draw_frame,
            "Tamano texto:",
            "detect_manchas.settings.appearance.font_scale",
            row=1,
            column=0,
            sticky="w",
            pady=4,
        )
        ttk.Spinbox(draw_frame, from_=0.4, to=1.5, increment=0.1, textvariable=self.font_scale, width=6).grid(row=1, column=1, sticky="w", pady=4)

        self._label_with_info(
            draw_frame,
            "Escala etiqueta area:",
            "detect_manchas.settings.appearance.area_scale",
            row=2,
            column=0,
            sticky="w",
            pady=4,
        )
        ttk.Spinbox(draw_frame, from_=0.2, to=1.0, increment=0.05, textvariable=self.area_text_scale, width=6).grid(row=2, column=1, sticky="w", pady=4)

        # ------------------------------------------------------------------
        # COLUMNA DERECHA: Info
        # ------------------------------------------------------------------
        info_frame = ttk.LabelFrame(lf, text="Informacion", padding=8)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))

        ttk.Label(info_frame, text="Ajusta estos valores para\nmodificar como se dibujan\nlas detecciones en el video.",
                  font=("Segoe UI", 9), foreground="#555", justify="left").pack(anchor="w", pady=(0, 10))

        ttk.Label(info_frame, text="Tip: Un grosor de 2 y escala\nde 0.5 suele funcionar bien.",
                  font=("Segoe UI", 8), foreground="#777", justify="left").pack(anchor="w")

        # --- Tab Sectores ---
        tab_sectores = ttk.Frame(nb)
        nb.add(tab_sectores, text="🧭 Sectores")

        # Frame principal sin scroll (layout directo)
        # --- SCROLLABLE WRAPPER START ---
        canvas_sect = tk.Canvas(tab_sectores, highlightthickness=0)
        vsb_sect = ttk.Scrollbar(tab_sectores, orient="vertical", command=canvas_sect.yview)

        sf = ttk.Frame(canvas_sect, padding=10)
        sf_id = canvas_sect.create_window((0, 0), window=sf, anchor="nw")

        def _on_sf_cfg(e):
            canvas_sect.configure(scrollregion=canvas_sect.bbox("all"))
        sf.bind("<Configure>", _on_sf_cfg)

        def _on_canvas_cfg(e):
            if canvas_sect.winfo_exists():
                width = e.width
                canvas_sect.itemconfig(sf_id, width=width)
        canvas_sect.bind("<Configure>", _on_canvas_cfg)

        canvas_sect.configure(yscrollcommand=vsb_sect.set)

        vsb_sect.pack(side="right", fill="y")
        canvas_sect.pack(side="left", fill="both", expand=True)

        def _on_mousewheel(event):
            canvas_sect.yview_scroll(int(-1*(event.delta/120)), "units")
    
        # Bind seguro para scroll
        sf.bind("<Enter>", lambda _: canvas_sect.bind_all("<MouseWheel>", _on_mousewheel))
        sf.bind("<Leave>", lambda _: canvas_sect.unbind_all("<MouseWheel>"))
        # --- SCROLLABLE WRAPPER END ---

        # Layout de 2 columnas: izquierda fija con controles, derecha expandible con diagrama
        left_col = ttk.Frame(sf)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right_col = ttk.Frame(sf)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        
        sf.columnconfigure(0, weight=0, minsize=260)  # Controles fijos
        sf.columnconfigure(1, weight=3)  # Diagrama expandible (mÃ¡s peso)
        sf.rowconfigure(0, weight=1)

        # ------------------------------------------------------------------
        # COLUMNA IZQUIERDA: A) VISUALIZACIÃ“N + B) DELIMITACIÃ“N
        # ------------------------------------------------------------------
        row_l = 0
        
        # --- A) VISUALIZACIÃ“N ---
        ttk.Label(left_col, text="A) Visualizacion", font=("Segoe UI", 11, "bold")).grid(
            row=row_l, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row_l += 1

        self._check_with_info(
            left_col,
            "Mostrar sectorizacion",
            self.sector_mostrar,
            "detect_manchas.settings.sectors.show_sectorization",
            row=row_l,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2,
            command=self._on_sector_config_change,
        )
        row_l += 1

        self._check_with_info(
            left_col,
            "Mostrar etiquetas de sector",
            self.sector_mostrar_etiquetas,
            "detect_manchas.settings.sectors.show_labels",
            row=row_l,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2,
            command=self._on_sector_config_change,
        )
        row_l += 1

        self._check_with_info(
            left_col,
            "Mostrar borde de banda",
            self.sector_mostrar_borde_banda,
            "detect_manchas.settings.sectors.show_band_border",
            row=row_l,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2,
            command=self._on_sector_config_change,
        )
        row_l += 1

        # Opacidad y grosor de lÃ­neas
        frm_lineas = ttk.Frame(left_col)
        frm_lineas.grid(row=row_l, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(frm_lineas, text="Opacidad:").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frm_lineas, "detect_manchas.settings.sectors.line_opacity").pack(side="left", padx=(6, 0))
        ttk.Scale(frm_lineas, from_=0.1, to=1.0, variable=self.sector_opacidad_lineas, 
                  orient="horizontal", length=80, command=lambda e: self._on_sector_config_change()).pack(side="left", padx=4)
        ttk.Label(frm_lineas, text="Grosor:").pack(side="left", padx=(8,0))
        if InfoIcon is not None:
            InfoIcon(frm_lineas, "detect_manchas.settings.sectors.line_width").pack(side="left", padx=(6, 0))
        ttk.Spinbox(frm_lineas, from_=1, to=5, textvariable=self.sector_grosor_lineas, 
                    width=4, command=self._on_sector_config_change).pack(side="left")
        row_l += 1

        ttk.Separator(left_col, orient="horizontal").grid(row=row_l, column=0, columnspan=2, sticky="we", pady=12)
        row_l += 1

        # --- B) DELIMITACIÃ“N DE BANDA ---
        ttk.Label(left_col, text="B) Delimitacion de banda", font=("Segoe UI", 11, "bold")).grid(
            row=row_l, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row_l += 1

        ttk.Label(left_col, text="Clases que delimitan cada borde:", 
                  font=("Segoe UI", 9), foreground="#666").grid(row=row_l, column=0, columnspan=2, sticky="w")
        row_l += 1

        clases_disponibles = ["(Ninguno)", "(Auto)"] + sorted(self.class_cfg.keys()) if self.class_cfg else ["(Ninguno)"]

        self._label_with_info(
            left_col,
            "Arriba:",
            "detect_manchas.settings.sectors.border_top",
            row=row_l,
            column=0,
            sticky="w",
            pady=2,
        )
        combo_sup = ttk.Combobox(left_col, textvariable=self.sector_borde_sup, values=clases_disponibles, width=15)
        combo_sup.grid(row=row_l, column=1, sticky="w", pady=2)
        combo_sup.bind("<<ComboboxSelected>>", lambda e: self._on_sector_config_change())
        row_l += 1

        self._label_with_info(
            left_col,
            "Abajo:",
            "detect_manchas.settings.sectors.border_bottom",
            row=row_l,
            column=0,
            sticky="w",
            pady=2,
        )
        combo_inf = ttk.Combobox(left_col, textvariable=self.sector_borde_inf, values=clases_disponibles, width=15)
        combo_inf.grid(row=row_l, column=1, sticky="w", pady=2)
        combo_inf.bind("<<ComboboxSelected>>", lambda e: self._on_sector_config_change())
        row_l += 1

        self._label_with_info(
            left_col,
            "Izquierda:",
            "detect_manchas.settings.sectors.border_left",
            row=row_l,
            column=0,
            sticky="w",
            pady=2,
        )
        combo_izq = ttk.Combobox(left_col, textvariable=self.sector_borde_izq, values=clases_disponibles, width=15)
        combo_izq.grid(row=row_l, column=1, sticky="w", pady=2)
        combo_izq.bind("<<ComboboxSelected>>", lambda e: self._on_sector_config_change())
        row_l += 1

        self._label_with_info(
            left_col,
            "Derecha:",
            "detect_manchas.settings.sectors.border_right",
            row=row_l,
            column=0,
            sticky="w",
            pady=2,
        )
        combo_der = ttk.Combobox(left_col, textvariable=self.sector_borde_der, values=clases_disponibles, width=15)
        combo_der.grid(row=row_l, column=1, sticky="w", pady=2)
        combo_der.bind("<<ComboboxSelected>>", lambda e: self._on_sector_config_change())
        row_l += 1

        # ValidaciÃ³n
        self._lbl_warning_delim = ttk.Label(left_col, text="", foreground="#cc0000", font=("Segoe UI", 8), wraplength=200)
        self._lbl_warning_delim.grid(row=row_l, column=0, columnspan=2, sticky="w", pady=2)
        row_l += 1

        # BotÃ³n probar
        ttk.Button(left_col, text="Probar delimitacion", command=self._probar_delimitacion).grid(row=row_l, column=0, columnspan=2, sticky="w", pady=4)
        row_l += 1
        self._lbl_estado_delim = ttk.Label(left_col, text="", font=("Segoe UI", 8, "italic"))
        self._lbl_estado_delim.grid(row=row_l, column=0, columnspan=2, sticky="w")
        row_l += 1

        # ------------------------------------------------------------------
        # CONTINUACIÃ“N COLUMNA IZQUIERDA: C) DIVISIÃ“N + D) PRESETS
        # ------------------------------------------------------------------
        
        ttk.Separator(left_col, orient="horizontal").grid(row=row_l, column=0, columnspan=2, sticky="we", pady=10)
        row_l += 1
        
        # --- C) DIVISIÃ“N EN SECTORES ---
        ttk.Label(left_col, text="C) Division en sectores", font=("Segoe UI", 11, "bold")).grid(
            row=row_l, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row_l += 1

        self._label_with_info(
            left_col,
            "Tipo de malla:",
            "detect_manchas.settings.sectors.grid_mode",
            row=row_l,
            column=0,
            sticky="w",
            pady=2,
        )
        frm_modo = ttk.Frame(left_col)
        frm_modo.grid(row=row_l, column=0, columnspan=2, sticky="w", pady=2)

        def _on_modo_change(modo):
            self.sector_modo.set(modo)
            self._actualizar_visibilidad_spinners()
            self._actualizar_diagrama_sectores()
            self._on_sector_config_change()

        self._btn_modo_vert = ttk.Button(frm_modo, text="Vertical", width=9, 
                                          command=lambda: _on_modo_change("vertical"))
        self._btn_modo_vert.pack(side="left", padx=1)
        self._btn_modo_horiz = ttk.Button(frm_modo, text="Horizontal", width=9,
                                           command=lambda: _on_modo_change("horizontal"))
        self._btn_modo_horiz.pack(side="left", padx=1)
        self._btn_modo_rejilla = ttk.Button(frm_modo, text="Rejilla", width=9,
                                             command=lambda: _on_modo_change("rejilla"))
        self._btn_modo_rejilla.pack(side="left", padx=1)
        row_l += 1

        # Spinners condicionales
        self._frm_spinners_sector = ttk.Frame(left_col)
        self._frm_spinners_sector.grid(row=row_l, column=0, columnspan=2, sticky="w", pady=4)

        self._lbl_sect_vert = ttk.Label(self._frm_spinners_sector, text="Verticales:")
        self._icon_sect_vert = InfoIcon(self._frm_spinners_sector, "detect_manchas.settings.sectors.grid_verticals") if InfoIcon is not None else None
        self._spin_sect_vert = ttk.Spinbox(self._frm_spinners_sector, from_=1, to=10, 
                                            textvariable=self.sector_num_vert, width=5,
                                            command=lambda: (self._on_sector_config_change(), self._actualizar_diagrama_sectores()))
        self._lbl_sect_horiz = ttk.Label(self._frm_spinners_sector, text="Horizontales:")
        self._icon_sect_horiz = InfoIcon(self._frm_spinners_sector, "detect_manchas.settings.sectors.grid_horizontals") if InfoIcon is not None else None
        self._spin_sect_horiz = ttk.Spinbox(self._frm_spinners_sector, from_=1, to=10,
                                             textvariable=self.sector_num_horiz, width=5,
                                             command=lambda: (self._on_sector_config_change(), self._actualizar_diagrama_sectores()))
        
        self._actualizar_visibilidad_spinners()
        row_l += 1

        # Perspectiva
        self._check_with_info(
            left_col,
            "Seguir perspectiva de banda",
            self.sector_use_perspective,
            "detect_manchas.settings.sectors.perspective",
            row=row_l,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2,
            command=self._on_sector_config_change,
        )
        row_l += 1
        ttk.Label(left_col, text="(recomendado si la camara esta girada)", 
                  font=("Segoe UI", 7, "italic"), foreground="#777").grid(row=row_l, column=0, columnspan=2, sticky="w")
        row_l += 1

        ttk.Separator(left_col, orient="horizontal").grid(row=row_l, column=0, columnspan=2, sticky="we", pady=10)
        row_l += 1

        # --- D) PRESETS ---
        ttk.Label(left_col, text="D) Presets", font=("Segoe UI", 10, "bold")).grid(
            row=row_l, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row_l += 1

        self._label_with_info(
            left_col,
            "Preset de sectores:",
            "detect_manchas.settings.sectors.presets_select",
            row=row_l,
            column=0,
            columnspan=2,
            sticky="w",
            pady=2,
        )
        row_l += 1

        self.combo_preset_sector = ttk.Combobox(left_col, textvariable=self.preset_sector_var, width=20)
        self.combo_preset_sector.grid(row=row_l, column=0, columnspan=2, sticky="w", pady=2)
        
        initial_presets = []
        if self.config and "presets" in self.config and "sectores" in self.config["presets"]:
             initial_presets = sorted(self.config["presets"]["sectores"].keys())
        self.combo_preset_sector.config(values=initial_presets)
        row_l += 1

        preset_btns = ttk.Frame(left_col)
        preset_btns.grid(row=row_l, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Button(preset_btns, text="Cargar", command=self._cargar_preset_sector, width=8).pack(side="left", padx=2)
        ttk.Button(preset_btns, text="Guardar", command=self._guardar_preset_sector, width=8).pack(side="left", padx=2)
        ttk.Button(preset_btns, text="Restablecer", command=self._restablecer_sector_defaults).pack(side="left", padx=2)
        row_l += 1


        # ------------------------------------------------------------------
        # COLUMNA DERECHA: PANEL DE SECTORES INTERACTIVO (EXPANDIBLE)
        # ------------------------------------------------------------------
        
        # Crear widget SectorControlPanel (sin tamaÃ±o fijo, se expande)
        self._sector_panel = SectorControlPanel(
            right_col,
            on_change=self._on_sector_config_change
        )
        self._sector_panel.grid(row=0, column=0, sticky="nsew", pady=4, padx=4)
        right_col.rowconfigure(0, weight=1)
        right_col.columnconfigure(0, weight=1)
        
        # Cargar configuraciÃ³n existente en el panel
        if "sectores" in self.config:
            self._sector_panel.set_config(self.config["sectores"])
        
        # Pasar clases disponibles para chips de restricciones
        if self.class_cfg:
            self._sector_panel.set_available_classes(list(self.class_cfg.keys()))
        
        # Pasar clases de borde iniciales (vista cenital)
        bsup = self.sector_borde_sup.get().replace("(Ninguno)", "").replace("(Auto)", "")
        binf = self.sector_borde_inf.get().replace("(Ninguno)", "").replace("(Auto)", "")
        bizq = self.sector_borde_izq.get().replace("(Ninguno)", "").replace("(Auto)", "")
        bder = self.sector_borde_der.get().replace("(Ninguno)", "").replace("(Auto)", "")
        self._sector_panel.set_border_classes(top=bsup, bottom=binf, left=bizq, right=bder)
        
        # FunciÃ³n para actualizar el diagrama usando el nuevo SectorControlPanel
        def _actualizar_diagrama_sectores():
            modo = self.sector_modo.get()
            try:
                n_vert = max(1, int(self.sector_num_vert.get()))
                n_horiz = max(1, int(self.sector_num_horiz.get()))
            except:
                n_vert, n_horiz = 2, 2
            
            self._sector_panel.update_layout(modo, n_vert, n_horiz)
        
        self._actualizar_diagrama_sectores = _actualizar_diagrama_sectores
        _actualizar_diagrama_sectores()

        # NUEVO: Refrescar clases si cambian los modelos con el diÃ¡logo abierto
        def _refresh_sectores_ui(*_):
            if not dlg.winfo_exists(): return
            available = ["(Ninguno)", "(Auto)"] + sorted(self.class_cfg.keys()) if self.class_cfg else ["(Ninguno)"]
            for cb in (combo_sup, combo_inf, combo_izq, combo_der):
                if cb.winfo_exists():
                    curr = cb.get()
                    cb.config(values=available)
                    if curr not in available and curr not in ("(Ninguno)", "(Auto)"):
                        cb.set("(Ninguno)")
            if hasattr(self, "_sector_panel") and self._sector_panel.winfo_exists():
                self._sector_panel.set_available_classes(list(self.class_cfg.keys()))

        self._settings_traces.append((self.model_path, self.model_path.trace_add("write", _refresh_sectores_ui)))
        self._settings_traces.append((self.model_path2, self.model_path2.trace_add("write", _refresh_sectores_ui)))

        # ------------------------------------------------------------------
        # ANCHO COMPLETO: SECCIÃ“N PLEGABLE AVANZADO (CON SCROLL)
        # ------------------------------------------------------------------
        row_full = 1
        

        def _toggle_avanzado():
            self._sector_avanzado_visible.set(not self._sector_avanzado_visible.get())
            if self._sector_avanzado_visible.get():
                self._frm_sector_avanzado.grid()
                self._btn_toggle_avanzado.configure(text="Avanzado / Diagnostico")
                
                # Auto-expandir ventana con LIMITE
                try:
                    top = self._frm_sector_avanzado.winfo_toplevel()
                    top.update_idletasks() 
                    req_h = top.winfo_reqheight()
                    screen_h = top.winfo_screenheight()
                    
                    # Limite: Altura pantalla - 100px (barra tareas/margen)
                    max_h = screen_h - 100
                    target_h = min(req_h, max_h)
                    
                    if top.winfo_height() < target_h:
                        top.geometry(f"{top.winfo_width()}x{target_h}")
                except Exception:
                    pass
            else:
                self._frm_sector_avanzado.grid_remove()
                self._btn_toggle_avanzado.configure(text="Avanzado / Diagnostico")

        self._btn_toggle_avanzado = ttk.Button(sf, text="Avanzado / Diagnostico", 
                                                 command=_toggle_avanzado, width=28)
        self._btn_toggle_avanzado.grid(row=row_full, column=0, columnspan=2, sticky="w", pady=(8, 4))
        row_full += 1

        # LOGICA MODIFICADA: Eliminamos scroll y contenedores intermedios.
        # Directamente un LabelFrame en el grid principal que permitira expandir la ventana.
        self._frm_sector_avanzado = ttk.LabelFrame(sf, text="Configuracion avanzada", padding=6)
        self._frm_sector_avanzado.grid(row=row_full, column=0, columnspan=2, sticky="nsew", pady=2)
        # Ocultar inicialmente
        self._frm_sector_avanzado.grid_remove()

        # Configurar peso para que expanda si hay espacio, pero lo importante es que empuje la ventana
        sf.rowconfigure(row_full, weight=1)
        adv_row = 0

        # --- Fila 1: Modo Delim + Radio EstabilizaciÃ³n ---
        f1 = ttk.Frame(self._frm_sector_avanzado)
        f1.grid(row=adv_row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(f1, text="Estabilidad:").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(f1, "detect_manchas.settings.sectors.advanced.stability").pack(side="left", padx=(6, 0))
        for val in ["Baja", "Media", "Alta"]:
            ttk.Radiobutton(f1, text=val, value=val, variable=self.sector_estabilidad,
                            command=self._on_estabilidad_change).pack(side="left", padx=2)
        adv_row += 1

        # --- Fila 2: Fallo + MÃ¡rgenes ---
        f2 = ttk.Frame(self._frm_sector_avanzado)
        f2.grid(row=adv_row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(f2, text="Si falla:").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(f2, "detect_manchas.settings.sectors.advanced.fail_mode").pack(side="left", padx=(6, 0))
        ttk.Combobox(f2, textvariable=self.sector_comportamiento_fallo,
                     values=["Congelar", "Rectangulo", "Desactivar"], 
                     state="readonly", width=12).pack(side="left", padx=2)
        ttk.Label(f2, text=" Margen:").pack(side="left", padx=(4, 0))
        if InfoIcon is not None:
            InfoIcon(f2, "detect_manchas.settings.sectors.advanced.margin").pack(side="left", padx=(6, 0))
        ttk.Spinbox(f2, from_=0, to=20, textvariable=self.sector_inset, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=2)
        ttk.Label(f2, text=" Alpha:").pack(side="left", padx=(4, 0))
        if InfoIcon is not None:
            InfoIcon(f2, "detect_manchas.settings.sectors.advanced.alpha").pack(side="left", padx=(6, 0))
        ttk.Spinbox(f2, from_=0.0, to=1.0, increment=0.05, textvariable=self.sector_smooth_alpha, width=4,
                    command=self._on_sector_config_change).pack(side="left", padx=2)
        adv_row += 1

        ttk.Separator(self._frm_sector_avanzado, orient="horizontal").grid(row=adv_row, column=0, columnspan=3, sticky="we", pady=4)
        adv_row += 1

        # === Bordes curvos (ultra-compacto) ===
        frm_curvo = ttk.Frame(self._frm_sector_avanzado)
        frm_curvo.grid(row=adv_row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Checkbutton(frm_curvo, text="Bordes curvos",
                        variable=self.sector_curved_enabled,
                        command=self._on_sector_config_change).pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frm_curvo, "detect_manchas.settings.sectors.advanced.curved_edges").pack(side="left", padx=(6, 0))
        ttk.Label(frm_curvo, text=" Seg V/H:").pack(side="left", padx=(4,0))
        if InfoIcon is not None:
            InfoIcon(frm_curvo, "detect_manchas.settings.sectors.advanced.curve_bins").pack(side="left", padx=(6, 0))
        ttk.Spinbox(frm_curvo, from_=0, to=50, textvariable=self.sector_curved_bins_vert, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=1)
        ttk.Spinbox(frm_curvo, from_=0, to=50, textvariable=self.sector_curved_bins_horiz, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=1)
        adv_row += 1

        # === Padding de malla (ultra-compacto) ===
        frm_pad = ttk.Frame(self._frm_sector_avanzado)
        frm_pad.grid(row=adv_row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(frm_pad, text="Padding ????:").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frm_pad, "detect_manchas.settings.sectors.advanced.padding").pack(side="left", padx=(6, 0))
        ttk.Spinbox(frm_pad, from_=0, to=500, textvariable=self.sector_padding_top, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=1)
        ttk.Spinbox(frm_pad, from_=0, to=500, textvariable=self.sector_padding_bottom, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=1)
        ttk.Spinbox(frm_pad, from_=0, to=500, textvariable=self.sector_padding_left, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=1)
        ttk.Spinbox(frm_pad, from_=0, to=500, textvariable=self.sector_padding_right, width=3,
                    command=self._on_sector_config_change).pack(side="left", padx=1)
        adv_row += 1

        frm_quant = ttk.Frame(self._frm_sector_avanzado)
        frm_quant.grid(row=adv_row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(frm_quant, text="Cuantizacion ROI/Lineas (px):").pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frm_quant, "detect_manchas.settings.sectors.advanced.quant_step").pack(side="left", padx=(6, 0))
        ttk.Spinbox(frm_quant, from_=1, to=50, textvariable=self.sector_roi_quant_step, width=4,
                    command=self._on_sector_config_change).pack(side="left", padx=2)
        ttk.Spinbox(frm_quant, from_=1, to=50, textvariable=self.sector_line_quant_step, width=4,
                    command=self._on_sector_config_change).pack(side="left", padx=2)
        adv_row += 1

        ttk.Separator(self._frm_sector_avanzado, orient="horizontal").grid(row=adv_row, column=0, columnspan=3, sticky="we", pady=4)
        adv_row += 1

        # DiagnÃ³stico
        frm_diag = ttk.Frame(self._frm_sector_avanzado)
        frm_diag.grid(row=adv_row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Checkbutton(frm_diag, text="Debug overlay", 
                        variable=self.sector_debug_overlay,
                        command=self._on_sector_config_change).pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frm_diag, "detect_manchas.settings.sectors.advanced.debug_overlay").pack(side="left", padx=(6, 0))
        adv_row += 1

        # Botones de acciÃ³n (barra inferior siempre visible)
        btns = ttk.Frame(dlg)
        btns.grid(row=2, column=0, sticky="we", pady=8, padx=10)
        btns.columnconfigure(0, weight=1)

        def apply_changes(close=False):
            # Guardar cambios de clases
            for name in self.class_cfg.keys():
                v_m1 = vars_m1.get(name)
                v_m2 = vars_m2.get(name)
                a_m1 = vars_cm2_m1.get(name)
                a_m2 = vars_cm2_m2.get(name)
                self.class_cfg[name]['m1'] = bool(v_m1.get()) if v_m1 is not None else False
                self.class_cfg[name]['m2'] = bool(v_m2.get()) if v_m2 is not None else False
                self.class_cfg[name]['cm2_m1'] = bool(a_m1.get()) if a_m1 is not None else False
                self.class_cfg[name]['cm2_m2'] = bool(a_m2.get()) if a_m2 is not None else False
                try:
                    inherit_val = bool(vars_inherit[name].get()) if name in vars_inherit else True
                    self.class_cfg[name]['thr_inherit'] = inherit_val
                    thr_val = self.conf if inherit_val else float(vars_thr[name].get())
                    self.class_cfg[name]['thr'] = float(_clamp01(thr_val, self.conf))
                except Exception:
                    pass
            self._apply_class_config()

            # Guardar configuraciÃ³n de sectores (interactivos + spinners)
            if "sectores" not in self.config:
                self.config["sectores"] = {}
            
            # Obtener datos del panel interactivo (excluidos, sensibilidades extra)
            panel_data = self._sector_panel.get_config()
            self.config["sectores"]["excluidos"] = panel_data["excluidos"]
            self.config["sectores"]["sensibilidades"] = panel_data["sensibilidades"]
            # Guardamos sensibilidades extra en ajustes_locales para compatibilidad
            self.config["sectores"]["ajustes_locales"] = panel_data["sensibilidades"]
            if "restricciones_clase" in panel_data:
                self.config["sectores"]["restricciones_clase"] = panel_data["restricciones_clase"]
            
            # Persistir a disco
            self._save_config()
            self._restart_if_core_settings_changed()

            if close:
                _safe_close_dialog()

        ttk.Button(btns, text="Aplicar", command=lambda: apply_changes(False)).pack(side="right", padx=4)
        ttk.Button(btns, text="Guardar", command=lambda: apply_changes(True)).pack(side="right", padx=4)
        ttk.Button(btns, text="Cerrar", command=_safe_close_dialog).pack(side="right", padx=4)

    def _apply_preset(self, category, combo_var):
        """Aplica la configuraciÃ³n de un preset (alias) seleccionado."""
        alias = combo_var.get()
        if not alias: return
        
        presets = self.config.get("presets", {}).get(category, {})
        data = presets.get(alias)
        if not data: return
        
        if category == "models":
            self.model_path.set(data.get("model_path", ""))
            self.model_path2.set(data.get("model_path2", ""))
            self.video_path.set(data.get("video_path", ""))
            self._update_ui_model_names()
        elif category == "rtsp_in":
            self.rtsp_host.set(data.get("host", ""))
            self.rtsp_port.set(data.get("port", ""))
            self.rtsp_user.set(data.get("user", ""))
            self.rtsp_password.set(data.get("password", ""))
            path_value = data.get("path", "")
            self.rtsp_path.set(path_value)
            manual_enabled = data.get("manual_enabled")
            manual_url = data.get("manual_url")
            if manual_url is None and isinstance(path_value, str) and path_value.strip().startswith("rtsp://"):
                manual_url = path_value.strip()
                if manual_enabled is None:
                    manual_enabled = True
            if manual_enabled is None:
                manual_enabled = False
            self.rtsp_manual_enabled.set(bool(manual_enabled))
            if manual_url is not None:
                self.rtsp_manual_url.set(str(manual_url))
        elif category == "rtsp_out":
            self.rtsp_out_url.set(data.get("url", ""))
            self.rtsp_out_codec.set(data.get("codec", ""))
            self.rtsp_out_transport.set(data.get("transport", ""))
            
        self._set_status(f"Preset '{alias}' aplicado.")
        if category in {"models", "rtsp_in"}:
            self._restart_if_core_settings_changed()

    def _save_preset_ui(self, category, combo_var):
        """Guarda la configuraciÃ³n actual en un preset (alias)."""
        alias = combo_var.get().strip()
        if not alias:
            messagebox.showwarning("Aviso", "Introduce un nombre (alias) para el preset.")
            return
            
        data = {}
        if category == "models":
            data = {
                "model_path": self.model_path.get(),
                "model_path2": self.model_path2.get(),
                "video_path": self.video_path.get()
            }
        elif category == "rtsp_in":
            data = {
                "host": self.rtsp_host.get(),
                "port": self.rtsp_port.get(),
                "user": self.rtsp_user.get(),
                "password": self.rtsp_password.get(),
                "path": self.rtsp_path.get(),
                "manual_enabled": bool(self.rtsp_manual_enabled.get()),
                "manual_url": self.rtsp_manual_url.get(),
            }
        elif category == "rtsp_out":
            data = {
                "url": self.rtsp_out_url.get(),
                "codec": self.rtsp_out_codec.get(),
                "transport": self.rtsp_out_transport.get()
            }
            
        if "presets" not in self.config: self.config["presets"] = {"models":{}, "rtsp_in":{}, "rtsp_out":{}}
        self.config["presets"][category][alias] = data
        self._save_config()
        
        # Actualizar lista de valores del combo
        new_values = sorted(self.config["presets"][category].keys())
        if category == "models" and hasattr(self, "combo_p_models"):
            self.combo_p_models.config(values=new_values)
        elif category == "rtsp_in" and hasattr(self, "combo_p_in"):
            self.combo_p_in.config(values=new_values)
        elif category == "rtsp_out" and hasattr(self, "combo_p_out"):
            self.combo_p_out.config(values=new_values)
            
        self._set_status(f"Preset '{alias}' guardado correctamente.")
        messagebox.showinfo("Exito", f"Preset '{alias}' guardado.")

    # =========================================================================
    # PerfTrace: Fine-grained Performance Logging
    # =========================================================================
    def _resolve_perftrace_folder(self) -> str:
        """Resuelve la carpeta donde guardar logs de PerfTrace."""
        pt_cfg = self.config.get("perf_trace", {})
        log_dir_cfg = pt_cfg.get("log_dir", "auto") if isinstance(pt_cfg, dict) else "auto"
        
        if log_dir_cfg == "auto" or not log_dir_cfg:
            # Intentar carpeta junto al config: ../config/logs/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            primary = os.path.join(base_dir, "..", "config", "logs")
            try:
                os.makedirs(primary, exist_ok=True)
                test_file = os.path.join(primary, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return os.path.normpath(primary)
            except Exception:
                pass
            # Fallback a LOCALAPPDATA
            local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            fallback = os.path.join(local_app, "DetectManchas", "logs")
            try:
                os.makedirs(fallback, exist_ok=True)
                return os.path.normpath(fallback)
            except Exception:
                return os.path.normpath(os.path.join(base_dir, "logs"))
        else:
            try:
                os.makedirs(log_dir_cfg, exist_ok=True)
                return os.path.normpath(log_dir_cfg)
            except Exception:
                return os.path.normpath(os.path.join(os.path.dirname(__file__), "logs"))

    def _set_perftrace_enabled(self, enabled: bool) -> None:
        """Activa o desactiva PerfTrace en caliente."""
        import uuid
        from datetime import datetime
        
        if enabled:
            # Si ya hay un handler activo, no duplicar
            if self._perftrace_handler is not None:
                return
            
            # Generar ID de run y archivo
            self._perftrace_run_id = uuid.uuid4().hex[:8]
            self._perftrace_frame_counter = 0
            log_folder = self._resolve_perftrace_folder()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perf_trace_{timestamp}.log"
            self._perftrace_log_path = os.path.join(log_folder, filename)
            
            # Crear logger dedicado
            self._perftrace_logger = logging.getLogger(f"PerfTrace_{self._perftrace_run_id}")
            self._perftrace_logger.setLevel(logging.INFO)
            self._perftrace_logger.propagate = False
            
            # Crear handler con rotaciÃ³n (max 50MB, 3 backups)
            try:
                from logging.handlers import RotatingFileHandler
                handler = RotatingFileHandler(
                    self._perftrace_log_path,
                    maxBytes=50 * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8"
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._perftrace_handler = handler
                self._perftrace_logger.addHandler(handler)
            except Exception as e:
                print(f"[PerfTrace] Error creando handler: {e}", flush=True)
                self._perftrace_log_path = None
                self._perftrace_logger = None
                return
            
            # Cargar configuraciÃ³n
            pt_cfg = self.config.get("perf_trace", {})
            if isinstance(pt_cfg, dict):
                self._perftrace_config["slow_frame_ms"] = pt_cfg.get("slow_frame_ms", 45)
                self._perftrace_config["det_threshold"] = pt_cfg.get("det_threshold", 15)
                self._perftrace_config["baseline_interval"] = pt_cfg.get("baseline_interval", 100)
            
            print(f"[PerfTrace] ACTIVADO - Log: {self._perftrace_log_path}", flush=True)
            LOGGER.info(f"PerfTrace enabled: {self._perftrace_log_path}")
            
        else:
            # Desactivar: cerrar handler
            if self._perftrace_handler is not None:
                try:
                    self._perftrace_handler.flush()
                    self._perftrace_handler.close()
                except Exception:
                    pass
                if self._perftrace_logger is not None:
                    self._perftrace_logger.removeHandler(self._perftrace_handler)
            self._perftrace_handler = None
            self._perftrace_logger = None
            print(f"[PerfTrace] DESACTIVADO - Ãšltimo log: {self._perftrace_log_path}", flush=True)
            # Mantener _perftrace_log_path para referencia
        
        # Persistir en config
        if "perf_trace" not in self.config:
            self.config["perf_trace"] = {}
        self.config["perf_trace"]["enabled"] = enabled
        self._save_config_debounced()

    def _log_perftrace_event(self, event_type: str, data: dict) -> None:
        """Escribe un evento de rendimiento en formato JSON Lines."""
        if self._perftrace_logger is None:
            return
        
        from datetime import datetime
        import json
        
        self._perftrace_frame_counter += 1
        
        # Condiciones de logging
        cfg = self._perftrace_config
        slow_ms = cfg.get("slow_frame_ms", 45)
        det_thr = cfg.get("det_threshold", 15)
        baseline = cfg.get("baseline_interval", 100)
        
        total_ms = data.get("total_ms", 0)
        det_count = data.get("det_count", 0)
        frame_idx = self._perftrace_frame_counter
        
        # Solo loguear si cumple condiciones
        should_log = (
            total_ms >= slow_ms or
            det_count >= det_thr or
            (frame_idx % baseline == 0)
        )
        
        if not should_log:
            return
        
        event = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "run": self._perftrace_run_id,
            "frame": frame_idx,
            "event": event_type,
        }
        event.update(data)
        
        try:
            self._perftrace_logger.info(json.dumps(event, ensure_ascii=False))
        except Exception:
            pass

    def _open_perftrace_folder(self) -> None:
        """Abre la carpeta del log actual en el explorador."""
        if not self._perftrace_log_path or not os.path.exists(self._perftrace_log_path):
            messagebox.showinfo("PerfTrace", "No hay un archivo de log activo.\nActiva PerfTrace primero.")
            return
        folder = os.path.dirname(self._perftrace_log_path)
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la carpeta:\n{e}")

    def _copy_perftrace_path(self) -> None:
        """Copia la ruta del log al portapapeles."""
        path = self._perftrace_log_path or "(no activo)"
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.root.update()
            self._set_status("Ruta de PerfTrace copiada al portapapeles.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo copiar:\n{e}")

    def _open_perf_plot_window(self):
        # Ventana con grÃ¡fica de FPS y Detecciones en tiempo real
        try:
            if self.perf_plot_win is not None and self.perf_plot_win.winfo_exists():
                self.perf_plot_win.deiconify()
                self.perf_plot_win.lift()
                return
        except Exception:
            pass

        win = tk.Toplevel(self.root)
        win.title("Analisis de Rendimiento y Detecciones")
        win.geometry("850x500")
        self.perf_plot_win = win
        win.minsize(700, 400)
        
        # Panel superior para controles
        header = tk.Frame(win, bg="#f8f9fa", pady=8, padx=12)
        header.pack(side="top", fill="x")
        
        tk.Label(header, text="Clase a rastrear:", bg="#f8f9fa", font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
        
        # Obtener clases disponibles
        classes = sorted(list(self._load_model_names(self.model_path.get()) | self._load_model_names(self.model_path2.get())))
        if not classes:
            classes = ["(Sin clases)"]
            
        class_selector = ttk.Combobox(header, textvariable=self.perf_tracked_class, values=classes, state="readonly", width=25)
        class_selector.pack(side="left")
        
        # Limpiar historial si cambia la clase para no mezclar datos viejos
        def _on_class_change(*_):
            with self.perf_lock:
                self.detection_series.clear()
        self.perf_tracked_class.trace_add("write", _on_class_change)

        canvas = tk.Canvas(win, background="white", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Cerrar limpiamente
        def _on_close():
            try:
                if win and win.winfo_exists():
                    win.destroy()
            finally:
                self.perf_plot_win = None
        win.protocol("WM_DELETE_WINDOW", _on_close)
        
        # Rutina de dibujo
        def _draw():
            try:
                if self.perf_plot_win is None or (not win.winfo_exists()):
                    return
                W = max(1, canvas.winfo_width())
                H = max(1, canvas.winfo_height())
                canvas.delete("all")
                
                # MÃ¡rgenes y Ã¡rea de grÃ¡fica
                L, R, T, B = 65, 65, 25, 45  # MÃ¡s margen derecho para el segundo eje Y
                x0, y0 = L, H - B
                x1, y1 = W - R, T
                plot_w = max(10, x1 - x0)
                plot_h = max(10, y0 - y1)

                with self.perf_lock:
                    fps_data = list(self.fps_series)
                    det_data = list(self.detection_series)

                if len(fps_data) >= 2:
                    t_first = fps_data[0][0]
                    t_last = fps_data[-1][0]
                    time_span = max(1.0, t_last - t_first)
                    label_text = f"Tiempo (s) - Ultimos {int(PERF_HISTORY_SECONDS)}s"
                else:
                    t_first = 0.0
                    time_span = 60.0
                    label_text = "Esperando datos..."

                # Determinar escalas
                try:
                    tgt_fps = float(self.target_fps.get())
                except Exception:
                    tgt_fps = 25.0
                
                max_fps_val = max([v for _, v in fps_data] + [tgt_fps]) if fps_data else 30.0
                ymax_fps = max(5.0, max_fps_val * 1.25)
                
                max_det_val = max([v for _, v in det_data] + [5.0]) if det_data else 5.0
                ymax_det = max(5.0, max_det_val * 1.25)

                # Dibuja ejes
                canvas.create_line(x0, y0, x1, y0, fill="#ccc", width=1)  # X
                canvas.create_line(x0, y0, x0, y1, fill="#0066cc", width=1.5)  # Y1 (FPS)
                canvas.create_line(x1, y0, x1, y1, fill="#e67e22", width=1.5)  # Y2 (Dets)
                
                canvas.create_text((x0 + x1) // 2, H - 15, text=label_text, fill="#555", font=("Segoe UI", 9))
                canvas.create_text(25, (y0 + y1) // 2, text="FPS", fill="#0066cc", font=("Segoe UI", 9, "bold"), angle=90)
                canvas.create_text(W - 25, (y0 + y1) // 2, text="Detecciones", fill="#e67e22", font=("Segoe UI", 9, "bold"), angle=270)

                # LÃ­neas de rejilla horizontales (usando escala FPS)
                for k in range(0, 6):
                    ratio = k / 5.0
                    y = y0 - ratio * plot_h
                    canvas.create_line(x0, y, x1, y, fill="#f0f0f0")
                    # Labels FPS
                    fps_val = ratio * ymax_fps
                    canvas.create_text(x0 - 10, y, text=f"{fps_val:.0f}", anchor="e", fill="#0066cc", font=("Consolas", 8))
                    # Labels Detecciones
                    det_val = ratio * ymax_det
                    canvas.create_text(x1 + 10, y, text=f"{det_val:.1f}", anchor="w", fill="#e67e22", font=("Consolas", 8))

                # LÃ­nea objetivo FPS
                y_tgt = y0 - (tgt_fps / ymax_fps) * plot_h
                canvas.create_line(x0, y_tgt, x1, y_tgt, fill="#cc0000", dash=(4, 2), width=1)
                canvas.create_text(x0 + 10, y_tgt - 8, text=f"Target: {tgt_fps:.0f} FPS", anchor="w", fill="#cc0000", font=("Segoe UI", 8, "italic"))

                # Dibujar serie FPS (Azul)
                if len(fps_data) >= 2:
                    pts_fps = []
                    for t, v in fps_data:
                        xx = x0 + ((t - t_first) / time_span) * plot_w
                        yy = y0 - (v / ymax_fps) * plot_h
                        pts_fps.extend([xx, yy])
                    canvas.create_line(*pts_fps, fill="#0066cc", width=2, joinstyle="round")
                    curr_fps = fps_data[-1][1]
                    canvas.create_text(x0 + 10, y1 + 10, anchor="nw", text=f"FPS Actual: {curr_fps:.1f}", fill="#0066cc", font=("Segoe UI", 9, "bold"))

                # Dibujar serie Detecciones (Naranja)
                if len(det_data) >= 2:
                    pts_det = []
                    for t, v in det_data:
                        xx = x0 + ((t - t_first) / time_span) * plot_w
                        yy = y0 - (v / ymax_det) * plot_h
                        pts_det.extend([xx, yy])
                    canvas.create_line(*pts_det, fill="#e67e22", width=2, joinstyle="round")
                    curr_det = det_data[-1][1]
                    cname = self.perf_tracked_class.get() or "Seleccionada"
                    canvas.create_text(x1 - 10, y1 + 10, anchor="ne", text=f"{cname}: {curr_det:.0f}", fill="#e67e22", font=("Segoe UI", 9, "bold"))
                elif not det_data and not fps_data:
                    canvas.create_text((x0+x1)//2, (y0+y1)//2, text="Esperando datos de ejecucion...", fill="#999")

                win.after(500, _draw)
            except Exception as e:
                print(f"[Grafica] Error en draw: {e}")
                try: win.after(1000, _draw)
                except: pass
        
        win.after(300, _draw)

def main():
    # -----------------------------------------------------------
    # NOTA: Se recomienda usar init.py como punto de entrada
    # principal del programa para asegurar el correcto arranque
    # de todos los servicios.
    # -----------------------------------------------------------
    print("Iniciando desde detect_manchas_gui_rtsp.py...")
    print("Para un arranque completo de servicios, use: python init.py")
    
    # Auto-inicio de MediaMTX (Servidor RTSP)
    try:
        from utils import ensure_mediamtx_running, set_app_logo
        ensure_mediamtx_running()
    except Exception as e:
        print(f"Advertencia: No se pudo verificar MediaMTX. Detalles: {e}")
    # -----------------------------------------------------------

    root = tk.Tk()
    root.title("Deteccion de Manchas")
    
    # Configurar el logo de la aplicación
    set_app_logo(root)

    
    # ConfiguraciÃ³n de la ventana principal (mÃ¡s grande por defecto)
    root.geometry("1280x900")  # TamaÃ±o inicial mÃ¡s grande
    root.minsize(1024, 768)     # TamaÃ±o mÃ­nimo mÃ¡s grande
    
    # Estilo visual mÃ¡s limpio
    style = ttk.Style()
    style.configure('TButton', padding=6, font=('Arial', 10))
    style.configure('TLabel', padding=4)
    style.configure('TEntry', padding=4)
    
    # Hacer que la ventana ocupe toda la pantalla
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.state('zoomed')  # Maximizar la ventana
    
    # Iniciar la aplicaciÃ³n
    app = DetectorGUI(root)
    
    # Iniciar el bucle principal
    root.mainloop()


if __name__ == "__main__":
    main()








