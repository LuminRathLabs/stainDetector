# -*- coding: utf-8 -*-
"""RTSP Viewer ligero para Raspberry (Tkinter + OpenCV).

Pensado para Raspbian/Debian:
- No mete parametros en la URL (evita 404 del servidor RTSP).
- En Linux solo fuerza transporte via OPENCV_FFMPEG_CAPTURE_OPTIONS con formato simple.
- Reduce lag: buffer minimo, descarta frames viejos, limita FPS de pintado y baja resolucion.
- Mantiene heartbeat y overlays del detector.

Requisitos:
  sudo apt install python3-opencv python3-tk
  (en tu venv) pip install pillow numpy
"""

from __future__ import annotations

import json
import logging
import os
import sys
import socket
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from itertools import cycle
from tkinter import messagebox, ttk
from typing import Callable, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


# Cambia esto si cambia el stream en el detector
RTSP_STREAM_URL = "rtsp://10.176.3.49:554/decapado_sup"

# Heartbeat
HEARTBEAT_DETECTOR_PORT = 9101
HEARTBEAT_VIEWER_PORT = 9102
HEARTBEAT_INTERVAL_SEC = 5.0
_HEARTBEAT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Latencia / watchdog
STREAM_STALE_SEC = 2.0   # si no llega frame nuevo en este tiempo, muestra placeholder
RECONNECT_BACKOFF_MAX = 8.0

# Ajustes de rendimiento para Raspberry
IS_POSIX = os.name == "posix"
MAX_FRAME_WIDTH_POSIX = 800           # baja resolucion antes de pintar
DISPLAY_FPS_POSIX = 25            # limite de FPS de refresco en pantalla
DISPLAY_FPS_WINDOWS = 25

# UI
UI_BG = "#f5f3ef"
UI_PANEL_BG = "#ffffff"
UI_HEADER_BG = "#18333d"
UI_HEADER_FG = "#f6f2ea"
UI_ACCENT = "#1f7a8c"
UI_ACCENT_DARK = "#145a67"
UI_BORDER = "#d7d2c9"
UI_TEXT = "#1f2933"
UI_MUTED = "#5f6c7b"
PLACEHOLDER_BG = "#111417"
PLACEHOLDER_FG = "#f6f2ea"
PLACEHOLDER_MUTED = "#9aa3ad"
STATUS_COLORS = {
    "idle": "#8a8f98",
    "connecting": "#f0a202",
    "live": "#2d936c",
    "warning": "#e07a5f",
    "error": "#c44536",
}
MAX_LOG_LINES = 400

if getattr(sys, 'frozen', False):
    # Si esta "congelado" (PyInstaller), usamos la carpeta del .exe
    SETTINGS_DIR = os.path.dirname(sys.executable)
else:
    # Si es script normal, usamos la carpeta del .py
    SETTINGS_DIR = os.path.dirname(__file__)

SETTINGS_FILE = os.path.join(SETTINGS_DIR, "rtsp_viewer_settings.json")
BRIGHTNESS_MIN = -20
BRIGHTNESS_MAX = 80
BRIGHTNESS_STEP = 3
CONTRAST_MIN = 80
CONTRAST_MAX = 140
CONTRAST_STEP = 5
DEFAULT_BRIGHTNESS = 0
DEFAULT_CONTRAST = 100


# Logger simple
LOGGER = logging.getLogger("RTSPViewer")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _normalize_rtsp_url(url: str, transport: str) -> str:
    """En Windows mantenemos rtsp_transport en query.
    En Linux no tocamos la URL para evitar 404.
    """
    if IS_POSIX:
        return url

    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if parsed.scheme.lower() != "rtsp":
        return url

    normalized = (transport or "").strip().lower()
    if normalized not in {"tcp", "udp"}:
        normalized = "tcp"

    params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "rtsp_transport"]
    params.append(("rtsp_transport", normalized))
    return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))


def _set_ffmpeg_transport(transport: str) -> None:
    """Fuerza TCP/UDP usando OPENCV_FFMPEG_CAPTURE_OPTIONS.
    Formato recomendado por OpenCV: key;value|key;value.
    En Linux solo usamos rtsp_transport para no romper FFmpeg de Debian.
    """
    t = (transport or "").strip().lower()
    if t not in {"tcp", "udp"}:
        t = "tcp"

    if IS_POSIX:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{t}"
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{t}|timeout;15000000|stimeout;15000000"


def _load_font(size: int = 22) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
        "segoeui.ttf",
        "arial.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


_OVERLAY_FONT = _load_font(22)
_PLACEHOLDER_FONT = _load_font(28)
_PLACEHOLDER_SUB_FONT = _load_font(16)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = (color or "#ffffff").strip()
    if color.startswith("#") and len(color) == 7:
        try:
            return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        except ValueError:
            pass
    return (255, 255, 255)


@dataclass
class _OverlayMessage:
    text: str
    color: tuple[int, int, int]
    duration_ms: int
    opacity: float
    created: float


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
        if not isinstance(port, int) or not (0 < port <= 65535):
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


class RTSPViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Detector de Manchas")
        self.root.configure(bg=UI_BG)
        self.root.minsize(960, 600)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._configure_fonts()
        self._configure_styles()

        self.status_var = tk.StringVar(value="Detenido")
        self.transport_var = tk.StringVar(value="TCP")
        self.rtsp_url_var = tk.StringVar(value=RTSP_STREAM_URL)
        self.rtsp_url_var.trace_add("write", lambda *_: self._sync_preset_to_url())
        self._transport_info_var = tk.StringVar(value="Transporte: TCP")
        self._fps_var = tk.StringVar(value="FPS: --")
        self._age_var = tk.StringVar(value="Ultimo frame: --")
        self._res_var = tk.StringVar(value="Resolucion: --")

        self._brightness_var = tk.IntVar(value=DEFAULT_BRIGHTNESS)
        self._contrast_var = tk.IntVar(value=DEFAULT_CONTRAST)
        self._brightness_label_var = tk.StringVar(value="")
        self._contrast_label_var = tk.StringVar(value="")
        self._brightness_scale: ttk.Scale | None = None
        self._contrast_scale: ttk.Scale | None = None

        self._settings_path = SETTINGS_FILE
        self._presets: list[dict[str, str]] = []
        self._last_url: str | None = None
        self._preset_var = tk.StringVar(value="")
        self._preset_combo: ttk.Combobox | None = None

        self._status_level = "idle"
        self._status_colors = dict(STATUS_COLORS)
        self._status_canvas: tk.Canvas | None = None
        self._status_dot: int | None = None
        self._details_frame: ttk.Frame | None = None
        self._details_window: tk.Toplevel | None = None
        self._details_visible = False
        self._fullscreen = False
        self._geometry_locked = False
        self._placeholder_key: tuple[int, int, str, str] | None = None
        self._placeholder_img: Image.Image | None = None
        self._placeholder_photo_key: tuple[int, int, str, str] | None = None
        self._placeholder_photo: ImageTk.PhotoImage | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self._fps_counter = 0
        self._fps_last_ts = time.monotonic()
        self._last_age_update = 0.0
        self._last_res: str | None = None

        # Estado del stream
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_ts = 0.0
        self._last_display_ts = 0.0

        # Heartbeat + overlays
        self._hb_bridge: _HeartbeatBridge | None = None
        self._hb_cycle = cycle(_HEARTBEAT_ALPHABET)
        self._hb_job: str | None = None
        self._hb_local_box: tk.Text | None = None
        self._hb_remote_box: tk.Text | None = None

        self._overlay_messages: list[_OverlayMessage] = []

        self.root.bind("<F11>", lambda _: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda _: self._exit_fullscreen())
        self._locked_geometry_size: tuple[int, int] | None = None
        self._ignore_configure = False
        self.root.bind("<Configure>", self._on_root_configure)

        self._load_settings()
        if self._last_url:
            self.rtsp_url_var.set(self._last_url)

        self._build_layout()
        self._sync_adjust_labels()
        self._refresh_presets()
        self._set_status("Detenido", "idle")
        self._init_heartbeat()
        self.root.after(200, self._lock_geometry)
        self._schedule_refresh()

    # UI
    def _configure_fonts(self) -> None:
        families = set(tkfont.families(self.root))

        def _pick(candidates: list[str], fallback: str) -> str:
            for name in candidates:
                if name in families:
                    return name
            return fallback

        base_fallback = tkfont.nametofont("TkDefaultFont").actual("family")
        mono_fallback = tkfont.nametofont("TkFixedFont").actual("family")

        base_family = _pick(
            ["Segoe UI", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Helvetica"],
            base_fallback,
        )
        mono_family = _pick(
            ["Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Liberation Mono"],
            mono_fallback,
        )

        tkfont.nametofont("TkDefaultFont").configure(family=base_family, size=9)
        tkfont.nametofont("TkTextFont").configure(family=base_family, size=9)
        tkfont.nametofont("TkHeadingFont").configure(family=base_family, size=10, weight="bold")
        tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=9)

        self._title_font = tkfont.Font(family=base_family, size=14, weight="bold")
        self._subtitle_font = tkfont.Font(family=base_family, size=9)
        self._mono_font = tkfont.Font(family=mono_family, size=9)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Main.TFrame", background=UI_BG)
        style.configure("Card.TFrame", background=UI_PANEL_BG, borderwidth=1, relief="solid")
        style.configure("Header.TFrame", background=UI_HEADER_BG)
        style.configure("TLabel", background=UI_BG, foreground=UI_TEXT)
        style.configure("Muted.TLabel", background=UI_BG, foreground=UI_MUTED)
        style.configure("HeaderTitle.TLabel", background=UI_HEADER_BG, foreground=UI_HEADER_FG, font=self._title_font)
        style.configure("HeaderSub.TLabel", background=UI_HEADER_BG, foreground=UI_HEADER_FG, font=self._subtitle_font)
        style.configure("HeaderStatus.TLabel", background=UI_HEADER_BG, foreground=UI_HEADER_FG, font=self._subtitle_font)
        style.configure("Stats.TFrame", background=UI_BG)
        style.configure("Stats.TLabel", background=UI_BG, foreground=UI_TEXT)
        style.configure("Primary.TButton", background=UI_ACCENT, foreground="#ffffff", padding=(8, 3), borderwidth=0)
        style.map("Primary.TButton", background=[("active", UI_ACCENT_DARK), ("pressed", UI_ACCENT_DARK)])
        style.configure("TButton", padding=(6, 3))
        style.configure("TEntry", fieldbackground=UI_PANEL_BG, background=UI_PANEL_BG)
        style.configure("TCombobox", fieldbackground=UI_PANEL_BG, background=UI_PANEL_BG)
        style.configure("TNotebook", background=UI_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 6), background=UI_PANEL_BG)
        style.map("TNotebook.Tab", background=[("selected", UI_BG)])

    def _load_settings(self) -> None:
        self._presets = []
        self._last_url = None

        data = None
        if os.path.exists(self._settings_path):
            try:
                with open(self._settings_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                data = None

        if isinstance(data, dict):
            presets = data.get("presets")
            if isinstance(presets, list):
                for item in presets:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "")).strip()
                    url = str(item.get("url", "")).strip()
                    if name and url:
                        self._presets.append({"name": name, "url": url})

            last_url = data.get("last_url")
            if isinstance(last_url, str) and last_url.strip():
                self._last_url = last_url.strip()

        if not self._presets:
            self._presets.append({"name": "Detector", "url": RTSP_STREAM_URL})

        if not self._last_url:
            self._last_url = self._presets[0]["url"]

    def _save_settings(self) -> None:
        payload = {"presets": self._presets, "last_url": self._last_url}
        try:
            with open(self._settings_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
        except Exception as exc:
            LOGGER.warning("No se pudo guardar settings: %s", exc)

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, style="Main.TFrame", padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main, style="Header.TFrame", padding=(10, 5))
        header.pack(fill=tk.X, pady=(0, 8))

        title_block = ttk.Frame(header, style="Header.TFrame")
        title_block.pack(side=tk.LEFT, anchor="w")
        title_line = ttk.Frame(title_block, style="Header.TFrame")
        title_line.pack(anchor="w")
        ttk.Label(title_line, text="Detector de Manchas", style="HeaderTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_line, text="Vista en tiempo real", style="HeaderSub.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        status_block = ttk.Frame(header, style="Header.TFrame")
        status_block.pack(side=tk.RIGHT, anchor="e")
        self._status_canvas = tk.Canvas(
            status_block, width=12, height=12, highlightthickness=0, bg=UI_HEADER_BG
        )
        self._status_dot = self._status_canvas.create_oval(2, 2, 10, 10, fill=STATUS_COLORS["idle"], outline="")
        self._status_canvas.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(status_block, textvariable=self.status_var, style="HeaderStatus.TLabel").pack(side=tk.LEFT)

        toolbar = ttk.Frame(main, style="Main.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 8))
        toolbar.columnconfigure(0, weight=1)
        toolbar.columnconfigure(1, weight=1)

        toolbar_left = ttk.Frame(toolbar, style="Main.TFrame")
        toolbar_left.grid(row=0, column=0, sticky="w")
        toolbar_right = ttk.Frame(toolbar, style="Main.TFrame")
        toolbar_right.grid(row=0, column=1, sticky="e")
        toolbar_left_bottom = ttk.Frame(toolbar, style="Main.TFrame")
        toolbar_left_bottom.grid(row=1, column=0, sticky="we", pady=(6, 0))
        toolbar_left_bottom.columnconfigure(1, weight=1)
        toolbar_right_bottom = ttk.Frame(toolbar, style="Main.TFrame")
        toolbar_right_bottom.grid(row=1, column=1, sticky="e", pady=(6, 0))

        self.btn_start = ttk.Button(toolbar_left, text="Iniciar", command=self._toggle_stream, style="Primary.TButton")
        self.btn_start.pack(side=tk.LEFT)

        self.btn_fullscreen = ttk.Button(toolbar_left, text="Pantalla completa", command=self._toggle_fullscreen)
        self.btn_fullscreen.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_details = ttk.Button(toolbar_left, text="Detalles", command=self._toggle_details)
        self.btn_details.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(toolbar_right, text="Transporte:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        combo_transport = ttk.Combobox(
            toolbar_right, textvariable=self.transport_var, values=("TCP", "UDP"), state="readonly", width=6
        )
        combo_transport.pack(side=tk.LEFT)
        combo_transport.bind("<<ComboboxSelected>>", lambda *_: self._on_transport_change())

        ttk.Label(toolbar_right, text="Presets:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        self._preset_combo = ttk.Combobox(
            toolbar_right,
            textvariable=self._preset_var,
            state="readonly",
            width=22,
        )
        self._preset_combo.pack(side=tk.LEFT)
        self._preset_combo.bind("<<ComboboxSelected>>", lambda *_: self._apply_preset())
        ttk.Button(toolbar_right, text="Guardar", command=self._save_current_preset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar_right, text="Eliminar", command=self._delete_preset).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(toolbar_left_bottom, text="RTSP URL:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar_left_bottom, textvariable=self.rtsp_url_var).grid(
            row=0, column=1, sticky="we", padx=(6, 0)
        )

        ttk.Label(toolbar_right_bottom, text="Brillo:", style="Muted.TLabel").pack(side=tk.LEFT)
        self._brightness_scale = ttk.Scale(
            toolbar_right_bottom,
            from_=BRIGHTNESS_MIN,
            to=BRIGHTNESS_MAX,
            command=self._on_brightness_change,
            length=150,
        )
        self._brightness_scale.pack(side=tk.LEFT, padx=(6, 6))
        ttk.Label(toolbar_right_bottom, textvariable=self._brightness_label_var, style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar_right_bottom, text="Reset", command=self._reset_brightness).pack(side=tk.LEFT)

        ttk.Label(toolbar_right_bottom, text="Contraste:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 0))
        self._contrast_scale = ttk.Scale(
            toolbar_right_bottom,
            from_=CONTRAST_MIN,
            to=CONTRAST_MAX,
            command=self._on_contrast_change,
            length=150,
        )
        self._contrast_scale.pack(side=tk.LEFT, padx=(6, 6))
        ttk.Label(toolbar_right_bottom, textvariable=self._contrast_label_var, style="Muted.TLabel").pack(
            side=tk.LEFT
        )

        video_card = ttk.Frame(main, style="Card.TFrame", padding=6)
        video_card.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(
            video_card,
            bg="#101114",
            bd=0,
            highlightthickness=0,
            anchor="center",
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        stats_frame = ttk.Frame(main, style="Stats.TFrame")
        stats_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(stats_frame, textvariable=self._fps_var, style="Stats.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats_frame, textvariable=self._age_var, style="Stats.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats_frame, textvariable=self._res_var, style="Stats.TLabel").pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(stats_frame, textvariable=self._transport_info_var, style="Stats.TLabel").pack(side=tk.LEFT)

        self._build_details_window()

    def _build_details_window(self) -> None:
        details = tk.Toplevel(self.root)
        details.title("Detalles - Detector de Manchas")
        details.configure(bg=UI_BG)
        details.geometry("760x320")
        details.minsize(620, 240)
        details.protocol("WM_DELETE_WINDOW", self._hide_details_window)
        details.withdraw()

        self._details_window = details
        self._details_frame = ttk.Frame(details, style="Main.TFrame", padding=8)
        self._details_frame.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(self._details_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        logs_frame = ttk.Frame(notebook, style="Main.TFrame")
        notebook.add(logs_frame, text="Logs")
        self.log_text = tk.Text(
            logs_frame,
            height=8,
            wrap="word",
            state="disabled",
            bg=UI_PANEL_BG,
            fg=UI_TEXT,
            relief="flat",
            font=self._mono_font,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        hb_frame = ttk.Frame(notebook, style="Main.TFrame", padding=8)
        notebook.add(hb_frame, text="Heartbeat")
        hb_frame.columnconfigure(1, weight=1)

        ttk.Label(hb_frame, text="Enviado (visor):", style="Muted.TLabel").grid(row=0, column=0, sticky="nw", padx=6)
        self._hb_local_box = tk.Text(
            hb_frame,
            height=2,
            width=48,
            wrap="word",
            state="disabled",
            bg=UI_PANEL_BG,
            fg=UI_TEXT,
            relief="flat",
            font=self._mono_font,
        )
        self._hb_local_box.grid(row=0, column=1, sticky="we", padx=(0, 6), pady=(0, 6))

        ttk.Label(hb_frame, text="Recibido (detector):", style="Muted.TLabel").grid(
            row=1, column=0, sticky="nw", padx=6
        )
        self._hb_remote_box = tk.Text(
            hb_frame,
            height=2,
            width=48,
            wrap="word",
            state="disabled",
            bg=UI_PANEL_BG,
            fg=UI_TEXT,
            relief="flat",
            font=self._mono_font,
        )
        self._hb_remote_box.grid(row=1, column=1, sticky="we", padx=(0, 6), pady=(0, 6))

        ttk.Label(
            hb_frame,
            text="Se envia un mensaje cada 5s para comprobar conectividad.",
            style="Muted.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6)

    def _refresh_presets(self) -> None:
        if self._preset_combo is None:
            return
        names = [p["name"] for p in self._presets]
        self._preset_combo.configure(values=names)
        if not names:
            self._preset_var.set("")
            return

        current = self._preset_var.get()
        if current in names:
            return

        url = (self.rtsp_url_var.get() or "").strip()
        for item in self._presets:
            if item["url"] == url:
                self._preset_var.set(item["name"])
                return

        self._preset_var.set("")

    def _sync_preset_to_url(self) -> None:
        if self._preset_combo is None:
            return
        url = (self.rtsp_url_var.get() or "").strip()
        for item in self._presets:
            if item["url"] == url:
                self._preset_var.set(item["name"])
                return
        self._preset_var.set("")

    def _apply_preset(self) -> None:
        name = (self._preset_var.get() or "").strip()
        if not name:
            return
        preset = next((p for p in self._presets if p["name"] == name), None)
        if not preset:
            return
        self.rtsp_url_var.set(preset["url"])
        self._last_url = preset["url"]
        self._save_settings()
        self._log(f"Preset seleccionado: {name}", logging.INFO)

    def _generate_preset_name(self, url: str) -> str:
        parsed = urlparse(url)
        host = parsed.hostname or "Stream"
        path = parsed.path.strip("/")
        suffix = path.split("/")[-1] if path else ""

        base = host
        if suffix and suffix.lower() not in {host.lower(), ""}:
            base = f"{host}-{suffix}"
        base = base.replace("_", "-")

        existing = {p["name"] for p in self._presets}
        if base not in existing:
            return base

        idx = 2
        while f"{base} ({idx})" in existing:
            idx += 1
        return f"{base} ({idx})"

    def _save_current_preset(self) -> None:
        url = (self.rtsp_url_var.get() or "").strip()
        if not url:
            messagebox.showwarning("RTSP", "Introduce una URL RTSP valida.")
            return

        for item in self._presets:
            if item["url"] == url:
                self._preset_var.set(item["name"])
                self._save_settings()
                self._log(f"Preset ya guardado: {item['name']}", logging.INFO)
                return

        name = self._generate_preset_name(url)
        self._presets.append({"name": name, "url": url})
        self._preset_var.set(name)
        self._last_url = url
        self._refresh_presets()
        self._save_settings()
        self._log(f"Preset guardado: {name}", logging.INFO)

    def _delete_preset(self) -> None:
        name = (self._preset_var.get() or "").strip()
        if not name:
            return
        confirm = messagebox.askyesno("Presets", f"Eliminar preset '{name}'?")
        if not confirm:
            return

        self._presets = [p for p in self._presets if p["name"] != name]
        if not self._presets:
            self._presets.append({"name": "Detector", "url": RTSP_STREAM_URL})
        self._refresh_presets()
        self._last_url = (self.rtsp_url_var.get() or "").strip()
        self._save_settings()
        self._log(f"Preset eliminado: {name}", logging.INFO)

    def _sync_adjust_labels(self) -> None:
        self._brightness_label_var.set(self._format_brightness(self._brightness_var.get()))
        self._contrast_label_var.set(f"{self._contrast_var.get()}%")
        if self._brightness_scale is not None:
            self._brightness_scale.set(self._brightness_var.get())
        if self._contrast_scale is not None:
            self._contrast_scale.set(self._contrast_var.get())

    def _format_brightness(self, value: int) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value}"

    def _on_brightness_change(self, value: str) -> None:
        snapped = int(round(float(value) / BRIGHTNESS_STEP) * BRIGHTNESS_STEP)
        snapped = max(BRIGHTNESS_MIN, min(BRIGHTNESS_MAX, snapped))
        if self._brightness_var.get() != snapped:
            self._brightness_var.set(snapped)
        if self._brightness_scale is not None and abs(float(value) - snapped) > 0.1:
            self._brightness_scale.set(snapped)
        self._brightness_label_var.set(self._format_brightness(snapped))

    def _on_contrast_change(self, value: str) -> None:
        snapped = int(round(float(value) / CONTRAST_STEP) * CONTRAST_STEP)
        snapped = max(CONTRAST_MIN, min(CONTRAST_MAX, snapped))
        if self._contrast_var.get() != snapped:
            self._contrast_var.set(snapped)
        if self._contrast_scale is not None and abs(float(value) - snapped) > 0.1:
            self._contrast_scale.set(snapped)
        self._contrast_label_var.set(f"{snapped}%")

    def _reset_brightness(self) -> None:
        self._brightness_var.set(DEFAULT_BRIGHTNESS)
        self._brightness_label_var.set(self._format_brightness(DEFAULT_BRIGHTNESS))
        if self._brightness_scale is not None:
            self._brightness_scale.set(DEFAULT_BRIGHTNESS)

    def _apply_adjustments(self, bgr: np.ndarray) -> np.ndarray:
        brightness = self._brightness_var.get()
        contrast = self._contrast_var.get() / 100.0
        if brightness == DEFAULT_BRIGHTNESS and abs(contrast - 1.0) < 0.001:
            return bgr
        return cv2.convertScaleAbs(bgr, alpha=contrast, beta=brightness)

    def _lock_geometry(self) -> None:
        if self._geometry_locked:
            return
        try:
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            if width > 0 and height > 0:
                self._locked_geometry_size = (width, height)
                self._geometry_locked = True
        except tk.TclError:
            pass

    def _on_root_configure(self, event: tk.Event) -> None:
        if not self._geometry_locked:
            return
        if self._fullscreen:
            return
        if self.root.state() == "zoomed":
            return
        target = self._locked_geometry_size
        if target is None:
            return
        current = (event.width, event.height)
        if current == target:
            return
        if self._ignore_configure:
            return
        self._ignore_configure = True
        try:
            self.root.geometry(f"{target[0]}x{target[1]}")
        except tk.TclError:
            pass
        finally:
            self._ignore_configure = False

    def _toggle_details(self) -> None:
        if self._details_window is None:
            self._build_details_window()
        if self._details_visible:
            self._hide_details_window()
            return
        try:
            self._details_window.deiconify()
            self._details_window.lift()
            self._details_window.focus_force()
        except tk.TclError:
            pass
        self._details_visible = True
        self.btn_details.configure(text="Ocultar detalles")

    def _hide_details_window(self) -> None:
        if self._details_window is None:
            return
        try:
            self._details_window.withdraw()
        except tk.TclError:
            pass
        self._details_visible = False
        self.btn_details.configure(text="Detalles")

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        try:
            self.root.attributes("-fullscreen", self._fullscreen)
        except tk.TclError:
            pass
        if self._fullscreen:
            self.btn_fullscreen.configure(text="Salir pantalla completa")
        else:
            self.btn_fullscreen.configure(text="Pantalla completa")
            self._enforce_locked_geometry_later()

    def _exit_fullscreen(self) -> None:
        if not self._fullscreen:
            return
        self._fullscreen = False
        try:
            self.root.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        self.btn_fullscreen.configure(text="Pantalla completa")
        self._enforce_locked_geometry_later()

    def _enforce_locked_geometry_later(self) -> None:
        self.root.after(50, self._enforce_locked_geometry)

    def _enforce_locked_geometry(self) -> None:
        if not self._geometry_locked or self._fullscreen or self._locked_geometry_size is None:
            return
        if self._ignore_configure:
            return
        self._ignore_configure = True
        try:
            self.root.geometry(f"{self._locked_geometry_size[0]}x{self._locked_geometry_size[1]}")
        except tk.TclError:
            pass
        finally:
            self._ignore_configure = False

    def _set_status(self, text: str, level: str) -> None:
        def _apply() -> None:
            self.status_var.set(text)
            self._status_level = level
            if self._status_canvas is None or self._status_dot is None:
                return
            color = self._status_colors.get(level, self._status_colors["idle"])
            try:
                self._status_canvas.itemconfigure(self._status_dot, fill=color)
            except tk.TclError:
                pass

        try:
            self.root.after(0, _apply)
        except tk.TclError:
            pass

    def _trim_log_lines(self) -> None:
        try:
            line_count = int(self.log_text.index("end-1c").split(".")[0])
        except Exception:
            return
        excess = line_count - MAX_LOG_LINES
        if excess > 0:
            try:
                self.log_text.delete("1.0", f"{excess + 1}.0")
            except tk.TclError:
                pass

    def _record_display_stats(self, frame_shape: tuple[int, int]) -> None:
        self._fps_counter += 1
        now = time.monotonic()
        if now - self._fps_last_ts >= 1.0:
            fps = self._fps_counter / max(0.1, now - self._fps_last_ts)
            self._fps_var.set(f"FPS: {fps:.1f}")
            self._fps_counter = 0
            self._fps_last_ts = now

        h, w = frame_shape[:2]
        res = f"{w}x{h}"
        if res != self._last_res:
            self._res_var.set(f"Resolucion: {res}")
            self._last_res = res

    def _update_frame_age(self, ts: float) -> None:
        now = time.monotonic()
        if now - self._last_age_update < 0.5:
            return
        self._last_age_update = now
        if ts > 0:
            age = max(0.0, now - ts)
            self._age_var.set(f"Ultimo frame: {age:.1f}s")
        else:
            self._age_var.set("Ultimo frame: --")

    def _placeholder_message(self, has_frame: bool, is_running: bool) -> tuple[str, str]:
        if not is_running:
            return ("DETENIDO", "Pulsa Iniciar para conectar")
        if not has_frame:
            return ("CONECTANDO", "Esperando stream...")
        return ("SIN SENAL", "Reintentando conexion...")

    def _render_placeholder(self, width: int, height: int, title: str, subtitle: str) -> Image.Image:
        key = (width, height, title, subtitle)
        if self._placeholder_key == key and self._placeholder_img is not None:
            return self._placeholder_img

        img = Image.new("RGB", (width, height), PLACEHOLDER_BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, width, 4), fill=UI_ACCENT)

        title = (title or "").strip().upper()
        subtitle = (subtitle or "").strip()

        title_box = draw.textbbox((0, 0), title, font=_PLACEHOLDER_FONT)
        title_w = title_box[2] - title_box[0]
        title_h = title_box[3] - title_box[1]

        sub_box = draw.textbbox((0, 0), subtitle, font=_PLACEHOLDER_SUB_FONT) if subtitle else (0, 0, 0, 0)
        sub_w = sub_box[2] - sub_box[0]
        sub_h = sub_box[3] - sub_box[1]

        total_h = title_h + (sub_h + 6 if subtitle else 0)
        base_y = max(10, (height - total_h) // 2)
        title_x = max(10, (width - title_w) // 2)
        draw.text((title_x, base_y), title, font=_PLACEHOLDER_FONT, fill=PLACEHOLDER_FG)

        if subtitle:
            sub_x = max(10, (width - sub_w) // 2)
            draw.text((sub_x, base_y + title_h + 6), subtitle, font=_PLACEHOLDER_SUB_FONT, fill=PLACEHOLDER_MUTED)

        self._placeholder_key = key
        self._placeholder_img = img
        return img

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

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        try:
            LOGGER.log(level, msg)
        except Exception:
            print(msg, flush=True)

        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}\n"

        def _append() -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, line)
            self._trim_log_lines()
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")

        self.root.after(0, _append)

    # Stream control
    def _toggle_stream(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_stream()
        else:
            url = (self.rtsp_url_var.get() or "").strip()
            if not url:
                messagebox.showwarning("RTSP", "Introduce una URL RTSP valida.")
                return
            self._start_stream(url)

    def _on_transport_change(self) -> None:
        t = (self.transport_var.get() or "TCP").strip().upper()
        if t not in {"TCP", "UDP"}:
            t = "TCP"
            self.transport_var.set(t)
        _set_ffmpeg_transport(t)
        self._transport_info_var.set(f"Transporte: {t}")

        current_url = (self.rtsp_url_var.get() or "").strip()
        updated = _normalize_rtsp_url(current_url, t)
        if updated != current_url:
            self.rtsp_url_var.set(updated)

    def _start_stream(self, url: str) -> None:
        self._on_transport_change()

        self._stop_event.clear()
        self._latest_frame = None
        self._latest_frame_ts = 0.0
        self._last_display_ts = 0.0
        self._fps_counter = 0
        self._fps_last_ts = time.monotonic()
        self._last_res = None
        self._fps_var.set("FPS: --")
        self._age_var.set("Ultimo frame: --")
        self._res_var.set("Resolucion: --")
        self._last_url = url
        self._save_settings()

        self.btn_start.config(text="Detener")
        self._set_status("Conectando...", "connecting")
        self._log(f"Intentando conectar a {url}...", logging.INFO)

        self._refresh_heartbeat_targets(url)

        self._thread = threading.Thread(target=self._video_loop, args=(url,), daemon=True)
        self._thread.start()

    def _stop_stream(self) -> None:
        self._stop_event.set()
        self._set_status("Deteniendo...", "warning")
        self.btn_start.config(text="Iniciar")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._fps_var.set("FPS: --")
        self._age_var.set("Ultimo frame: --")
        self._res_var.set("Resolucion: --")
        self._last_res = None

        self._set_status("Detenido", "idle")
        self._log("Stream detenido.", logging.INFO)

    def _video_loop(self, base_url: str) -> None:
        if IS_POSIX:
            try:
                cv2.setNumThreads(1)
            except Exception:
                pass

        backoff = 1.0
        avisado = False

        while not self._stop_event.is_set():
            t = (self.transport_var.get() or "TCP").strip().upper()
            _set_ffmpeg_transport(t)

            url = _normalize_rtsp_url(base_url, t)
            self._set_status("Conectando...", "connecting")
            LOGGER.info("Intentando abrir RTSP con OpenCV: %s", url)

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                if not avisado:
                    self._log(f"No se pudo abrir el stream ({url}). Reintentando...", logging.WARNING)
                    avisado = True
                self._set_status("Reintentando...", "warning")
                time.sleep(backoff)
                backoff = min(backoff * 1.5, RECONNECT_BACKOFF_MAX)
                continue

            avisado = False
            backoff = 1.0
            self._cap = cap
            try:
                backend = getattr(cap, "getBackendName", lambda: "ffmpeg")()
            except Exception:
                backend = "ffmpeg"

            self._log(f"Stream abierto con backend {backend}.", logging.INFO)
            self._set_status("Transmitiendo", "live")

            # Mantiene baja latencia: el hilo siempre sobrescribe _latest_frame con el frame mas nuevo.
            # El refresco de UI se encarga de "saltar" frames viejos si no puede pintar todos.
            # Cuando grab_n=0 usamos cap.read() para evitar retrieve() sin grab().
            grab_n = 0

            while not self._stop_event.is_set():
                if grab_n <= 0:
                    ok, frame = cap.read()
                else:
                    # Descarta algunos frames para reducir latencia
                    for _ in range(grab_n):
                        if not cap.grab():
                            break
                    ok, frame = cap.retrieve()
                if not ok or frame is None:
                    self._log("Sin senal RTSP. Reintentando...", logging.WARNING)
                    self._set_status("Sin senal", "warning")
                    break

                if IS_POSIX:
                    # Baja resolucion para pintar mas ligero
                    try:
                        h, w = frame.shape[:2]
                        if w > MAX_FRAME_WIDTH_POSIX:
                            scale = MAX_FRAME_WIDTH_POSIX / float(w)
                            frame = cv2.resize(
                                frame,
                                (MAX_FRAME_WIDTH_POSIX, max(1, int(h * scale))),
                                interpolation=cv2.INTER_AREA,
                            )
                    except Exception:
                        pass

                ts = time.monotonic()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_ts = ts

            try:
                cap.release()
            except Exception:
                pass
            if self._cap is cap:
                self._cap = None

        LOGGER.info("Loop de video finalizado.")

    # Display refresh
    def _schedule_refresh(self) -> None:
        target_fps = DISPLAY_FPS_POSIX if IS_POSIX else DISPLAY_FPS_WINDOWS
        target_interval = 1.0 / max(1, target_fps)
        start_ts = time.monotonic()

        try:
            frame = None
            ts = 0.0
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    ts = self._latest_frame_ts

            now = time.monotonic()
            label_w = max(200, self.video_label.winfo_width() or 200)
            label_h = max(200, self.video_label.winfo_height() or 200)

            if frame is not None and ts > self._last_display_ts:
                self._last_display_ts = ts
                img = self._render_for_display(frame, label_w, label_h)
                self._photo = ImageTk.PhotoImage(img)
                self.video_label.configure(image=self._photo)
                self._record_display_stats(frame.shape)
            elif frame is None or (now - ts) > STREAM_STALE_SEC:
                title, subtitle = self._placeholder_message(
                    frame is not None,
                    bool(self._thread and self._thread.is_alive()),
                )
                placeholder = self._render_placeholder(label_w, label_h, title, subtitle)
                if self._placeholder_photo is None or self._placeholder_photo_key != self._placeholder_key:
                    self._placeholder_photo = ImageTk.PhotoImage(placeholder)
                    self._placeholder_photo_key = self._placeholder_key
                self._photo = self._placeholder_photo
                self.video_label.configure(image=self._photo)
                if self._status_level == "live":
                    self._set_status("Sin senal", "warning")

            self._update_frame_age(ts)
        finally:
            elapsed = time.monotonic() - start_ts
            delay_ms = max(1, int((target_interval - elapsed) * 1000))
            self.root.after(delay_ms, self._schedule_refresh)

    def _render_for_display(self, bgr: np.ndarray, label_w: int, label_h: int) -> Image.Image:
        # Resize a la etiqueta, manteniendo aspecto, sin canvas extra
        h, w = bgr.shape[:2]
        scale = min(label_w / w, label_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        adjusted = self._apply_adjustments(resized)
        rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Overlays
        self._cleanup_overlay_messages()
        if self._overlay_messages:
            base = pil_img.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            y = 10
            for msg in self._overlay_messages:
                if not msg.text:
                    continue
                alpha = int(max(0.0, min(1.0, msg.opacity)) * 255)
                if alpha <= 0:
                    continue
                bbox = draw.textbbox((10, y), msg.text, font=_OVERLAY_FONT)
                pad = 6
                rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
                draw.rectangle(rect, fill=(0, 0, 0, max(60, int(alpha * 0.6))))
                draw.text((10, y), msg.text, font=_OVERLAY_FONT, fill=(*msg.color, alpha))
                y = rect[3] + 6
            pil_img = Image.alpha_composite(base, overlay).convert("RGB")

        return pil_img

    # Overlay management
    def _add_overlay_message(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text", "")).strip()
        if not text:
            return
        color = _hex_to_rgb(str(payload.get("color", "#ffbc00")))
        try:
            duration = max(500, int(payload.get("duration_ms", 4000) or 4000))
        except Exception:
            duration = 4000
        try:
            opacity = float(payload.get("opacity", 0.8) or 0.8)
        except Exception:
            opacity = 0.8
        opacity = max(0.0, min(1.0, opacity))
        self._overlay_messages.append(
            _OverlayMessage(text=text, color=color, duration_ms=duration, opacity=opacity, created=time.time())
        )

    def _cleanup_overlay_messages(self) -> None:
        if not self._overlay_messages:
            return
        now = time.time()
        self._overlay_messages = [m for m in self._overlay_messages if (now - m.created) * 1000 < m.duration_ms]

    # Heartbeat
    def _init_heartbeat(self) -> None:
        self._hb_bridge = _HeartbeatBridge(
            name="visor",
            listen_port=HEARTBEAT_VIEWER_PORT,
            tk_root=self.root,
            on_message=self._on_heartbeat_message,
        )
        ok = self._hb_bridge.start()
        if not ok:
            self._set_textarea(self._hb_local_box, "Heartbeat inactivo (puerto ocupado)")
            return

        self._refresh_heartbeat_targets(self.rtsp_url_var.get())
        self._schedule_heartbeat()

    def _refresh_heartbeat_targets(self, url: str | None) -> None:
        if self._hb_bridge is None:
            return
        bridge = self._hb_bridge
        bridge.add_target(("127.0.0.1", HEARTBEAT_DETECTOR_PORT))

        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            bridge.add_target((local_ip, HEARTBEAT_DETECTOR_PORT))
        except Exception:
            pass

        if url:
            try:
                host = urlparse(url).hostname
                if host and host not in {"127.0.0.1", "localhost"}:
                    bridge.add_target((host, HEARTBEAT_DETECTOR_PORT))
            except Exception:
                pass

    def _schedule_heartbeat(self) -> None:
        letter = next(self._hb_cycle)
        payload = self._heartbeat_payload()
        payload["beat"] = f"VIS-{letter}"
        payload["stamp"] = time.strftime("%H:%M:%S")

        text = json.dumps(payload, ensure_ascii=False)
        self._set_textarea(self._hb_local_box, text)

        if self._hb_bridge is not None:
            self._hb_bridge.send_message(text)

        self._hb_job = self.root.after(int(HEARTBEAT_INTERVAL_SEC * 1000), self._schedule_heartbeat)

    def _on_heartbeat_message(self, payload: dict, addr: tuple[str, int]) -> None:
        text = payload.get("text") or ""
        sender = payload.get("sender") or addr[0]

        try:
            ts = float(payload.get("ts", 0))
            stamp = time.strftime("%H:%M:%S", time.localtime(ts))
        except Exception:
            stamp = time.strftime("%H:%M:%S")

        display = f"{sender} @ {addr[0]}:{addr[1]} {stamp}\n{text}"
        self._set_textarea(self._hb_remote_box, display)

        try:
            info = json.loads(text)
        except Exception:
            info = None

        if isinstance(info, dict) and info.get("type") == "overlay_message":
            self._add_overlay_message(info.get("payload"))

    def _heartbeat_payload(self) -> dict:
        now = time.time()
        with self._frame_lock:
            last_frame_wall = self._latest_frame_ts
        since_frame = None
        if last_frame_wall:
            since_frame = max(0.0, time.monotonic() - last_frame_wall)

        return {
            "role": "viewer",
            "state": "transmitiendo" if (self._thread and self._thread.is_alive()) else "detenido",
            "since_last_frame": since_frame,
            "ts": now,
        }

    def _stop_heartbeat(self) -> None:
        if self._hb_job is not None:
            try:
                self.root.after_cancel(self._hb_job)
            except Exception:
                pass
            self._hb_job = None
        if self._hb_bridge is not None:
            try:
                self._hb_bridge.send_message("VIS-STOP")
            except Exception:
                pass
            self._hb_bridge.stop()
            self._hb_bridge = None

    # Close
    def _on_close(self) -> None:
        self._last_url = (self.rtsp_url_var.get() or "").strip()
        self._save_settings()
        self._stop_stream()
        self._stop_heartbeat()
        if self._details_window is not None:
            try:
                self._details_window.destroy()
            except tk.TclError:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = RTSPViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
