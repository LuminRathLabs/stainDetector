# -*- coding: utf-8 -*-
"""
Centralized tooltip management for Tkinter widgets.

Provides:
- TooltipManager: loads/saves tooltip text from JSON.
    - InfoIcon: reusable Tkinter label with tooltip + edit menu.
    - ToolTip: lightweight hover tooltip widget.
"""

from __future__ import annotations

import json
import logging
import threading
import tkinter as tk
import weakref
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

LOGGER = logging.getLogger("TooltipManager")

_DEFAULT_TOOLTIPS_PATH = Path(__file__).resolve().parent.parent / "config" / "tooltips.json"


class TooltipManager:
    _instance: Optional["TooltipManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TooltipManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._data: dict = {}
        self._path: Path = _DEFAULT_TOOLTIPS_PATH
        self._dirty = False
        self._callbacks: list[object] = []

    @classmethod
    def initialize(cls, path: str | Path | None = None) -> "TooltipManager":
        instance = cls()
        if path:
            instance._path = Path(path)
        instance._load()
        return instance

    @classmethod
    def get_instance(cls) -> "TooltipManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                LOGGER.warning("Tooltip file not found: %s. Creating empty store.", self._path)
                self._data = {
                    "_version": "1.0",
                    "_description": "Tooltip texts",
                }
                self._save()
                return

            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
                LOGGER.info("Tooltips loaded from %s", self._path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to load tooltips: %s", exc)
                self._data = {}

    def _save(self) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, ensure_ascii=False, indent=2)
                self._dirty = False
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to save tooltips: %s", exc)

    @staticmethod
    def _split_key(key: str) -> list[str]:
        parts = [part.strip() for part in key.split(".") if part.strip()]
        if not parts or any(part.startswith("_") for part in parts):
            return []
        return parts

    @classmethod
    def get(cls, key: str, default: str = "") -> str:
        instance = cls.get_instance()
        parts = cls._split_key(key)
        if not parts:
            return default
        current: object = instance._data
        for part in parts:
            if not isinstance(current, dict):
                return default
            current = current.get(part)
            if current is None:
                return default
        return str(current) if not isinstance(current, dict) else default

    @classmethod
    def set(cls, key: str, value: str) -> None:
        instance = cls.get_instance()
        parts = cls._split_key(key)
        if not parts:
            LOGGER.warning("Ignoring invalid tooltip key: %s", key)
            return
        current = instance._data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        instance._dirty = True
        instance._save()
        for callback in instance._iter_callbacks():
            try:
                callback(key, value)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Tooltip callback failed: %s", exc)

    @classmethod
    def has_text(cls, key: str) -> bool:
        return bool(cls.get(key, "").strip())

    def _wrap_callback(self, callback: Callable[[str, str], None]) -> object:
        try:
            if getattr(callback, "__self__", None) is not None:
                return weakref.WeakMethod(callback)
            return weakref.ref(callback)
        except TypeError:
            return callback

    def _iter_callbacks(self) -> list[Callable[[str, str], None]]:
        alive: list[object] = []
        result: list[Callable[[str, str], None]] = []
        for item in self._callbacks:
            if isinstance(item, weakref.ReferenceType):
                callback = item()
                if callback is None:
                    continue
                alive.append(item)
            else:
                callback = item
                alive.append(item)
            result.append(callback)
        self._callbacks = alive
        return result

    @classmethod
    def register_callback(cls, callback: Callable[[str, str], None]) -> None:
        instance = cls.get_instance()
        instance._callbacks.append(instance._wrap_callback(callback))

    @classmethod
    def unregister_callback(cls, callback: Callable[[str, str], None]) -> None:
        instance = cls.get_instance()
        remaining: list[object] = []
        for item in instance._callbacks:
            if isinstance(item, weakref.ReferenceType):
                resolved = item()
                if resolved is None or resolved == callback:
                    continue
            elif item == callback:
                continue
            remaining.append(item)
        instance._callbacks = remaining


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 800) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip_window: tk.Toplevel | None = None
        self.scheduled_id: str | None = None
        self.pinned = False

        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")

    def update_text(self, new_text: str) -> None:
        self.text = new_text

    def _on_enter(self, event: tk.Event | None = None) -> None:
        if self.pinned:
            return
        self._unschedule()
        self.scheduled_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, event: tk.Event | None = None) -> None:
        if self.pinned:
            return
        self._unschedule()
        self._hide()

    def _unschedule(self) -> None:
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = None

    def _show(self) -> None:
        if self.tip_window or not self.text:
            return
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#FFFFE1",
            foreground="#333333",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=10,
            pady=8,
            wraplength=350,
        )
        label.pack()

        tw.lift()
        tw.attributes("-topmost", True)

    def _hide(self) -> None:
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

    def show_now(self) -> None:
        self._unschedule()
        self._show()

    def hide_now(self) -> None:
        self._unschedule()
        self._hide()

    def set_pinned(self, pinned: bool) -> None:
        self.pinned = bool(pinned)
        if not self.pinned:
            self.hide_now()


class InfoIcon(tk.Canvas):
    COLOR_ACTIVE = "#2196F3"
    COLOR_EMPTY = "#AAAAAA"
    COLOR_HOVER = "#1976D2"
    BG_NORMAL = "#e3f2fd"
    BG_HOVER = "#bbdefb"
    BG_ACTIVE = "#90caf9"
    SIZE = 18

    def __init__(
        self,
        master: tk.Widget,
        key: str,
        *,
        show_empty: bool = True,
        editable: bool = True,
        **kwargs,
    ) -> None:
        self.key = key
        self.editable = editable
        self._text = TooltipManager.get(key, "")

        has_text = bool(self._text.strip())
        if not has_text and not show_empty:
            super().__init__(master, **kwargs)
            return

        foreground = self.COLOR_ACTIVE if has_text else self.COLOR_EMPTY

        bg = kwargs.pop("background", None)
        if bg is None:
            try:
                bg = master.cget("background")
            except tk.TclError:
                try:
                    bg = master.cget("bg")
                except tk.TclError:
                    bg = master.winfo_toplevel().cget("background")
        super().__init__(
            master,
            width=self.SIZE,
            height=self.SIZE,
            highlightthickness=0,
            bg=bg,
            bd=0,
            **kwargs,
        )

        self._circle_id = self.create_oval(
            2,
            2,
            self.SIZE - 3,
            self.SIZE - 3,
            outline="#1565C0",
            width=1,
            fill=self.BG_NORMAL,
            tags=("circle",),
        )
        self._text_id = self.create_text(
            self.SIZE / 2,
            self.SIZE / 2,
            text="i",
            font=("Segoe UI", 11, "bold"),
            fill=foreground,
            tags=("text",),
        )
        self._tooltip = ToolTip(self, self._text or "(Sin ayuda definida)")
        for tag in ("circle", "text", "all"):
            self.tag_bind(tag, "<ButtonPress-1>", self._on_click, add="+")
            self.tag_bind(tag, "<ButtonRelease-1>", self._on_release, add="+")

        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")

        if editable:
            self.bind("<Button-3>", self._on_right_click, add="+")
            self.bind("<Control-Button-1>", self._on_right_click, add="+")

        self.bind("<Double-Button-1>", self._on_double_click, add="+")

        TooltipManager.register_callback(self._on_tooltip_update)

        self._hovered = False
        self._pressed = False
        self._pinned = False
        self._single_click_after_id: str | None = None
        self._unbind_top_click_id: str | None = None
        self._unbind_top_escape_id: str | None = None
        self._update_appearance()

    def _on_enter(self, event: tk.Event | None = None) -> None:
        self._hovered = True
        self._update_appearance()

    def _on_leave(self, event: tk.Event | None = None) -> None:
        self._hovered = False
        self._update_appearance()

    def _on_click(self, event: tk.Event | None = None) -> None:
        if self._single_click_after_id is not None:
            try:
                self.after_cancel(self._single_click_after_id)
            except Exception:
                pass
        self._single_click_after_id = self.after(220, self._toggle_pin)

    def _on_release(self, event: tk.Event | None = None) -> None:
        return

    def _on_double_click(self, event: tk.Event | None = None) -> None:
        if self._single_click_after_id is not None:
            try:
                self.after_cancel(self._single_click_after_id)
            except Exception:
                pass
            self._single_click_after_id = None
        self._open_editor()

    def _toggle_pin(self) -> None:
        self._toggle_pin_impl()
        self._single_click_after_id = None

    def _toggle_pin_impl(self) -> None:
        if self._pinned:
            self._set_pinned(False)
            return
        self._set_pinned(True)

    def _set_pinned(self, pinned: bool) -> None:
        self._pinned = bool(pinned)
        self._tooltip.set_pinned(self._pinned)
        if self._pinned:
            self._pressed = True
            self._tooltip.show_now()
            self._install_close_handlers()
        else:
            self._pressed = False
            self._tooltip.hide_now()
            self._remove_close_handlers()
        self._update_appearance()

    def _install_close_handlers(self) -> None:
        top = self.winfo_toplevel()
        if self._unbind_top_click_id is None:
            self._unbind_top_click_id = top.bind("<Button-1>", self._on_any_click, add="+")
        if self._unbind_top_escape_id is None:
            self._unbind_top_escape_id = top.bind("<Escape>", self._on_escape, add="+")

    def _remove_close_handlers(self) -> None:
        top = self.winfo_toplevel()
        if self._unbind_top_click_id is not None:
            try:
                top.unbind("<Button-1>", self._unbind_top_click_id)
            except Exception:
                pass
            self._unbind_top_click_id = None
        if self._unbind_top_escape_id is not None:
            try:
                top.unbind("<Escape>", self._unbind_top_escape_id)
            except Exception:
                pass
            self._unbind_top_escape_id = None

    def _on_any_click(self, event: tk.Event) -> None:
        if not self._pinned:
            return
        if event.widget is self:
            return
        self._set_pinned(False)

    def _on_escape(self, _event: tk.Event) -> None:
        if self._pinned:
            self._set_pinned(False)

    def _on_right_click(self, event: tk.Event) -> None:
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Editar ayuda...", command=self._open_editor)
        if self._text.strip():
            menu.add_separator()
            menu.add_command(label="Borrar texto", command=self._clear_text)
        menu.add_separator()
        menu.add_command(label="Copiar clave", command=self._copy_key)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _open_editor(self) -> None:
        dialog = _TooltipEditorDialog(self.winfo_toplevel(), self.key, self._text)
        new_text = dialog.show()
        if new_text is not None:
            TooltipManager.set(self.key, new_text)
            self._text = new_text
            self._tooltip.update_text(new_text or "(Sin ayuda definida)")
            self._update_appearance()

    def _clear_text(self) -> None:
        TooltipManager.set(self.key, "")
        self._text = ""
        self._tooltip.update_text("(Sin ayuda definida)")
        self._update_appearance()

    def _copy_key(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.key)

    def _update_appearance(self) -> None:
        has_text = bool(self._text.strip())
        fg = self.COLOR_ACTIVE if has_text else self.COLOR_EMPTY
        if self._pressed:
            bg = self.BG_ACTIVE
        elif self._hovered:
            bg = self.BG_HOVER
        else:
            bg = self.BG_NORMAL
        cursor = "hand2" if has_text or self.editable else ""
        self.config(cursor=cursor)
        self.itemconfigure(self._circle_id, fill=bg)
        self.itemconfigure(self._text_id, fill=fg)

    def _on_tooltip_update(self, key: str, value: str) -> None:
        if key == self.key:
            self._text = value
            self._tooltip.update_text(value or "(Sin ayuda definida)")
            self._update_appearance()


class _TooltipEditorDialog:
    def __init__(self, master: tk.Misc, key: str, current_text: str) -> None:
        self.master = master
        self.key = key
        self.current_text = current_text
        self.result: str | None = None

        self.window = tk.Toplevel(master)
        self.window.title("Editar texto de ayuda")
        self.window.geometry("500x300")
        self.window.resizable(True, True)
        self.window.transient(master)
        self.window.grab_set()

        self._build_ui()
        self._center_window()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.window, padding=15)
        frame.grid(row=0, column=0, sticky="nsew")
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        key_label = ttk.Label(
            frame,
            text=f"Clave: {self.key}",
            font=("Consolas", 9),
            foreground="#666666",
        )
        key_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.text_area = tk.Text(
            frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            height=8,
        )
        self.text_area.grid(row=1, column=0, sticky="nsew")
        self.text_area.insert("1.0", self.current_text)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.text_area.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.text_area.configure(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="e", pady=(15, 0))

        ttk.Button(btn_frame, text="Cancelar", command=self._on_cancel).pack(side="right", padx=(5, 0))
        ttk.Button(btn_frame, text="Guardar", command=self._on_save).pack(side="right")

        self.window.bind("<Escape>", lambda e: self._on_cancel())
        self.window.bind("<Control-Return>", lambda e: self._on_save())

    def _center_window(self) -> None:
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"+{x}+{y}")

    def _on_save(self) -> None:
        self.result = self.text_area.get("1.0", "end-1c").strip()
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()

    def show(self) -> str | None:
        self.text_area.focus_set()
        self.window.wait_window()
        return self.result


def init_tooltips(path: str | Path | None = None) -> None:
    TooltipManager.initialize(path)
