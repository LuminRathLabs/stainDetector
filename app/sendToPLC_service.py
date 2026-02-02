# -*- coding: utf-8 -*-
"""Servicio y UI opcional para supervisar las órdenes hacia el PLC."""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import re
import sys

# Agrupar iconos en la barra de tareas de Windows
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MetalRollBand.ManchasGuida.App.v1")
    except Exception:
        pass

import queue
import random
import shutil
import string
import subprocess
import sys
import threading
import time
import traceback
import uuid
import plc_bit_writer
from collections import Counter, deque
from contextlib import suppress
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

try:
    import tkinter as tk
    from tkinter import messagebox, ttk, simpledialog
    try:
        from tkinter.scrolledtext import ScrolledText
    except Exception:  # noqa: BLE001
        ScrolledText = None  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    tk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    ScrolledText = None  # type: ignore[assignment]

try:
    from tooltip_manager import InfoIcon, init_tooltips, TooltipManager
except Exception:  # noqa: BLE001
    InfoIcon = None  # type: ignore[assignment]
    init_tooltips = None  # type: ignore[assignment]
    TooltipManager = None  # type: ignore[assignment]

try:
    import snap7
    from snap7.types import Areas
    import snap7.util as snap7_util
    SNAP7_IMPORT_ERROR: Exception | None = None
    SNAP7_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    SNAP7_AVAILABLE = False
    SNAP7_IMPORT_ERROR = exc
    snap7 = None  # type: ignore[assignment]
    Areas = None  # type: ignore[assignment]
    snap7_util = None  # type: ignore[assignment]

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - dependencia opcional
    FPDF = None  # type: ignore[assignment]

_RULES_IMPORT_ERROR: Exception | None = None
_RULES_IMPORT_TRACE: str | None = None

LOGGER = logging.getLogger("SendToPLC")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.WARNING)

LOGGER.propagate = False

CONDITION_KIND_VISION = "vision"
CONDITION_KIND_PLC_BIT = "plc_bit"
CONDITION_KIND_RULE = "rule"  # Meta-regla: disparo basado en otras reglas
AREA_UNIT_PX = "px"
AREA_UNIT_CM = "cm"
WINDOW_SHORT_SEC = 3.0
WINDOW_LONG_SEC = 30.0
SNAPSHOT_POLL_INTERVAL_SEC = 0.2
PLC_CONDITION_CACHE_TTL_SEC = 1.0
PLC_NUMERIC_CACHE_TTL_SEC = 0.3
PLC_CONDITION_POLL_INTERVAL_SEC = 0.1
PLC_NUMERIC_TYPES = ("BYTE", "WORD", "DWORD", "INT", "DINT", "REAL")
SNAPSHOT_SHORT_WINDOW_SEC = WINDOW_SHORT_SEC
SNAPSHOT_LONG_WINDOW_SEC = WINDOW_LONG_SEC
SNAPSHOT_DEFAULT_WRITE_INTERVAL_MS = 1500
SNAPSHOT_MIN_WRITE_INTERVAL_MS = 200
SNAPSHOT_DEFAULT_CLEAN_INTERVAL_SEC = 60.0
DEFAULT_TRIGGER_CLASS = ""
DEFAULT_RESUME_DELAY_SEC = 10
MANUAL_LEVELS = ("NORMAL", "SLOW1", "SLOW2", "SLOW3")

try:  # pylint: disable=ungrouped-imports
    from detect_manchas_rules import Rule, RuleEngine, RuleEvaluation, RuleEffects
    _ExternalRuleEngine = RuleEngine
    _EXTERNAL_RULE_ENGINE = True
except ImportError as exc:  # pragma: no cover - fallback
    _EXTERNAL_RULE_ENGINE = False
    _ExternalRuleEngine = None
    _RULES_IMPORT_ERROR = exc
    _RULES_IMPORT_TRACE = traceback.format_exc()

    @dataclass
    class RuleEffects:  # type: ignore[override]
        blocked_classes: set[str] = field(default_factory=set)
        muted_triggers: list[dict[str, object]] = field(default_factory=list)
        forced_level: Optional[str] = None
        resume_level: Optional[str] = None
        overlay_messages: list[dict[str, object]] = field(default_factory=list)
        snapshot_requests: list[dict[str, object]] = field(default_factory=list)
        plc_orders: list[dict[str, Any]] = field(default_factory=list)

    @dataclass
    class RuleEvaluation:  # type: ignore[override]
        effects: RuleEffects
        matches: list = field(default_factory=list)

    @dataclass
    class Rule:  # type: ignore[override]
        rule_id: str
        name: str = "Regla"
        enabled: bool = True
        priority: int = 0
        condition: dict[str, object] = field(default_factory=dict)
        actions: list[dict[str, object]] = field(default_factory=list)
        condition_tree: dict[str, object] | None = None

        @classmethod
        def from_dict(cls, payload: dict[str, object]) -> "Rule":
            condition_payload = _normalize_rule_condition(payload.get("condition"))
            tree_payload = _normalize_condition_tree(payload.get("condition_tree"))
            if tree_payload is None and condition_payload:
                tree_payload = _wrap_condition_as_tree(condition_payload)
            primary_condition = _extract_primary_condition(tree_payload) or condition_payload

            rule = cls(
                rule_id=str(payload.get("rule_id", uuid.uuid4().hex)),
                name=str(payload.get("name", "Regla")),
                enabled=bool(payload.get("enabled", True)),
                priority=int(payload.get("priority", 0)),
                condition=primary_condition or {},
                actions=[copy.deepcopy(a) for a in payload.get("actions", []) if isinstance(a, dict)],
            )
            rule.condition_tree = tree_payload
            return rule

        def to_dict(self) -> dict[str, object]:
            payload = {
                "rule_id": self.rule_id,
                "name": self.name,
                "enabled": self.enabled,
                "priority": self.priority,
                "condition": copy.deepcopy(self.condition),
                "actions": [copy.deepcopy(a) for a in self.actions],
            }
            tree = getattr(self, "condition_tree", None)
            if tree:
                payload["condition_tree"] = copy.deepcopy(tree)
            return payload



if _EXTERNAL_RULE_ENGINE and not all(
    hasattr(RuleEngine, attr) for attr in ("_matches", "_actions_to_effects")
):
    LOGGER.warning(
        "RuleEngine externo sin soporte de interfaz extendida; se usará motor interno con prioridades."
    )
    _EXTERNAL_RULE_ENGINE = False


def get_resource_path(relative_path: str, is_config: bool = False) -> str:
    """
    Obtiene la ruta absoluta de un recurso, compatible con PyInstaller.
    Para archivos de configuración o datos que deben ser editables (fuera del .exe),
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


# ---------------------------------------------------------------------------
# RuleFiringHistory: Historial de activaciones de reglas para meta-reglas
# ---------------------------------------------------------------------------

@dataclass
class RuleFiringRecord:
    """Registro individual de un disparo de regla."""
    rule_id: str
    timestamp: float  # time.monotonic()


class RuleFiringHistory:
    """
    Buffer circular que registra disparos de reglas con soporte para:
    - Ventana temporal configurable
    - Debounce por regla
    - Limpieza automática de registros antiguos
    """

    def __init__(self, max_window_sec: float = 600.0) -> None:
        self._records: deque[RuleFiringRecord] = deque()
        self._max_window = max_window_sec
        self._last_firing_by_rule: dict[str, float] = {}  # Para debounce
        self._cooldown_until: dict[str, float] = {}  # Para cooldown de meta-reglas
        self._lock = threading.Lock()

    def record_firing(self, rule_id: str, debounce_ms: int = 0) -> bool:
        """
        Registra un disparo de regla.
        
        Args:
            rule_id: ID de la regla que disparó
            debounce_ms: Tiempo mínimo entre disparos para contar (0 = sin debounce)
        
        Returns:
            True si se registró, False si fue ignorado por debounce
        """
        now = time.monotonic()
        with self._lock:
            # Verificar debounce
            if debounce_ms > 0:
                last = self._last_firing_by_rule.get(rule_id, 0.0)
                if (now - last) * 1000 < debounce_ms:
                    return False
            
            self._records.append(RuleFiringRecord(rule_id, now))
            self._last_firing_by_rule[rule_id] = now
            self._cleanup(now)
        return True

    def get_firing_count(self, rule_id: str, window_sec: float) -> int:
        """
        Cuenta disparos de una regla en los últimos window_sec segundos.
        
        Args:
            rule_id: ID de la regla a contar
            window_sec: Ventana de tiempo en segundos
            
        Returns:
            Número de disparos en la ventana
        """
        now = time.monotonic()
        cutoff = now - window_sec
        with self._lock:
            return sum(1 for r in self._records if r.rule_id == rule_id and r.timestamp >= cutoff)

    def get_firing_count_multi(self, rule_ids: list[str], window_sec: float) -> int:
        """
        Cuenta disparos de múltiples reglas combinadas en los últimos window_sec segundos.
        """
        now = time.monotonic()
        cutoff = now - window_sec
        rule_id_set = set(rule_ids)
        with self._lock:
            return sum(1 for r in self._records if r.rule_id in rule_id_set and r.timestamp >= cutoff)

    def set_cooldown(self, rule_id: str, cooldown_sec: float) -> None:
        """Establece un cooldown para una meta-regla."""
        if cooldown_sec <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._cooldown_until[rule_id] = now + cooldown_sec

    def is_in_cooldown(self, rule_id: str) -> bool:
        """Verifica si una meta-regla está en período de cooldown."""
        now = time.monotonic()
        with self._lock:
            until = self._cooldown_until.get(rule_id, 0.0)
            if now >= until:
                # Limpiar cooldown expirado
                self._cooldown_until.pop(rule_id, None)
                return False
            return True

    def _cleanup(self, now: float) -> None:
        """Limpia registros más antiguos que la ventana máxima."""
        cutoff = now - self._max_window
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()
        # Limpiar cooldowns expirados ocasionalmente
        if len(self._records) % 100 == 0:
            expired = [k for k, v in self._cooldown_until.items() if now >= v]
            for k in expired:
                del self._cooldown_until[k]


# Instancia global del historial de disparos
_rule_firing_history = RuleFiringHistory()


if not _EXTERNAL_RULE_ENGINE:

    class RuleEngine:  # type: ignore[override]
        def __init__(
            self,
            rules_path: Path | None = None,
            *,
            plc_condition_provider: Callable[[dict[str, object]], bool] | None = None,
            vision_diag_callback: Callable[[dict[str, object]], None] | None = None,
        ) -> None:
            if rules_path is None:
                # Ruta por defecto robusta para empaquetado
                rules_path = Path(get_resource_path(os.path.join("..", "config", "sendToPLC_config.json"), is_config=True))
            
            self.rules_path = rules_path
            self._rules: list[Rule] = []
            self._available_classes: set[str] = set()
            self._plc_condition_provider = plc_condition_provider
            self._vision_diag_callback = vision_diag_callback
            self._muted_rule_ids: set[str] = set()
            self._load_from_disk()

        def set_muted_rules(self, rule_ids: Iterable[str]) -> None:
            self._muted_rule_ids = {str(rid) for rid in rule_ids if rid}

        # ------------------------------------------------------------------
        def _load_from_disk(self) -> None:
            if self.rules_path is None or not self.rules_path.is_file():
                return
            try:
                with self.rules_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudieron cargar reglas desde disco: %s", exc)
                return
            if isinstance(data, list):
                try:
                    self._rules = [Rule.from_dict(item) for item in data if isinstance(item, dict)]
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("No se pudieron interpretar algunas reglas: %s", exc)

        # ------------------------------------------------------------------
        def save_to_disk(self) -> None:
            if self.rules_path is None:
                return
            try:
                self.rules_path.parent.mkdir(parents=True, exist_ok=True)
                with self.rules_path.open("w", encoding="utf-8") as fh:
                    json.dump([rule.to_dict() for rule in self._rules], fh, ensure_ascii=False, indent=2)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudieron guardar reglas: %s", exc)

        # ------------------------------------------------------------------
        def process(self, snapshot: SnapshotState | None) -> RuleEvaluation:
            enabled_rules = [r for r in self._rules if r.enabled]
            # Ordenar por prioridad descendente: las reglas más importantes se evalúan primero.
            # En _merge_effects, los valores escalares (como forced_level) ya establecidos en 'base' (alta prio)
            # tienen preferencia sobre los nuevos 'new' (baja prio).
            # Las listas (mensajes, acciones PLC) se concatenan.
            enabled_rules.sort(key=lambda r: r.priority, reverse=True)

            if not enabled_rules:
                return RuleEvaluation(effects=RuleEffects(), matches=[])

            matched_rules: list[Rule] = []
            # matched_no_effect: list[Rule] = [] 
            aggregated_effects = RuleEffects()
            
            # Flag para saber si hubo algún match real
            any_match = False

            for rule in enabled_rules:
                rule_id = str(getattr(rule, "rule_id", "") or "")
                
                # 1. Verificar si está muteada
                if rule_id in self._muted_rule_ids:
                    # LOGGER.debug("Regla %s ignorada por estar deshabilitada temporalmente", rule_id)
                    continue

                # 2. Verificar coincidencia (Condición)
                match_result = self._matches(rule, snapshot)
                if not match_result:
                    continue
                
                any_match = True

                # 3. Registrar disparo (Crucial para Meta-Reglas)
                # Se registra ANTES de verificar si tiene efectos, porque una regla 
                # podría existir solo para ser contada por otra regla (meta-regla).
                if rule_id:
                    _rule_firing_history.record_firing(rule_id)
                    cooldown_sec = self._get_rule_cooldown(rule)
                    if cooldown_sec > 0:
                        _rule_firing_history.set_cooldown(rule_id, cooldown_sec)

                # 4. Obtener efectos
                effects = self._actions_to_effects(rule.actions)
                has_payload = self._effects_has_payload(effects)

                # Logs a nivel debug solo si es relevante
                # LOGGER.info("Regla %s coincide. Efectos: %s", getattr(rule, "name", ""), has_payload)

                if has_payload:
                    aggregated_effects = self._merge_effects(aggregated_effects, effects)
                    matched_rules.append(rule)
                # else:
                #     matched_no_effect.append(rule)

            if any_match:
                # Retornamos la evaluación combinada de TODAS las reglas que coincidieron
                return RuleEvaluation(effects=aggregated_effects, matches=matched_rules)

            # LOGGER.debug("Ninguna regla coincidió con el snapshot actual")
            return RuleEvaluation(effects=RuleEffects(), matches=[])

        @staticmethod
        def _merge_effects(base: RuleEffects, new: RuleEffects) -> RuleEffects:
            return _merge_rule_effects(base, new)

        def get_rules(self) -> Sequence[Rule]:
            return tuple(self._rules)

        def update_rules(self, rules: Iterable[Rule]) -> None:
            self._rules = [copy.deepcopy(rule) for rule in rules]
            self.save_to_disk()

        def set_available_classes(self, classes: Iterable[str]) -> None:
            self._available_classes = {str(c).strip() for c in classes if c}

        def _get_rule_cooldown(self, rule: Rule) -> float:
            """
            Extrae el cooldown de una regla si contiene condiciones de tipo 'rule'.
            Retorna el cooldown máximo encontrado o 0.
            """
            tree = getattr(rule, "condition_tree", None)
            if not isinstance(tree, dict):
                return 0.0
            
            def extract_cooldown(node: dict[str, object]) -> float:
                max_cooldown = 0.0
                if node.get("type") == "group":
                    children = node.get("children") or []
                    for child in children:
                        if isinstance(child, dict):
                            max_cooldown = max(max_cooldown, extract_cooldown(child))
                elif node.get("condition"):
                    cond = node.get("condition")
                    if isinstance(cond, dict) and cond.get("kind") == CONDITION_KIND_RULE:
                        try:
                            max_cooldown = float(cond.get("cooldown_sec", 0) or 0)
                        except (TypeError, ValueError):
                            pass
                return max_cooldown
            
            return extract_cooldown(tree)

        def _matches(self, rule: Rule, snapshot: SnapshotState | None) -> bool:
            # Prioridad: condition_tree > condition
            tree = rule.condition_tree
            if isinstance(tree, dict) and tree.get("type") == "group":
                return self._matches_tree(tree, snapshot, rule)
            
            # Fallback a condición simple (legado)
            condition = rule.condition if isinstance(rule.condition, dict) else {}
            kind = str(condition.get("kind") or CONDITION_KIND_VISION).strip().lower()
            if kind == CONDITION_KIND_PLC_BIT:
                return self._matches_plc_bit(condition, rule)
            if kind == CONDITION_KIND_RULE:
                return self._matches_rule(condition, rule)
            return self._matches_vision(condition, snapshot, rule)

        def _matches_tree(self, node: dict[str, object], snapshot: SnapshotState | None, rule: Rule) -> bool:
            node_type = str(node.get("type", "")).lower()
            negated = bool(node.get("negated"))
            result = False

            if node_type == "group":
                operator = str(node.get("operator", "and")).lower()
                children = node.get("children")
                if not isinstance(children, list) or not children:
                    # Grupo vacío: ¿se considera true o false? 
                    # Asumimos True para AND (identidad) y False para OR.
                    # Pero si es el root y está vacío, mejor False (sin condiciones).
                    result = True if operator == "and" else False
                else:
                    if operator == "or":
                        result = False
                        for child in children:
                            if isinstance(child, dict) and self._matches_tree(child, snapshot, rule):
                                result = True
                                break
                    else:  # AND
                        result = True
                        for child in children:
                            if isinstance(child, dict) and not self._matches_tree(child, snapshot, rule):
                                result = False
                                break
            
            elif node_type == "condition" or node.get("condition"):
                cond_payload = node.get("condition")
                if isinstance(cond_payload, dict):
                    kind = str(cond_payload.get("kind") or CONDITION_KIND_VISION).strip().lower()
                    if kind == CONDITION_KIND_PLC_BIT:
                        result = self._matches_plc_bit(cond_payload, rule)
                    elif kind == CONDITION_KIND_RULE:
                        result = self._matches_rule(cond_payload, rule)
                    else:
                        result = self._matches_vision(cond_payload, snapshot, rule)
                else:
                    result = False
            
            else:
                # Nodo desconocido
                result = False

            return not result if negated else result

        @staticmethod
        def _effects_has_payload(effects: RuleEffects) -> bool:
            return _effects_has_payload_fallback(effects)

        def _matches_vision(
            self,
            condition: dict[str, object],
            snapshot: SnapshotState | None,
            rule: Rule | None = None,
        ) -> bool:
            rule_id = getattr(rule, "rule_id", None)
            # NUEVO: Parsear filtro de sector opcional (int, list[int], o None)
            sector_filter_raw = condition.get("sector")
            if sector_filter_raw is None:
                sector_filter = None
            elif isinstance(sector_filter_raw, list):
                sector_filter = [int(s) for s in sector_filter_raw if s is not None]
            else:
                try:
                    sector_filter = int(sector_filter_raw)
                except (TypeError, ValueError):
                    sector_filter = None
            
            info: dict[str, object] = {
                "rule_id": rule_id,
                "class_name": condition.get("class_name"),
                "sector": sector_filter,
                "condition": condition,
            }

            if snapshot is None:
                info["reason"] = "snapshot_none"
                self._log_vision_debug(info)
                return False

            class_name = str(condition.get("class_name", "")).strip()
            if not class_name:
                info["reason"] = "class_empty"
                self._log_vision_debug(info)
                return False

            window_sec_raw = condition.get("window_sec", WINDOW_SHORT_SEC)
            try:
                window_sec = int(window_sec_raw or WINDOW_SHORT_SEC)
            except (TypeError, ValueError):
                window_sec = int(WINDOW_SHORT_SEC)
            window_key = WINDOW_SHORT_SEC if window_sec <= WINDOW_SHORT_SEC else WINDOW_LONG_SEC
            window = "short" if window_key == WINDOW_SHORT_SEC else "long"
            info.update({"window_sec": window_sec, "window": window})

            # NUEVO: Modo de evaluación de sectores (aggregate vs any)
            sector_mode = str(condition.get("sector_mode", "aggregate")).lower()
            if sector_mode == "any":
                # Determinar lista de sectores a evaluar
                if sector_filter is not None:
                    sectors_to_check = sector_filter if isinstance(sector_filter, list) else [sector_filter]
                else:
                    # "Todos los sectores" con "por sector": usar sectores disponibles del snapshot
                    sectors_data = snapshot.sectors_short if window == "short" else snapshot.sectors_long
                    sectors_to_check = sorted(sectors_data.keys()) if sectors_data else []
                
                if sectors_to_check:
                    # Evaluar cada sector independientemente (OR)
                    for single_sector in sectors_to_check:
                        # Crear condición modificada para un solo sector
                        single_condition = dict(condition)
                        single_condition["sector"] = single_sector
                        single_condition["sector_mode"] = "aggregate"  # Evitar recursión
                        if self._matches_vision(single_condition, snapshot, rule):
                            info["reason"] = f"ok_any_sector_{single_sector}"
                            self._log_vision_debug(info)
                            return True
                    info["reason"] = "no_sector_matched_any"
                    self._log_vision_debug(info)
                    return False

            # NUEVO: Usar estadísticas por sector si se especifica filtro
            if sector_filter is not None:
                stats_raw = snapshot.get_sector_class_stats(sector_filter, class_name, window)
            else:
                stats_raw = snapshot.get_class_stats(class_name, window)
            stats = stats_raw if isinstance(stats_raw, dict) else None
            stats_count = int(stats.get("count", 0) or 0) if stats else 0
            stats_area_avg = _ensure_float(stats.get("area_avg")) if stats else None
            stats_area_max = _ensure_float(stats.get("area_max")) if stats else None
            stats_area_avg_cm2 = _ensure_float(stats.get("area_avg_cm2")) if stats else None
            stats_area_max_cm2 = _ensure_float(stats.get("area_max_cm2")) if stats else None
            stats_conf_raw = stats.get("conf_avg") if stats else None
            stats_conf_avg = _ensure_float(stats_conf_raw) if stats else None
            info.update(
                {
                    "stats_count": stats_count,
                    "stats_area_avg": stats_area_avg,
                    "stats_area_max": stats_area_max,
                    "stats_area_avg_cm2": stats_area_avg_cm2,
                    "stats_area_max_cm2": stats_area_max_cm2,
                    "conf_avg": stats_conf_avg,
                    "conf_raw": stats_conf_raw,
                }
            )

            min_count_raw = condition.get("min_count", 1)
            try:
                count_required = int(min_count_raw or 0)
            except (TypeError, ValueError):
                count_required = 0 if min_count_raw in {0, "0"} else 1
            if count_required < 0:
                count_required = 0
            info["min_count"] = count_required

            max_count_raw = condition.get("max_count")
            try:
                max_allowed = int(max_count_raw) if max_count_raw not in {None, ""} else None
            except (TypeError, ValueError):
                max_allowed = None
            if isinstance(max_allowed, int) and max_allowed < 0:
                max_allowed = None
            info["max_count"] = max_allowed

            actual_count = stats_count if stats is not None else snapshot.get_count(class_name, window)
            info["actual_count"] = actual_count

            # Modo simple: solo detectar presencia (ignorar el resto de filtros)
            detection_only = bool(condition.get("detection_only"))
            info["detection_only"] = detection_only
            if detection_only:
                if actual_count >= 1:
                    info["reason"] = "ok_detection_only"
                    self._log_vision_debug(info)
                    return True
                else:
                    info["reason"] = "not_detected"
                    self._log_vision_debug(info)
                    return False

            if count_required > 0 and actual_count < count_required:
                info["reason"] = "count_below_min"
                self._log_vision_debug(info)
                return False
            if max_allowed is not None and actual_count > max_allowed:
                info["reason"] = "count_above_max"
                self._log_vision_debug(info)
                return False
            if count_required == 0:
                if max_allowed == 0 and actual_count != 0:
                    info["reason"] = "count_not_zero"
                    self._log_vision_debug(info)
                    return False
                if max_allowed is None and actual_count != 0:
                    info["reason"] = "count_positive_when_zero_expected"
                    self._log_vision_debug(info)
                    return False

            area_unit = str(condition.get("area_unit", AREA_UNIT_PX)).strip().lower() or AREA_UNIT_PX
            min_area = _ensure_float(condition.get("min_area"))
            max_area = _ensure_float(condition.get("max_area"))
            metric = None
            metric_available = True
            metric_source = "area_avg_cm2" if area_unit == AREA_UNIT_CM else "area_avg_px"
            if area_unit == AREA_UNIT_CM:
                metric = _extract_area_cm(snapshot, class_name, window)
                if metric is None:
                    metric_available = False
            else:
                metric = _extract_area_px(snapshot, class_name, window)
                if metric is None:
                    metric_available = False
            info.update(
                {
                    "area_unit": area_unit,
                    "metric": metric,
                    "metric_available": metric_available,
                    "min_area": min_area,
                    "max_area": max_area,
                    "metric_source": metric_source,
                }
            )
            if metric_available:
                if min_area is not None and metric < min_area:
                    info["reason"] = "area_below_min"
                    self._log_vision_debug(info)
                    return False
                if max_area is not None and metric > max_area:
                    info["reason"] = "area_above_max"
                    self._log_vision_debug(info)
                    return False
            elif area_unit == AREA_UNIT_CM and (min_area is not None or max_area is not None):
                temp_info = info.copy()
                temp_info["reason"] = "area_metric_unavailable"
                self._log_vision_debug(temp_info)

            min_conf = _ensure_float(condition.get("min_conf"))
            if min_conf is None:
                min_conf = 0.0
            info["min_conf"] = min_conf
            if min_conf > 0:
                if stats is None:
                    info["reason"] = "stats_missing"
                    self._log_vision_debug(info)
                    return False
                conf_avg = info.get("conf_avg")
                if conf_avg is None:
                    reason = "conf_invalid" if info.get("conf_raw") is not None else "conf_missing"
                    info["reason"] = reason
                    self._log_vision_debug(info)
                    return False
                if float(conf_avg) < float(min_conf):
                    info["reason"] = "conf_below_min"
                    self._log_vision_debug(info)
                    return False

            info["reason"] = "ok"
            self._log_vision_debug(info)
            return True

        def _log_vision_debug(self, info: dict[str, object]) -> None:
            if LOGGER.isEnabledFor(logging.DEBUG):
                match_status = "OK" if str(info.get("reason", "")).lower() == "ok" else "FAIL"
                message = (
                    "Visión %s | rule=%s class=%s window=%ss count=%s (min=%s max=%s) "
                    "area=%s (min=%s max=%s unit=%s src=%s avail=%s) conf=%s (min=%s) reason=%s"
                )
                LOGGER.debug(
                    message,
                    match_status,
                    info.get("rule_id") or "?",
                    info.get("class_name") or "?",
                    info.get("window_sec"),
                    info.get("actual_count"),
                    info.get("min_count"),
                    info.get("max_count"),
                    info.get("metric"),
                    info.get("min_area"),
                    info.get("max_area"),
                    info.get("area_unit"),
                    info.get("metric_source"),
                    info.get("metric_available"),
                    info.get("conf_avg"),
                    info.get("min_conf"),
                    info.get("reason"),
                )
                extra_parts: list[str] = []
                for key in ("stats_area_avg", "stats_area_max", "stats_area_avg_cm2", "stats_area_max_cm2"):
                    value = info.get(key)
                    if value is not None:
                        extra_parts.append(f"{key}={value}")
                if extra_parts:
                    LOGGER.debug("Visión extra -> %s", ", ".join(extra_parts))
            callback = self._vision_diag_callback
            if callback is not None:
                try:
                    callback(info.copy())
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Callback diagnóstico de visión falló: %s", exc)

        def _matches_plc_bit(self, condition: dict[str, object], rule: Rule | None = None) -> bool:
            if self._plc_condition_provider is None:
                return False
            payload = copy.deepcopy(condition) if isinstance(condition, dict) else {}
            try:
                return bool(self._plc_condition_provider(payload))
            except Exception:  # noqa: BLE001
                return False

        def _matches_rule(self, condition: dict[str, object], rule: Rule | None = None) -> bool:
            """
            Evalúa una condición de meta-regla: si otra regla ha disparado X veces en Y segundos.
            
            Campos de condition:
                - rule_id: ID de la regla a monitorear (o lista de IDs)
                - min_firings: Mínimo de disparos requeridos
                - max_firings: Máximo de disparos (opcional)
                - window_sec: Ventana de tiempo en segundos
                - debounce_ms: Tiempo mínimo entre disparos para contar (opcional)
                - cooldown_sec: Cooldown después de activarse (opcional)
            """
            current_rule_id = getattr(rule, "rule_id", None) if rule else None
            
            target_ids_raw = condition.get("rule_id") or condition.get("target_rule_id")
            if not target_ids_raw:
                LOGGER.debug("Condición de regla ignorada: no se especificó rule_id")
                return False
            
            # Soportar múltiples reglas
            if isinstance(target_ids_raw, list):
                target_ids = [str(rid) for rid in target_ids_raw if rid]
            else:
                target_ids = [str(target_ids_raw)]
            
            # Evitar auto-referencia
            if current_rule_id and current_rule_id in target_ids:
                LOGGER.warning(
                    "Meta-regla %s intenta monitorearse a sí misma, ignorando",
                    current_rule_id
                )
                target_ids = [tid for tid in target_ids if tid != current_rule_id]
                if not target_ids:
                    return False
            
            # Verificar cooldown
            if current_rule_id and _rule_firing_history.is_in_cooldown(current_rule_id):
                LOGGER.debug("Meta-regla %s en cooldown, ignorando", current_rule_id)
                return False
            
            # Obtener parámetros
            try:
                min_firings = int(condition.get("min_firings", 1) or 1)
            except (TypeError, ValueError):
                min_firings = 1
            
            max_firings_raw = condition.get("max_firings")
            try:
                max_firings = int(max_firings_raw) if max_firings_raw not in {None, ""} else None
            except (TypeError, ValueError):
                max_firings = None
            
            try:
                window_sec = float(condition.get("window_sec", 60) or 60)
            except (TypeError, ValueError):
                window_sec = 60.0
            
            # Contar disparos
            if len(target_ids) == 1:
                count = _rule_firing_history.get_firing_count(target_ids[0], window_sec)
            else:
                count = _rule_firing_history.get_firing_count_multi(target_ids, window_sec)
            
            # Evaluar condición
            if count < min_firings:
                LOGGER.debug(
                    "Meta-regla: regla(s) %s tiene %d disparos (min=%d) en %.1fs -> False",
                    target_ids, count, min_firings, window_sec
                )
                return False
            
            if max_firings is not None and count > max_firings:
                LOGGER.debug(
                    "Meta-regla: regla(s) %s tiene %d disparos (max=%d) en %.1fs -> False",
                    target_ids, count, max_firings, window_sec
                )
                return False
            
            LOGGER.info(
                "Meta-regla: regla(s) %s tiene %d disparos (min=%d, max=%s) en %.1fs -> True",
                target_ids, count, min_firings, max_firings, window_sec
            )
            return True

        def _actions_to_effects(self, actions: Iterable[dict[str, object]]) -> RuleEffects:
            effects = RuleEffects()
            for action in actions:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind")
                params = action.get("params") if isinstance(action.get("params"), dict) else {}
                if kind == "block_classes":
                    targets = params.get("triggers") if isinstance(params, dict) else []
                    if isinstance(targets, (list, tuple, set)) and targets:
                        duration = params.get("duration_sec") if isinstance(params, dict) else None
                        try:
                            duration_val = int(duration) if duration is not None else 0
                        except Exception:
                            duration_val = 0
                        for item in targets:
                            rule_id = ""
                            label = ""
                            if isinstance(item, dict):
                                rule_id = str(item.get("rule_id", ""))
                                label = str(item.get("label", ""))
                            else:
                                rule_id = str(item)
                            if not rule_id:
                                continue
                            _append_muted_target(effects, rule_id, duration_val, label)
                    else:
                        classes = params.get("classes") if isinstance(params, dict) else []
                        if isinstance(classes, (list, tuple, set)):
                            effects.blocked_classes.update(str(c) for c in classes if c)
                elif kind in {"mute_triggers", "block_triggers"}:
                    targets = params.get("triggers") if isinstance(params, dict) else []
                    duration = params.get("duration_sec") if isinstance(params, dict) else None
                    try:
                        duration_val = int(duration) if duration is not None else 0
                    except Exception:
                        duration_val = 0
                    for item in targets if isinstance(targets, (list, tuple, set)) else []:
                        rule_id = ""
                        label = ""
                        if isinstance(item, dict):
                            rule_id = str(item.get("rule_id", ""))
                            label = str(item.get("label", ""))
                        else:
                            rule_id = str(item)
                        if not rule_id:
                            continue
                        _append_muted_target(effects, rule_id, duration_val, label)
                elif kind in {"mute_triggers", "block_triggers"}:
                    targets = params.get("triggers") if isinstance(params, dict) else []
                    duration = params.get("duration_sec") if isinstance(params, dict) else None
                    try:
                        duration_val = int(duration) if duration is not None else 0
                    except Exception:
                        duration_val = 0
                    for item in targets if isinstance(targets, (list, tuple, set)) else []:
                        rule_id = ""
                        label = ""
                        if isinstance(item, dict):
                            rule_id = str(item.get("rule_id", ""))
                            label = str(item.get("label", ""))
                        else:
                            rule_id = str(item)
                        if not rule_id:
                            continue
                        _append_muted_target(effects, rule_id, duration_val, label)
                elif kind in {"mute_triggers", "block_triggers"}:
                    targets = params.get("triggers") if isinstance(params, dict) else []
                    duration = params.get("duration_sec") if isinstance(params, dict) else None
                    try:
                        duration_val = int(duration) if duration is not None else 0
                    except Exception:
                        duration_val = 0
                    for item in targets if isinstance(targets, (list, tuple, set)) else []:
                        rule_id = ""
                        label = ""
                        if isinstance(item, dict):
                            rule_id = str(item.get("rule_id", ""))
                            label = str(item.get("label", ""))
                        else:
                            rule_id = str(item)
                        if not rule_id:
                            continue
                        _append_muted_target(effects, rule_id, duration_val, label)
                elif kind == "force_manual_level":
                    level = params.get("level") if isinstance(params, dict) else None
                    if isinstance(level, str) and level:
                        effects.forced_level = level
                elif kind == "resume_level":
                    level = params.get("level") if isinstance(params, dict) else None
                    if isinstance(level, str) and level:
                        effects.resume_level = level
                elif kind == "show_message":
                    text = str(params.get("text", "")).strip()
                    if not text:
                        continue
                    color = str(params.get("color", "#ffbc00")).strip()
                    try:
                        duration = max(500, int(params.get("duration_ms", 4000) or 0))
                    except (TypeError, ValueError):
                        duration = 4000
                    payload = {
                        "text": text,
                        "color": color,
                        "duration_ms": duration,
                    }
                    opacity = params.get("opacity")
                    if isinstance(opacity, (int, float)):
                        payload["opacity"] = max(0.0, min(1.0, float(opacity)))
                    effects.overlay_messages.append(payload)
                elif kind == "take_snapshot":
                    label = str(params.get("label", "")).strip()
                    annotate = bool(params.get("annotate", False))
                    effects.snapshot_requests.append({
                        "label": label,
                        "annotate": annotate,
                    })
                elif kind == "send_plc":
                    try:
                        payload = _coerce_plc_action_params(params)
                    except ValueError as exc:  # noqa: PERF203
                        LOGGER.warning("Acción send_plc ignorada por error de datos: %s", exc)
                        continue
                    effects.plc_orders.append(payload)
            return effects


def _merge_rule_effects(base: RuleEffects, new: RuleEffects) -> RuleEffects:
    if base is new:
        return base
    merged = RuleEffects()
    merged.blocked_classes = set(getattr(base, "blocked_classes", set()))
    merged.blocked_classes.update(getattr(new, "blocked_classes", set()))

    merged.muted_triggers = [*getattr(base, "muted_triggers", []), *getattr(new, "muted_triggers", [])]

    merged.forced_level = getattr(base, "forced_level", None) or getattr(new, "forced_level", None)
    merged.resume_level = getattr(base, "resume_level", None) or getattr(new, "resume_level", None)

    merged.overlay_messages = [*getattr(base, "overlay_messages", []), *getattr(new, "overlay_messages", [])]
    merged.snapshot_requests = [*getattr(base, "snapshot_requests", []), *getattr(new, "snapshot_requests", [])]
    merged.plc_orders = [*getattr(base, "plc_orders", []), *getattr(new, "plc_orders", [])]
    return merged


def _effects_has_payload_fallback(effects: RuleEffects | None) -> bool:
    if effects is None:
        return False
    if getattr(effects, "blocked_classes", None):
        return True
    if getattr(effects, "muted_triggers", None):
        return bool(getattr(effects, "muted_triggers", []))
    if getattr(effects, "forced_level", None) or getattr(effects, "resume_level", None):
        return True
    if getattr(effects, "overlay_messages", None):
        return bool(effects.overlay_messages)
    if getattr(effects, "snapshot_requests", None):
        return bool(effects.snapshot_requests)
    if getattr(effects, "plc_orders", None):
        return bool(effects.plc_orders)
    return False


def _normalize_condition_tree(node: object) -> dict[str, object] | None:
    if not isinstance(node, dict):
        return None
    node_type = str(node.get("type") or node.get("node_type") or "").strip().lower()
    negated = bool(node.get("negated"))

    if node_type == "group":
        operator = str(node.get("operator", "and")).strip().lower()
        if operator not in {"and", "or"}:
            operator = "and"
        raw_children = node.get("children") if isinstance(node.get("children"), list) else []
        children = [child for child in (_normalize_condition_tree(child) for child in raw_children) if child]
        if not children:
            return None
        payload = {
            "type": "group",
            "operator": operator,
            "children": children,
            "negated": negated,
        }
        _assign_node_ids(payload)
        return payload

    if node_type in {"leaf", "condition"} or node.get("condition") or node.get("kind"):
        cond_payload = node.get("condition") if isinstance(node.get("condition"), dict) else node
        normalized = _normalize_rule_condition(cond_payload)
        if not normalized:
            return None
        payload = {"type": "condition", "condition": normalized, "negated": negated}
        _assign_node_ids(payload)
        return payload

    return None


def _wrap_condition_as_tree(condition: dict[str, object]) -> dict[str, object]:
    normalized = _normalize_rule_condition(condition) or {}
    group = _make_condition_group("and")
    group["children"].append(_make_condition_leaf(normalized))
    return group


def _make_condition_group(operator: str = "and", *, negated: bool = False) -> dict[str, object]:
    payload = {
        "type": "group",
        "operator": operator if operator in {"and", "or"} else "and",
        "negated": bool(negated),
        "children": [],
    }
    _assign_node_ids(payload)
    return payload


def _make_condition_leaf(condition: dict[str, object], *, negated: bool = False) -> dict[str, object]:
    payload = {
        "type": "condition",
        "condition": copy.deepcopy(condition),
        "negated": bool(negated),
    }
    _assign_node_ids(payload)
    return payload


def _extract_primary_condition(tree: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(tree, dict):
        return None
    if tree.get("type") == "condition" and isinstance(tree.get("condition"), dict):
        return copy.deepcopy(tree["condition"])
    if tree.get("type") == "group":
        children = tree.get("children") if isinstance(tree.get("children"), list) else []
        for child in children:
            result = _extract_primary_condition(child)
            if result:
                return result
    return None


def _assign_node_ids(node: dict[str, object]) -> None:
    if "node_id" not in node:
        node["node_id"] = uuid.uuid4().hex
    if node.get("type") == "group":
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child in children:
            if isinstance(child, dict):
                _assign_node_ids(child)


def _clone_condition_tree(tree: dict[str, object] | None) -> dict[str, object] | None:
    if tree is None:
        return None
    cloned = copy.deepcopy(tree)
    _assign_node_ids(cloned)
    return cloned



class _Tooltip:
    """Clase auxiliar para mostrar tooltips con retardo al hacer hover."""
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 1500) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip_window: tk.Toplevel | None = None
        self.id: str | None = None
        
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event: tk.Event | None = None) -> None:
        self._unschedule()
        self.id = self.widget.after(self.delay_ms, self.show_tip)

    def _on_leave(self, event: tk.Event | None = None) -> None:
        self._unschedule()
        self.hide_tip()

    def _unschedule(self) -> None:
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def show_tip(self, event: tk.Event | None = None) -> None:
        if self.tip_window or not self.text:
            return
        
        # Crear ventana sin bordes
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        
        # Posicionamiento absoluto relativo al widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        tw.wm_geometry(f"+{x}+{y}")
        
        # Estilo premium: fondo amarillento suave, borde fino
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#FFFFE1", foreground="#333333",
            relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9), padx=8, pady=6,
            wraplength=300 # Evitar tooltips excesivamente anchos
        )
        label.pack()
        
        # Elevar la ventana para asegurar que sea visible
        tw.lift()
        tw.attributes("-topmost", True)

    def hide_tip(self, event: tk.Event | None = None) -> None:
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def _describe_condition_human(condition: dict[str, object]) -> str:
    kind = str(condition.get("kind") or CONDITION_KIND_VISION)
    if kind == CONDITION_KIND_PLC_BIT:
        plc_mode = str(condition.get("plc_mode", "bit")).strip().lower()
        label = condition.get("label") or condition.get("preset_id") or "PLC"
        if plc_mode == "numeric":
            address = condition.get("address", "?")
            operator = condition.get("operator", "=")
            v1 = condition.get("value1", "0")
            if operator == "between":
                v2 = condition.get("value2", "0")
                return f"PLC {label} {address} entre {v1} y {v2}"
            return f"PLC {label} {address} {operator} {v1}"
        
        expected = condition.get("expected_value")
        if expected is None:
            expected = condition.get("expected", 1)
        area = condition.get("area", "?")
        byte = condition.get("byte_index", "?")
        bit = condition.get("bit_index", "?")
        val_desc = "1" if expected in {1, "1", True} else "0"
        return f"PLC {label} {area}{byte}.{bit}={val_desc}"

    if kind == CONDITION_KIND_RULE:
        label = str(condition.get("label", "")).strip() or str(condition.get("rule_id", ""))[:8]
        min_f = condition.get("min_firings", 1)
        max_f = condition.get("max_firings")
        win = condition.get("window_sec", 60)
        
        count_str = f"≥{min_f}"
        if max_f:
            count_str += f"-{max_f}"
            
        return f"Regla '{label}' {count_str}veces/{win}s"
    
    class_name = condition.get("class_name") or condition.get("class") or "Clase"
    
    # Preparar texto de sector
    sector_text = ""
    sector = condition.get("sector")
    sector_mode = str(condition.get("sector_mode", "aggregate")).lower()
    if sector is not None:
        if isinstance(sector, list):
            sector_text = f" | S:{','.join(str(s) for s in sector)}"
        else:
            sector_text = f" | S:{sector}"
    elif sector_mode == "any":
        sector_text = " | S:todos"
    if sector_mode == "any":
        sector_text += " | por-sector"
    
    # Modo simple: solo detectar presencia
    if bool(condition.get("detection_only")):
        return f"Visión {class_name} (presencia){sector_text}"
    
    min_count = condition.get("min_count", 1)
    max_count = condition.get("max_count")
    parts = [f"Visión {class_name}"]
    if min_count:
        parts.append(f"≥{min_count}")
    if isinstance(max_count, (int, float)):
        parts.append(f"≤{int(max_count)}")
    window = condition.get("window_sec")
    if window:
        parts.append(f"{window}s")
    return " ".join(parts) + sector_text


def _condition_tree_to_text(node: dict[str, object] | None) -> str:
    if not node:
        return "Sin condiciones"
    node_type = node.get("type")
    negated = bool(node.get("negated"))
    if node_type == "condition":
        text = _describe_condition_human(node.get("condition", {}))
        return f"NO ({text})" if negated else text
    if node_type == "group":
        operator = node.get("operator", "and")
        glue = " Y " if operator == "and" else " O "
        children = [
            _condition_tree_to_text(child)
            for child in node.get("children", [])
            if isinstance(child, dict)
        ]
        if not children:
            return "(vacío)"
        text = glue.join(children)
        text = f"({text})"
        if negated:
            text = f"NO {text}"
        return text
    return "(desconocido)"


if _EXTERNAL_RULE_ENGINE:
    _ExternalRuleEngineBase = RuleEngine

    class RuleEngine(_ExternalRuleEngineBase):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._muted_rule_ids: set[str] = set()

        def set_muted_rules(self, rule_ids: Iterable[str]) -> None:
            self._muted_rule_ids = {str(rid) for rid in rule_ids if rid}

        def process(self, snapshot: SnapshotState | None) -> RuleEvaluation:  # type: ignore[override]
            try:
                rules_attr = getattr(self, "_rules", None)
                matches_fn = getattr(self, "_matches", None)
                actions_to_effects = getattr(self, "_actions_to_effects", None)
                if rules_attr is None or not callable(matches_fn) or not callable(actions_to_effects):
                    return super().process(snapshot)
                effects_has_payload = getattr(self, "_effects_has_payload", None)
                if not callable(effects_has_payload):
                    effects_has_payload = _effects_has_payload_fallback

                matched_no_effect: list = []
                aggregated_effects = RuleEffects()
                matched_rules: list = []
                current_priority: int | None = None

                for rule in sorted(
                    (r for r in rules_attr if getattr(r, "enabled", True)),
                    key=lambda r: getattr(r, "priority", 0),
                    reverse=True,
                ):
                    LOGGER.debug(
                        "Evaluando regla %s (%s) prioridad=%s",
                        getattr(rule, "rule_id", "?"),
                        getattr(rule, "name", ""),
                        getattr(rule, "priority", ""),
                    )
                    if str(getattr(rule, "rule_id", "")) in getattr(self, "_muted_rule_ids", set()):
                        LOGGER.debug(
                            "Regla %s ignorada por deshabilitacion temporal",
                            getattr(rule, "rule_id", "?"),
                        )
                        continue
                    if not matches_fn(rule, snapshot):
                        LOGGER.debug("Regla %s no coincide con el snapshot", getattr(rule, "rule_id", "?"))
                        continue
                    try:
                        effects = actions_to_effects(rule.actions)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug(
                            "Acciones de la regla %s no se pudieron evaluar (%s)",
                            getattr(rule, "rule_id", "?"),
                            exc,
                        )
                        continue

                    if not effects_has_payload(effects):
                        LOGGER.debug(
                            "Regla %s coincide pero no produce efectos (acciones vacías o sin resultado)",
                            getattr(rule, "rule_id", "?"),
                        )
                        matched_no_effect.append(rule)
                        continue

                    rule_priority = getattr(rule, "priority", 0)
                    if current_priority is None:
                        current_priority = rule_priority
                    if rule_priority != current_priority:
                        LOGGER.debug(
                            "Regla %s ignorada por prioridad menor (%s < %s)",
                            getattr(rule, "rule_id", "?"),
                            rule_priority,
                            current_priority,
                        )
                        break

                    blocked_len = len(getattr(effects, "blocked_classes", []) or [])
                    overlay_len = len(getattr(effects, "overlay_messages", []) or [])
                    snapshot_len = len(getattr(effects, "snapshot_requests", []) or [])
                    plc_len = len(getattr(effects, "plc_orders", []) or [])
                    LOGGER.info(
                        "Regla %s activada con efectos: bloqueos=%s, forzado=%s, resume=%s, mensajes=%s, snapshots=%s, plc=%s",
                        getattr(rule, "rule_id", "?"),
                        blocked_len,
                        getattr(effects, "forced_level", None),
                        getattr(effects, "resume_level", None),
                        overlay_len,
                        snapshot_len,
                        plc_len,
                    )
                    aggregated_effects = _merge_rule_effects(aggregated_effects, effects)
                    matched_rules.append(rule)

                if matched_rules:
                    return RuleEvaluation(effects=aggregated_effects, matches=matched_rules)

                if matched_no_effect:
                    LOGGER.debug(
                        "Reglas coincidentes sin efecto: %s",
                        [getattr(rule, "rule_id", "?") for rule in matched_no_effect],
                    )
                    return RuleEvaluation(effects=RuleEffects(), matches=matched_no_effect)

                LOGGER.debug("Ninguna regla coincidió con el snapshot actual")
                return RuleEvaluation(effects=RuleEffects(), matches=[])
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("RuleEngine externo: se delega en implementación original (%s)", exc)
                return super().process(snapshot)

        def _matches(self, rule: Rule, snapshot: SnapshotState | None) -> bool:
            tree = getattr(rule, "condition_tree", None)
            if tree:
                return self._matches_condition_tree(tree, snapshot, rule)
            condition = rule.condition if isinstance(rule.condition, dict) else {}
            return self._matches_single_condition(condition, snapshot, rule)

        def _matches_single_condition(
            self,
            condition: dict[str, object],
            snapshot: SnapshotState | None,
            rule: Rule | None,
        ) -> bool:
            kind = str(condition.get("kind") or CONDITION_KIND_VISION).strip().lower()
            if kind == CONDITION_KIND_PLC_BIT:
                return self._matches_plc_bit(condition, rule)
            return self._matches_vision(condition, snapshot, rule)

        def _matches_condition_tree(
            self,
            node: dict[str, object],
            snapshot: SnapshotState | None,
            rule: Rule | None,
        ) -> bool:
            result = self._evaluate_tree_node(node, snapshot, rule)
            negated = bool(node.get("negated"))
            return (not result) if negated else result

        def _evaluate_tree_node(
            self,
            node: dict[str, object],
            snapshot: SnapshotState | None,
            rule: Rule | None,
        ) -> bool:
            node_type = str(node.get("type", "")).strip().lower()
            if node_type == "group":
                operator = str(node.get("operator", "and")).strip().lower()
                children = [child for child in node.get("children", []) if isinstance(child, dict)]
                if not children:
                    return False
                results = [self._matches_condition_tree(child, snapshot, rule) for child in children]
                return all(results) if operator == "and" else any(results)
            if node_type in {"leaf", "condition"}:
                condition = node.get("condition") if isinstance(node.get("condition"), dict) else {}
                return self._matches_single_condition(condition, snapshot, rule)
            # Fallback: si el nodo parece una condición plana
            if node.get("kind"):
                return self._matches_single_condition(node, snapshot, rule)
            return False

        def _log_vision_debug(self, info: dict[str, object]) -> None:
            if LOGGER.isEnabledFor(logging.DEBUG):
                match_status = "OK" if str(info.get("reason", "")).lower() == "ok" else "FAIL"
                message = (
                    "Visión %s | rule=%s class=%s window=%ss count=%s (min=%s max=%s) "
                    "area=%s (min=%s max=%s unit=%s src=%s avail=%s) conf=%s (min=%s) reason=%s"
                )
                LOGGER.debug(
                    message,
                    match_status,
                    info.get("rule_id") or "?",
                    info.get("class_name") or "?",
                    info.get("window_sec"),
                    info.get("actual_count"),
                    info.get("min_count"),
                    info.get("max_count"),
                    info.get("metric"),
                    info.get("min_area"),
                    info.get("max_area"),
                    info.get("area_unit"),
                    info.get("metric_source"),
                    info.get("metric_available"),
                    info.get("conf_avg"),
                    info.get("min_conf"),
                    info.get("reason"),
                )
                extra_parts: list[str] = []
                for key in ("stats_area_avg", "stats_area_max", "stats_area_avg_cm2", "stats_area_max_cm2"):
                    value = info.get(key)
                    if value is not None:
                        extra_parts.append(f"{key}={value}")
                if extra_parts:
                    LOGGER.debug("Visión extra -> %s", ", ".join(extra_parts))
            callback = self._vision_diag_callback
            if callback is not None:
                try:
                    callback(info.copy())
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Callback diagnóstico de visión falló: %s", exc)

        def _matches_vision(
            self,
            condition: dict[str, object],
            snapshot: SnapshotState | None,
            rule: Rule | None = None,
        ) -> bool:
            rule_id = getattr(rule, "rule_id", None)
            info: dict[str, object] = {
                "rule_id": rule_id,
                "class_name": condition.get("class_name"),
                "condition": condition,
            }

            if snapshot is None:
                info["reason"] = "snapshot_none"
                self._log_vision_debug(info)
                return False

            if snapshot is None:
                return False
            class_name = str(condition.get("class_name", "")).strip()
            if not class_name:
                info["reason"] = "class_empty"
                self._log_vision_debug(info)
                return False
            window_sec_raw = condition.get("window_sec", WINDOW_SHORT_SEC)
            try:
                window_sec = int(window_sec_raw or WINDOW_SHORT_SEC)
            except (TypeError, ValueError):
                window_sec = int(WINDOW_SHORT_SEC)
            window_key = WINDOW_SHORT_SEC if window_sec <= WINDOW_SHORT_SEC else WINDOW_LONG_SEC
            window = "short" if window_key == WINDOW_SHORT_SEC else "long"
            info.update({
                "window_sec": window_sec,
                "window": window,
            })

            # Parsear filtro de sector opcional (int, list[int], o None)
            sector_filter_raw = condition.get("sector")
            if sector_filter_raw is None:
                sector_filter = None
            elif isinstance(sector_filter_raw, list):
                sector_filter = [int(s) for s in sector_filter_raw if s is not None]
            else:
                try:
                    sector_filter = int(sector_filter_raw)
                except (TypeError, ValueError):
                    sector_filter = None
            info["sector"] = sector_filter

            # Modo de evaluación de sectores (aggregate vs any)
            sector_mode = str(condition.get("sector_mode", "aggregate")).lower()
            if sector_mode == "any":
                # Determinar lista de sectores a evaluar
                if sector_filter is not None:
                    sectors_to_check = sector_filter if isinstance(sector_filter, list) else [sector_filter]
                else:
                    # "Todos los sectores" con "por sector": usar sectores disponibles del snapshot
                    sectors_data = snapshot.sectors_short if window == "short" else snapshot.sectors_long
                    sectors_to_check = sorted(sectors_data.keys()) if sectors_data else []
                
                if sectors_to_check:
                    # Evaluar cada sector independientemente (OR)
                    for single_sector in sectors_to_check:
                        # Crear condición modificada para un solo sector
                        single_condition = dict(condition)
                        single_condition["sector"] = single_sector
                        single_condition["sector_mode"] = "aggregate"  # Evitar recursión
                        if self._matches_vision(single_condition, snapshot, rule):
                            info["reason"] = f"ok_any_sector_{single_sector}"
                            self._log_vision_debug(info)
                            return True
                    info["reason"] = "no_sector_matched_any"
                    self._log_vision_debug(info)
                    return False

            # Usar estadísticas por sector si se especifica filtro
            if sector_filter is not None:
                stats_raw = snapshot.get_sector_class_stats(sector_filter, class_name, window)
            else:
                stats_raw = snapshot.get_class_stats(class_name, window)
            stats = stats_raw if isinstance(stats_raw, dict) else None
            stats_count = int(stats.get("count", 0) or 0) if stats else 0
            stats_area_avg = _ensure_float(stats.get("area_avg")) if stats else None
            stats_area_max = _ensure_float(stats.get("area_max")) if stats else None
            stats_area_avg_cm2 = _ensure_float(stats.get("area_avg_cm2")) if stats else None
            stats_area_max_cm2 = _ensure_float(stats.get("area_max_cm2")) if stats else None
            stats_conf_raw = stats.get("conf_avg") if stats else None
            stats_conf_avg = _ensure_float(stats_conf_raw) if stats else None
            info.update(
                {
                    "stats_count": stats_count,
                    "stats_area_avg": stats_area_avg,
                    "stats_area_max": stats_area_max,
                    "stats_area_avg_cm2": stats_area_avg_cm2,
                    "stats_area_max_cm2": stats_area_max_cm2,
                    "conf_avg": stats_conf_avg,
                    "conf_raw": stats_conf_raw,
                }
            )

            min_count_raw = condition.get("min_count", 1)
            try:
                count_required = int(min_count_raw or 0)
            except (TypeError, ValueError):
                count_required = 0 if min_count_raw in {0, "0"} else 1
            if count_required < 0:
                count_required = 0
            info["min_count"] = count_required

            max_count_raw = condition.get("max_count")
            try:
                max_allowed = int(max_count_raw) if max_count_raw not in {None, ""} else None
            except (TypeError, ValueError):
                max_allowed = None
            if isinstance(max_allowed, int) and max_allowed < 0:
                max_allowed = None
            info["max_count"] = max_allowed

            actual_count = stats_count if stats is not None else snapshot.get_count(class_name, window)
            info["actual_count"] = actual_count

            # Modo simple: solo detectar presencia (ignorar el resto de filtros)
            detection_only = bool(condition.get("detection_only"))
            info["detection_only"] = detection_only
            if detection_only:
                if actual_count >= 1:
                    info["reason"] = "ok_detection_only"
                    self._log_vision_debug(info)
                    return True
                else:
                    info["reason"] = "not_detected"
                    self._log_vision_debug(info)
                    return False

            if count_required > 0 and actual_count < count_required:
                info["reason"] = "count_below_min"
                self._log_vision_debug(info)
                return False
            if max_allowed is not None and actual_count > max_allowed:
                info["reason"] = "count_above_max"
                self._log_vision_debug(info)
                return False
            if count_required == 0:
                if max_allowed == 0:
                    if actual_count != 0:
                        info["reason"] = "count_not_zero"
                        self._log_vision_debug(info)
                        return False
                elif max_allowed is None and actual_count != 0:
                    info["reason"] = "count_positive_when_zero_expected"
                    self._log_vision_debug(info)
                    return False
            area_unit = str(condition.get("area_unit", AREA_UNIT_PX)).strip().lower() or AREA_UNIT_PX
            min_area = _ensure_float(condition.get("min_area"))
            max_area = _ensure_float(condition.get("max_area"))
            metric = None
            metric_available = True
            metric_source = "area_avg_cm2" if area_unit == AREA_UNIT_CM else "area_avg_px"
            if area_unit == AREA_UNIT_CM:
                metric = _extract_area_cm(snapshot, class_name, window)
                if metric is None:
                    metric_available = False
            else:
                metric = _extract_area_px(snapshot, class_name, window)
                if metric is None:
                    metric_available = False
            info.update({
                "area_unit": area_unit,
                "metric": metric,
                "metric_available": metric_available,
                "min_area": min_area,
                "max_area": max_area,
                "metric_source": metric_source,
            })
            if metric_available:
                if min_area is not None and metric < min_area:
                    info["reason"] = "area_below_min"
                    self._log_vision_debug(info)
                    return False
                if max_area is not None and metric > max_area:
                    info["reason"] = "area_above_max"
                    self._log_vision_debug(info)
                    return False
            elif area_unit == AREA_UNIT_CM and (min_area is not None or max_area is not None):
                temp_info = info.copy()
                temp_info["reason"] = "area_metric_unavailable"
                self._log_vision_debug(temp_info)
            min_conf = _ensure_float(condition.get("min_conf"))
            if min_conf is None:
                min_conf = 0.0
            info["min_conf"] = min_conf
            if min_conf > 0:
                if stats is None:
                    info["reason"] = "stats_missing"
                    self._log_vision_debug(info)
                    return False
                conf_avg = info.get("conf_avg")
                if conf_avg is None:
                    reason = "conf_invalid" if info.get("conf_raw") is not None else "conf_missing"
                    info["reason"] = reason
                    self._log_vision_debug(info)
                    return False
                if float(conf_avg) < float(min_conf):
                    info["reason"] = "conf_below_min"
                    self._log_vision_debug(info)
                    return False
            info["reason"] = "ok"
            self._log_vision_debug(info)
            return True

        def _matches_plc_bit(self, condition: dict[str, object], rule: Rule | None = None) -> bool:
            if self._plc_condition_provider is None:
                LOGGER.debug("Condición PLC ignorada (proveedor no configurado): %s", condition)
                return False
            payload = copy.deepcopy(condition) if isinstance(condition, dict) else {}
            try:
                result = bool(self._plc_condition_provider(payload))
                LOGGER.debug(
                    "Condición PLC evaluada: rule=%s payload=%s resultado=%s",
                    getattr(rule, "rule_id", None),
                    payload,
                    result,
                )
                return result
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Condición PLC ignorada por error: %s", exc)
                return False

        def _actions_to_effects(self, actions: Iterable[dict[str, object]]) -> RuleEffects:
            effects = RuleEffects()
            for action in actions:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind")
                params = action.get("params") if isinstance(action.get("params"), dict) else {}
                if kind == "block_classes":
                    classes = params.get("classes") if isinstance(params, dict) else []
                    if isinstance(classes, (list, tuple, set)):
                        effects.blocked_classes.update(str(c) for c in classes if c)
                        LOGGER.debug("Acción block_classes aplicada: %s", effects.blocked_classes)
                elif kind == "force_manual_level":
                    level = params.get("level") if isinstance(params, dict) else None
                    if isinstance(level, str) and level:
                        effects.forced_level = level
                        LOGGER.debug("Acción force_manual_level aplicada: %s", level)
                elif kind == "resume_level":
                    level = params.get("level") if isinstance(params, dict) else None
                    if isinstance(level, str) and level:
                        effects.resume_level = level
                        LOGGER.debug("Acción resume_level aplicada: %s", level)
                elif kind == "show_message":
                    text = str(params.get("text", "")).strip()
                    if not text:
                        continue
                    color = str(params.get("color", "#ffbc00")).strip()
                    try:
                        duration = max(500, int(params.get("duration_ms", 4000) or 0))
                    except (TypeError, ValueError):
                        duration = 4000
                    payload = {
                        "text": text,
                        "color": color,
                        "duration_ms": duration,
                    }
                    opacity = params.get("opacity")
                    if isinstance(opacity, (int, float)):
                        payload["opacity"] = max(0.0, min(1.0, float(opacity)))
                    effects.overlay_messages.append(payload)
                    LOGGER.debug("Acción show_message preparada: %s", payload)
                elif kind == "take_snapshot":
                    label = str(params.get("label", "")).strip()
                    annotate = bool(params.get("annotate", False))
                    enabled = _ensure_bool(params.get("enabled", True), default=True)
                    require_trigger = _ensure_bool(params.get("require_trigger", True), default=True)
                    cooldown_raw = _ensure_float(params.get("cooldown_sec"))
                    cooldown_sec = max(0.0, float(cooldown_raw)) if cooldown_raw is not None else 0.0
                    effects.snapshot_requests.append({
                        "label": label,
                        "annotate": annotate,
                        "enabled": enabled,
                        "require_trigger": require_trigger,
                        "cooldown_sec": cooldown_sec,
                    })
                    LOGGER.debug("Acción take_snapshot preparada: label=%s annotate=%s enabled=%s require_trigger=%s cooldown_sec=%s", label, annotate, enabled, require_trigger, cooldown_sec)
                elif kind == "send_plc":
                    try:
                        payload = _coerce_plc_action_params(params)
                    except ValueError as exc:
                        LOGGER.warning("Acción send_plc ignorada por error de datos: %s", exc)
                        continue
                    effects.plc_orders.append(payload)
                    LOGGER.debug("Acción send_plc preparada: %s", payload)
                else:
                    LOGGER.debug("Acción desconocida/ignorada: %s", action)
                    continue
            return effects

        def _matches_plc_bit(self, condition: dict[str, object], rule: Rule | None = None) -> bool:
            if self._plc_condition_provider is None:
                LOGGER.debug("Condición PLC ignorada (proveedor no configurado): %s", condition)
                return False
            payload = copy.deepcopy(condition) if isinstance(condition, dict) else {}
            try:
                result = bool(self._plc_condition_provider(payload))
                LOGGER.debug(
                    "Condición PLC evaluada: rule=%s payload=%s resultado=%s",
                    getattr(rule, "rule_id", None),
                    payload,
                    result,
                )
                return result
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Condición PLC ignorada por error: %s", exc)
                return False

        def _actions_to_effects(self, actions: Iterable[dict[str, object]]) -> RuleEffects:
            effects = RuleEffects()
            for action in actions:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind")
                params = action.get("params") if isinstance(action.get("params"), dict) else {}
                if kind == "block_classes":
                    classes = params.get("classes") if isinstance(params, dict) else []
                    if isinstance(classes, (list, tuple, set)):
                        effects.blocked_classes.update(str(c) for c in classes if c)
                        LOGGER.debug("Acción block_classes aplicada: %s", effects.blocked_classes)
                elif kind == "force_manual_level":
                    level = params.get("level") if isinstance(params, dict) else None
                    if isinstance(level, str) and level:
                        effects.forced_level = level
                        LOGGER.debug("Acción force_manual_level aplicada: %s", level)
                elif kind == "resume_level":
                    level = params.get("level") if isinstance(params, dict) else None
                    if isinstance(level, str) and level:
                        effects.resume_level = level
                        LOGGER.debug("Acción resume_level aplicada: %s", level)
                elif kind == "show_message":
                    text = str(params.get("text", "")).strip()
                    if not text:
                        continue
                    color = str(params.get("color", "#ffbc00")).strip()
                    try:
                        duration = max(500, int(params.get("duration_ms", 4000) or 0))
                    except (TypeError, ValueError):
                        duration = 4000
                    payload = {
                        "text": text,
                        "color": color,
                        "duration_ms": duration,
                    }
                    opacity = params.get("opacity")
                    if isinstance(opacity, (int, float)):
                        payload["opacity"] = max(0.0, min(1.0, float(opacity)))
                    effects.overlay_messages.append(payload)
                    LOGGER.debug("Acción show_message preparada: %s", payload)
                elif kind == "take_snapshot":
                    label = str(params.get("label", "")).strip()
                    annotate = bool(params.get("annotate", False))
                    enabled = _ensure_bool(params.get("enabled", True), default=True)
                    require_trigger = _ensure_bool(params.get("require_trigger", True), default=True)
                    cooldown_raw = _ensure_float(params.get("cooldown_sec"))
                    cooldown_sec = max(0.0, float(cooldown_raw)) if cooldown_raw is not None else 0.0
                    effects.snapshot_requests.append({
                        "label": label,
                        "annotate": annotate,
                        "enabled": enabled,
                        "require_trigger": require_trigger,
                        "cooldown_sec": cooldown_sec,
                    })
                    LOGGER.debug("Acción take_snapshot preparada: label=%s annotate=%s enabled=%s require_trigger=%s cooldown_sec=%s", label, annotate, enabled, require_trigger, cooldown_sec)
                elif kind == "send_plc":
                    try:
                        payload = _coerce_plc_action_params(params)
                    except ValueError as exc:
                        LOGGER.warning("Acción send_plc ignorada por error de datos: %s", exc)
                        continue
                    effects.plc_orders.append(payload)
                    LOGGER.debug("Acción send_plc preparada: %s", payload)
                else:
                    LOGGER.debug("Acción desconocida/ignorada: %s", action)
                    continue
            return effects

        @staticmethod
        def _effects_has_payload(effects: RuleEffects) -> bool:
            return any(
                [
                    bool(effects.blocked_classes),
                    effects.forced_level is not None,
                    effects.resume_level is not None,
                    bool(effects.overlay_messages),
                    bool(effects.snapshot_requests),
                    bool(effects.plc_orders),
                ]
            )


def _summarize_snapshot(snapshot: SnapshotState | None) -> dict[str, object]:
    if snapshot is None:
        return {"snapshot": None}
    head_short = _head_dict(snapshot.classes_short)
    head_long = _head_dict(snapshot.classes_long)
    return {
        "timestamp": snapshot.timestamp_utc,
        "seconds_since_last_major": snapshot.seconds_since_last_major,
        "last_major_class": snapshot.last_major_class,
        "short_total_classes": len(snapshot.classes_short),
        "long_total_classes": len(snapshot.classes_long),
        "short_sample": head_short,
        "long_sample": head_long,
    }


def _head_dict(source: Mapping[str, Mapping[str, object]] | None, limit: int = 5) -> dict[str, Mapping[str, object]]:
    if not isinstance(source, Mapping):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for idx, (key, value) in enumerate(source.items()):
        if idx >= max(0, limit):
            break
        if isinstance(value, Mapping):
            result[key] = dict(value)
    return result


def _parse_sectors_dict(raw: Any) -> dict[int, dict[str, dict]]:
    """Convierte el payload de sectores a dict tipado: sector_id -> {class_name -> stats}."""
    if not isinstance(raw, dict):
        return {}
    result: dict[int, dict[str, dict]] = {}
    for sector_key, class_data in raw.items():
        try:
            sector_id = int(sector_key)
        except (TypeError, ValueError):
            continue
        if isinstance(class_data, dict):
            result[sector_id] = {
                str(k): dict(v) if isinstance(v, dict) else {}
                for k, v in class_data.items()
            }
    return result

def _effects_to_dict(effects: RuleEffects) -> dict[str, object]:
    return {
        "blocked_classes": sorted(effects.blocked_classes),
        "muted_triggers": [
            {
                "rule_id": str(item.get("rule_id", "")),
                "duration_sec": item.get("duration_sec"),
                "label": item.get("label", ""),
            }
            for item in getattr(effects, "muted_triggers", []) or []
        ],
        "forced_level": effects.forced_level,
        "resume_level": effects.resume_level,
        "overlay_messages": effects.overlay_messages,
        "snapshot_requests": effects.snapshot_requests,
        "plc_orders": effects.plc_orders,
    }


def _append_muted_target(effects: RuleEffects, rule_id: str, duration: int, label: str) -> None:
    try:
        muted_list = getattr(effects, "muted_triggers", None)
        if muted_list is None:
            muted_list = []
            setattr(effects, "muted_triggers", muted_list)
        if not isinstance(muted_list, list):
            muted_list = list(muted_list) if isinstance(muted_list, (set, tuple)) else []
            setattr(effects, "muted_triggers", muted_list)
        muted_list.append({
            "rule_id": rule_id,
            "duration_sec": duration,
            "label": label,
        })
    except Exception:
        LOGGER.debug("No se pudo registrar trigger muteado para %s", rule_id)


def _sanitize_metadata(metadata: dict | None) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        key_str = str(key)
        if isinstance(value, str):
            sanitized[key_str] = value.strip()
        elif isinstance(value, (list, tuple, set)):
            sanitized[key_str] = [str(item).strip() for item in value if item]
        else:
            sanitized[key_str] = value
    return sanitized


_DETECTOR_STARTED = False


def mark_detector_started() -> None:
    global _DETECTOR_STARTED
    _DETECTOR_STARTED = True


def mark_detector_stopped() -> None:
    global _DETECTOR_STARTED
    _DETECTOR_STARTED = False


def is_detector_started() -> bool:
    return _DETECTOR_STARTED


def _ensure_snapshot_path() -> Path:
    with SNAPSHOT_LOCK:
        base = SNAPSHOT_PATH
        if not base.parent.exists():
            base.parent.mkdir(parents=True, exist_ok=True)
        return base


SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "estado_linea.json"
SNAPSHOT_LOCK = threading.Lock()
SNAPSHOT_STATE: dict[str, object] = {}
CONFIG_PATH = Path(__file__).parent.parent / "config" / "sendToPLC_config.json"
RULES_PATH = Path(__file__).parent.parent / "config" / "sendToPLC_rules.json"
PLC_PRESETS_KEY = "plc_presets"
PLC_DEFAULT_PRESET_ID = "factory_default"
PLC_SEND_TIMEOUT_SEC = 3.0
PLC_VERIFY_DELAY_SEC = 0.15
AUTOMATION_ACTION_KINDS = {
    "show_message": "Mostrar mensaje en visor",
    "take_snapshot": "Captura de pantalla",
    "block_classes": "Deshabilitar trigger(s)",
    "mute_triggers": "Deshabilitar trigger(s)",
    "force_manual_level": "Forzar nivel manual",
    "resume_level": "Reanudar nivel manual",
    "send_plc": "Enviar señal PLC",
}
SNAPSHOT_POLL_INTERVAL_SEC = 0.5
PLC_CONDITION_CACHE_TTL_SEC = 1.0
PLC_CONDITION_POLL_INTERVAL_SEC = 0.1
SNAPSHOT_SHORT_WINDOW_SEC = 3.0
SNAPSHOT_LONG_WINDOW_SEC = 30.0
SNAPSHOT_DEFAULT_WRITE_INTERVAL_MS = 1500
SNAPSHOT_MIN_WRITE_INTERVAL_MS = 200
SNAPSHOT_DEFAULT_CLEAN_INTERVAL_SEC = 60.0
DEFAULT_TRIGGER_CLASS = ""
DEFAULT_RESUME_DELAY_SEC = 10
MANUAL_LEVELS = ("NORMAL", "SLOW1", "SLOW2", "SLOW3")

CONDITION_KIND_VISION = "vision"
CONDITION_KIND_PLC_BIT = "plc_bit"

AREA_UNIT_PX = "px"
AREA_UNIT_CM = "cm"


ActionChoice = Literal["ignore", "freeze_level"]
ResumeMode = Literal["instant", "delayed"]


@dataclass
class SnapshotState:
    raw: dict
    timestamp_utc: Optional[str]
    seconds_since_last_major: Optional[float]
    last_major_class: Optional[str]
    classes_short: dict[str, dict]
    classes_long: dict[str, dict]
    sectors_short: dict[int, dict[str, dict]]  # sector_id -> {class_name -> stats}
    sectors_long: dict[int, dict[str, dict]]
    total_major_short: int
    defect_rate_short: float
    trend_vs_long: str

    @classmethod
    def from_payload(cls, payload: dict) -> "SnapshotState":
        meta = payload.get("meta", {})
        stability = payload.get("stability_info", {})
        window_short = payload.get("window_short", {})
        window_long = payload.get("window_long", {})
        return cls(
            raw=payload,
            timestamp_utc=str(meta.get("timestamp")) if meta.get("timestamp") else None,
            seconds_since_last_major=_ensure_float(stability.get("seconds_since_last_major")),
            last_major_class=stability.get("last_major_class"),
            classes_short=_ensure_dict(window_short.get("classes")),
            classes_long=_ensure_dict(window_long.get("classes")),
            sectors_short=_parse_sectors_dict(window_short.get("sectors")),
            sectors_long=_parse_sectors_dict(window_long.get("sectors")),
            total_major_short=int(window_short.get("major_events", 0) or 0),
            defect_rate_short=float(window_short.get("defect_rate", 0.0) or 0.0),
            trend_vs_long=str(window_short.get("trend_vs_long", "stable")),
        )

    def has_class_in_short(self, class_name: str) -> bool:
        if not class_name:
            return False
        class_stats = self.classes_short.get(class_name)
        if not isinstance(class_stats, dict):
            return False
        count = class_stats.get("count")
        return bool(count and count > 0)

    def get_class_stats(self, class_name: str, window: str = "short") -> Optional[dict]:
        if not class_name:
            return None
        window_key = window.lower()
        if window_key == "long":
            return self.classes_long.get(class_name)
        return self.classes_short.get(class_name)

    def get_count(self, class_name: str, window: str = "short") -> int:
        stats = self.get_class_stats(class_name, window)
        if not isinstance(stats, dict):
            return 0
        try:
            return int(stats.get("count", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def get_sector_class_stats(
        self,
        sector: int | list[int] | None,
        class_name: str,
        window: str = "short",
    ) -> Optional[dict]:
        """Obtiene estadisticas de una clase filtrada por sector(es)."""
        if sector is None:
            return self.get_class_stats(class_name, window)

        sectors_data = self.sectors_short if window.lower() == "short" else self.sectors_long

        # Normalizar sector a lista
        sector_list = sector if isinstance(sector, list) else [sector]

        # Agregar estadisticas de los sectores especificados
        total_count = 0
        area_sum = 0.0
        area_weight = 0
        conf_sum = 0.0
        conf_weight = 0
        area_cm2_sum = 0.0
        area_cm2_count = 0
        area_cm2_max: float | None = None

        for s in sector_list:
            sector_stats = sectors_data.get(s, {})
            class_stats = sector_stats.get(class_name, {})
            count = int(class_stats.get("count", 0) or 0)
            total_count += count
            if count <= 0:
                continue
            area_avg = _ensure_float(class_stats.get("area_avg"))
            conf_avg = _ensure_float(class_stats.get("conf_avg"))
            if area_avg is not None:
                area_sum += area_avg * count
                area_weight += count
            if conf_avg is not None:
                conf_sum += conf_avg * count
                conf_weight += count
            area_cm2_avg = _ensure_float(class_stats.get("area_avg_cm2"))
            if area_cm2_avg is not None:
                area_cm2_sum += area_cm2_avg * count
                area_cm2_count += count
            area_cm2_max_val = _ensure_float(class_stats.get("area_max_cm2"))
            if area_cm2_max_val is not None:
                area_cm2_max = area_cm2_max_val if area_cm2_max is None else max(area_cm2_max, area_cm2_max_val)

        # Devolver stats con count=0 en vez de None para evitar fallback a global
        if total_count == 0:
            return {
                "count": 0,
                "area_avg": None,
                "conf_avg": None,
                "area_avg_cm2": None,
                "area_max_cm2": None,
            }

        return {
            "count": total_count,
            "area_avg": (area_sum / area_weight) if area_weight > 0 else None,
            "conf_avg": (conf_sum / conf_weight) if conf_weight > 0 else None,
            "area_avg_cm2": (area_cm2_sum / area_cm2_count) if area_cm2_count > 0 else None,
            "area_max_cm2": area_cm2_max,
        }



@dataclass
class OperatorOverrides:
    trigger_class: str = DEFAULT_TRIGGER_CLASS
    block_enabled: bool = False
    action_choice: ActionChoice = "ignore"
    resume_mode: ResumeMode = "instant"
    resume_delay_sec: int = DEFAULT_RESUME_DELAY_SEC
    manual_enabled: bool = False
    manual_level: str = MANUAL_LEVELS[0]


DEFAULT_PROFILE_ID = "default"


def _overrides_to_dict(overrides: OperatorOverrides) -> dict[str, object]:
    return {
        "trigger_class": overrides.trigger_class,
        "block_enabled": overrides.block_enabled,
        "action_choice": overrides.action_choice,
        "resume_mode": overrides.resume_mode,
        "resume_delay_sec": overrides.resume_delay_sec,
        "manual_enabled": overrides.manual_enabled,
        "manual_level": overrides.manual_level,
    }


def _overrides_from_dict(payload: dict | None) -> OperatorOverrides:
    payload = payload or {}
    return OperatorOverrides(
        trigger_class=str(payload.get("trigger_class", DEFAULT_TRIGGER_CLASS)).strip(),
        block_enabled=bool(payload.get("block_enabled", False)),
        action_choice=_cast_action(str(payload.get("action_choice", "ignore"))),
        resume_mode=_cast_resume(str(payload.get("resume_mode", "instant"))),
        resume_delay_sec=max(1, int(payload.get("resume_delay_sec", DEFAULT_RESUME_DELAY_SEC) or 0)),
        manual_enabled=bool(payload.get("manual_enabled", False)),
        manual_level=str(payload.get("manual_level", MANUAL_LEVELS[0]))
        if str(payload.get("manual_level", MANUAL_LEVELS[0])) in MANUAL_LEVELS
        else MANUAL_LEVELS[0],
    )


@dataclass
class ProfileData:
    profile_id: str
    name: str
    overrides: OperatorOverrides = field(default_factory=OperatorOverrides)
    rules: list[dict[str, object]] = field(default_factory=list)
    actions: list[dict[str, object]] = field(default_factory=list)
    conditions: list[dict[str, object]] = field(default_factory=list)

    def to_payload(self, include_id: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "overrides": _overrides_to_dict(self.overrides),
            "rules": copy.deepcopy(self.rules),
            "actions": copy.deepcopy(self.actions),
            "conditions": copy.deepcopy(self.conditions),
        }
        if include_id:
            data["profile_id"] = self.profile_id
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProfileData":
        profile_id = str(payload.get("profile_id", DEFAULT_PROFILE_ID))
        name = str(payload.get("name", "Perfil")) or "Perfil"
        overrides_payload = payload.get("overrides") if isinstance(payload.get("overrides"), dict) else None
        rules_payload = payload.get("rules") if isinstance(payload.get("rules"), list) else []
        actions_payload = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        conditions_payload = payload.get("conditions") if isinstance(payload.get("conditions"), list) else []
        return cls(
            profile_id=profile_id,
            name=name,
            overrides=_overrides_from_dict(overrides_payload),
            rules=copy.deepcopy(rules_payload),
            actions=copy.deepcopy(actions_payload),
            conditions=copy.deepcopy(conditions_payload),
        )


class ProfileManager:
    def __init__(self, config_path: Path | None) -> None:
        self.path = Path(config_path) if config_path else None
        self.active_profile_id: str = DEFAULT_PROFILE_ID
        self._profiles: dict[str, ProfileData] = {}
        self._plc_presets: dict[str, PLCPreset] = {}
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            self._profiles[DEFAULT_PROFILE_ID] = self._build_default_profile()
            self.active_profile_id = DEFAULT_PROFILE_ID
            self._dirty = True
            return

        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudo cargar configuración de perfiles: %s", exc)
            self._profiles[DEFAULT_PROFILE_ID] = self._build_default_profile()
            self._plc_presets = {PLC_DEFAULT_PRESET_ID: self._build_default_preset()}
            self.active_profile_id = DEFAULT_PROFILE_ID
            self._dirty = True
            return

        # Compatibilidad con formato antiguo (flat overrides)
        if not isinstance(data, dict) or "profiles" not in data:
            overrides = _overrides_from_dict(data if isinstance(data, dict) else {})
            self._profiles[DEFAULT_PROFILE_ID] = ProfileData(
                profile_id=DEFAULT_PROFILE_ID,
                name="Perfil por defecto",
                overrides=overrides,
            )
            self.active_profile_id = DEFAULT_PROFILE_ID
            self._dirty = True
            return

        profiles_section = data.get("profiles")
        if isinstance(profiles_section, dict):
            for profile_id, payload in profiles_section.items():
                if not isinstance(payload, dict):
                    continue
                merged_payload = {"profile_id": profile_id, **payload}
                try:
                    profile = ProfileData.from_dict(merged_payload)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Perfil ignorado por formato inválido (%s): %s", profile_id, exc)
                    continue
                self._profiles[profile.profile_id] = profile

        self.active_profile_id = str(data.get("active_profile_id", DEFAULT_PROFILE_ID))
        if self.active_profile_id not in self._profiles and self._profiles:
            self.active_profile_id = next(iter(self._profiles.keys()))

        presets_section = data.get(PLC_PRESETS_KEY)
        if isinstance(presets_section, list):
            for item in presets_section:
                if not isinstance(item, dict):
                    continue
                try:
                    preset = PLCPreset.from_payload(item)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Preset PLC ignorado por formato inválido: %s", exc)
                    continue
                self._plc_presets[preset.preset_id] = preset

        if not self._plc_presets:
            self._plc_presets = {PLC_DEFAULT_PRESET_ID: self._build_default_preset()}

        if not self._profiles:
            self._profiles[DEFAULT_PROFILE_ID] = self._build_default_profile()
            self.active_profile_id = DEFAULT_PROFILE_ID
            self._dirty = True

    # ------------------------------------------------------------------
    def _build_default_profile(self) -> ProfileData:
        return ProfileData(profile_id=DEFAULT_PROFILE_ID, name="Perfil por defecto")

    def _build_default_preset(self) -> PLCPreset:
        return PLCPreset(
            preset_id=PLC_DEFAULT_PRESET_ID,
            name="PLC principal",
            description="Preset inicial",
            ip="",
            rack=0,
            slot=2,
            area="M",
            db_number=None,
            default_byte=0,
            default_bit=0,
        )

    # ------------------------------------------------------------------
    def save(self) -> None:
        if self.path is None:
            self._dirty = False
            return
        day = datetime.utcnow().strftime("%Y%m%d")
        backup_path = self.path.with_name(f"{self.path.stem}_{day}.bak")
        data = {
            "active_profile_id": self.active_profile_id,
            "profiles": {profile_id: profile.to_payload() for profile_id, profile in self._profiles.items()},
            PLC_PRESETS_KEY: [preset.to_payload() for preset in self._plc_presets.values()],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                try:
                    self.path.replace(backup_path)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("No se pudo crear backup de perfiles: %s", exc)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudo guardar configuración de perfiles: %s", exc)

    # ------------------------------------------------------------------
    def save_if_dirty(self) -> None:
        if self._dirty:
            self.save()

    # ------------------------------------------------------------------
    def get_active_profile(self) -> ProfileData:
        return self._profiles[self.active_profile_id]

    def get_active_overrides(self) -> dict[str, object]:
        return copy.deepcopy(self.get_active_profile().overrides)

    def update_active_overrides(self, overrides: dict[str, object]) -> bool:
        profile = self.get_active_profile()
        if profile.overrides == overrides:
            return False
        profile.overrides = copy.deepcopy(overrides)
        self._profiles[profile.profile_id] = profile
        self._dirty = True
        return True

    # ------------------------------------------------------------------
    def list_profiles_payload(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for profile_id, profile in self._profiles.items():
            payload = profile.to_payload(include_id=True)
            payload["is_active"] = profile_id == self.active_profile_id
            items.append(payload)
        return items

    def set_active_profile(self, profile_id: str) -> ProfileData:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        self.active_profile_id = profile_id
        self._dirty = True
        return self._profiles[profile_id]

    def create_profile(self, name: str, base_profile_id: str | None = None) -> ProfileData:
        base = self._profiles.get(base_profile_id or self.active_profile_id)
        new_id = self._generate_profile_id(name)
        safe_name = name.strip() or f"Perfil {len(self._profiles) + 1}"
        profile = ProfileData(
            profile_id=new_id,
            name=safe_name,
            overrides=copy.deepcopy(base.overrides) if base else OperatorOverrides(),
            rules=copy.deepcopy(base.rules) if base else [],
            actions=copy.deepcopy(base.actions) if base else [],
            conditions=copy.deepcopy(base.conditions) if base else [],
        )
        self._profiles[new_id] = profile
        self._dirty = True
        return profile

    def delete_profile(self, profile_id: str) -> None:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        if len(self._profiles) == 1:
            raise ValueError("No se puede eliminar el único perfil disponible")
        self._profiles.pop(profile_id)
        if self.active_profile_id == profile_id:
            self.active_profile_id = next(iter(self._profiles.keys()))
        self._dirty = True

    def rename_profile(self, profile_id: str, new_name: str) -> ProfileData:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        profile = self._profiles[profile_id]
        safe_name = new_name.strip() or profile.name
        if profile.name != safe_name:
            profile.name = safe_name
            self._dirty = True
        return profile

    def replace_profile_actions(self, profile_id: str, actions: Iterable[dict[str, object]]) -> None:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        profile = self._profiles[profile_id]
        profile.actions = [copy.deepcopy(action) for action in actions if isinstance(action, dict)]
        self._dirty = True

    def replace_profile_conditions(self, profile_id: str, conditions: Iterable[dict[str, object]]) -> None:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        profile = self._profiles[profile_id]
        profile.conditions = [copy.deepcopy(item) for item in conditions if isinstance(item, dict)]
        self._dirty = True

    def replace_profile_rules(self, profile_id: str, rules: Iterable[dict[str, object]]) -> None:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        profile = self._profiles[profile_id]
        profile.rules = [copy.deepcopy(rule) for rule in rules if isinstance(rule, dict)]
        self._dirty = True

    def get_profile_rules(self, profile_id: str) -> list[dict[str, object]]:
        if profile_id not in self._profiles:
            raise KeyError(f"Perfil '{profile_id}' no encontrado")
        return [copy.deepcopy(rule) for rule in self._profiles[profile_id].rules]

    def get_active_rules(self) -> list[dict[str, object]]:
        return self.get_profile_rules(self.active_profile_id)

    # ------------------------------------------------------------------
    def _generate_profile_id(self, base_name: str) -> str:
        seed = "".join(ch for ch in base_name.lower() if ch.isalnum())
        if not seed:
            seed = "perfil"
        candidate = seed
        suffix = 1
        while candidate in self._profiles:
            suffix += 1
            candidate = f"{seed}{suffix}"
        return candidate

    # ------------------------------------------------------------------
    # Gestión de presets PLC
    # ------------------------------------------------------------------

    def list_plc_presets(self) -> list[dict[str, object]]:
        return [preset.to_payload() for preset in self._plc_presets.values()]

    def get_plc_preset(self, preset_id: str) -> PLCPreset:
        if preset_id not in self._plc_presets:
            raise KeyError(f"Preset '{preset_id}' no encontrado")
        return self._plc_presets[preset_id]

    def upsert_plc_preset(self, payload: dict[str, Any]) -> PLCPreset:
        preset = PLCPreset.from_payload(payload)
        self._plc_presets[preset.preset_id] = preset
        self._dirty = True
        return preset

    def delete_plc_preset(self, preset_id: str) -> None:
        if preset_id not in self._plc_presets:
            raise KeyError(f"Preset '{preset_id}' no encontrado")
        if preset_id == PLC_DEFAULT_PRESET_ID:
            raise ValueError("No se puede eliminar el preset por defecto")
        del self._plc_presets[preset_id]
        self._dirty = True

    def resolve_plc_action(self, payload: dict[str, object]) -> PLCAction:
        params = dict(payload)
        preset_id_raw = params.get("preset_id")
        preset_id = str(preset_id_raw).strip() if preset_id_raw else PLC_DEFAULT_PRESET_ID
        preset = self._plc_presets.get(preset_id)
        if preset is None:
            LOGGER.warning("Preset PLC '%s' no encontrado. Se usará el preset por defecto.", preset_id)
            preset = self._build_default_preset()
            preset_id = preset.preset_id

        def _override(key: str, default: object) -> object:
            value = params.get(key)
            if value in {None, ""}:
                return default
            return value

        ip = str(_override("ip", preset.ip) or "")
        area = str(_override("area", preset.area) or "M").upper()
        rack = int(_override("rack", preset.rack) or 0)
        slot = int(_override("slot", preset.slot) or 0)
        db_number_raw = _override("db_number", preset.db_number)
        db_number = None if db_number_raw in {None, ""} else int(db_number_raw)

        byte_index = params.get("byte_index", preset.default_byte)
        bit_index = params.get("bit_index", preset.default_bit)
        try:
            byte_index_int = int(byte_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("byte_index inválido en acción PLC") from exc
        try:
            bit_index_int = int(bit_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("bit_index inválido en acción PLC") from exc

        value_raw = params.get("value", True)
        value = bool(value_raw)
        tag = str(params.get("tag", "") or "")

        targets: list[PLCTarget] = []
        targets_payload = params.get("targets")
        if isinstance(targets_payload, list):
            for item in targets_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    target_byte = int(item.get("byte_index"))
                    target_bit = int(item.get("bit_index"))
                except (TypeError, ValueError):
                    LOGGER.warning("Objetivo PLC ignorado por índices inválidos: %s", item)
                    continue
                if target_bit < 0 or target_bit > 7:
                    LOGGER.warning("Objetivo PLC ignorado por bit fuera de rango: %s", item)
                    continue
                target_value = item.get("value", value)
                try:
                    target_bool = bool(target_value) if isinstance(target_value, bool) else bool(int(target_value))
                except (TypeError, ValueError):
                    LOGGER.warning("Objetivo PLC ignorado por valor inválido: %s", item)
                    continue
                targets.append(
                    PLCTarget(
                        byte_index=target_byte,
                        bit_index=target_bit,
                        value=target_bool,
                    )
                )

        return PLCAction(
            preset_id=preset_id,
            ip=ip,
            rack=rack,
            slot=slot,
            area=area,
            db_number=db_number,
            byte_index=byte_index_int,
            bit_index=bit_index_int,
            value=value,
            tag=tag,
            targets=targets,
        )

@dataclass
class PLCOrder:
    trigger_active: bool
    block_decisions: bool
    action_choice: ActionChoice
    force_level: Optional[str]
    resume_mode: ResumeMode
    resume_delay_sec: int
    resume_eta_sec: Optional[float]
    message: str
    plc_feedback: list[str] = field(default_factory=list)
    plc_checks: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class PLCTarget:
    byte_index: int
    bit_index: int
    value: bool


@dataclass
class PLCAction:
    preset_id: str
    ip: str
    rack: int
    slot: int
    area: str
    db_number: int | None
    byte_index: int
    bit_index: int
    value: bool
    tag: str = ""
    targets: list[PLCTarget] = field(default_factory=list)


@dataclass
class PLCPreset:
    preset_id: str
    name: str
    description: str
    ip: str
    rack: int
    slot: int
    area: str = "M"
    db_number: int | None = None
    default_byte: int = 0
    default_bit: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "description": self.description,
            "ip": self.ip,
            "rack": self.rack,
            "slot": self.slot,
            "area": self.area,
            "db_number": self.db_number,
            "default_byte": self.default_byte,
            "default_bit": self.default_bit,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PLCPreset":
        return cls(
            preset_id=str(payload.get("preset_id")) or uuid.uuid4().hex,
            name=str(payload.get("name", "Preset")) or "Preset",
            description=str(payload.get("description", "")),
            ip=str(payload.get("ip", "")),
            rack=int(payload.get("rack", 0) or 0),
            slot=int(payload.get("slot", 0) or 0),
            area=str(payload.get("area", "M")) or "M",
            db_number=(
                int(payload.get("db_number")) if payload.get("db_number") not in {None, ""} else None
            ),
            default_byte=int(payload.get("default_byte", 0) or 0),
            default_bit=int(payload.get("default_bit", 0) or 0),
        )


def _ping_tcp_102(ip: str, timeout: float = 2.0) -> bool:
    try:
        socket.create_connection((ip, 102), timeout=timeout).close()
        return True
    except Exception:
        try:
            param = "-n" if platform.system().lower() == "windows" else "-c"
            return (
                subprocess.call([
                    "ping",
                    param,
                    "1",
                    ip,
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                == 0
            )
        except Exception:
            return False


class PLCClient:
    """Cliente real de PLC con soporte para snap7 y modo simulación."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER
        self._client: Any | None = None
        self._lock = threading.Lock()
        if SNAP7_AVAILABLE:
            try:
                self._client = snap7.client.Client()
            except Exception as exc:  # pragma: no cover - error inusual
                self.logger.warning("No se pudo inicializar snap7.Client: %s", exc)
                self._client = None

    # ------------------------------------------------------------------
    def _ensure_connected(self, ip: str, rack: int, slot: int) -> bool:
        if not SNAP7_AVAILABLE or self._client is None:
            # Crear cliente en caliente si se ha vaciado tras un error.
            try:
                with self._lock:
                    if self._client is None:
                        self._client = snap7.client.Client()
            except Exception as exc:
                self.logger.warning("No se pudo crear snap7.Client: %s", exc)
                return False
        with self._lock:
            try:
                if self._client.get_connected():
                    return True
            except Exception:
                # Si falla, intentamos reconectar
                self._client = None

        if not _ping_tcp_102(ip):
            self.logger.warning("PLC %s: puerto 102 no accesible", ip)
            return False

        with self._lock:
            try:
                if self._client.get_connected():
                    return True
            except Exception:
                pass
            try:
                if self._client is None:
                    self._client = snap7.client.Client()
                self._client.connect(ip, rack, slot)
                return bool(self._client.get_connected())
            except Exception as exc:  # pragma: no cover - entorno sin PLC
                self.logger.error("No se pudo conectar al PLC %s: %s", ip, exc)
                self._client = None
                return False

    def _disconnect(self) -> None:
        if not SNAP7_AVAILABLE or self._client is None:
            return
        with self._lock:
            with suppress(Exception):
                if self._client.get_connected():
                    self._client.disconnect()
            # Forzar recreación en el siguiente uso para limpiar estados rotos.
            self._client = None

    # ------------------------------------------------------------------
    def send_action(self, action: PLCAction) -> tuple[bool, str]:
        """Escribe uno o varios bits en el PLC según el área indicada."""

        if not action.ip:
            return False, "IP no definida"

        area = action.area.upper()
        if area not in {"M", "DB", "Q", "I"}:
            return False, f"Área PLC desconocida: {action.area}"

        if area == "M" and action.byte_index < 0:
            return False, "Byte inválido"

        if action.bit_index < 0 or action.bit_index > 7:
            return False, "Bit debe estar entre 0 y 7"

        if area == "DB" and action.db_number is None:
            return False, "DB no definido"

        if not SNAP7_AVAILABLE or self._client is None:
            return False, "snap7 no disponible"

        if not self._ensure_connected(action.ip, action.rack, action.slot):
            return False, "No se pudo establecer conexión con el PLC"

        primary = PLCTarget(action.byte_index, action.bit_index, bool(action.value))
        targets = [primary, *action.targets]

        try:
            with self._lock:
                for target in targets:
                    if target.bit_index < 0 or target.bit_index > 7:
                        raise ValueError(f"Bit fuera de rango: {target.bit_index}")
                    if area == "M":
                        raw = self._client.read_area(Areas.MK, 0, target.byte_index, 1)
                        snap7_util.set_bool(raw, 0, target.bit_index, target.value)
                        self._client.write_area(Areas.MK, 0, target.byte_index, raw)
                    elif area == "DB":
                        raw = self._client.read_area(Areas.DB, action.db_number, target.byte_index, 1)
                        snap7_util.set_bool(raw, 0, target.bit_index, target.value)
                        self._client.write_area(Areas.DB, action.db_number, target.byte_index, raw)
                    elif area == "Q":
                        raw = self._client.read_area(Areas.PE, 0, target.byte_index, 1)
                        snap7_util.set_bool(raw, 0, target.bit_index, target.value)
                        self._client.write_area(Areas.PE, 0, target.byte_index, raw)
                    else:  # área I
                        raw = self._client.read_area(Areas.PA, 0, target.byte_index, 1)
                        snap7_util.set_bool(raw, 0, target.bit_index, target.value)
                        self._client.write_area(Areas.PA, 0, target.byte_index, raw)
        except ValueError as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover - errores snap7
            self.logger.error("Fallo al escribir en PLC (%s): %s", action.area, exc)
            self._disconnect()
            return False, f"Error PLC: {exc}"

        label = ", ".join(
            f"{area}{target.byte_index}.{target.bit_index}={int(target.value)}"
            for target in targets
        )
        return True, f"OK {label}"

    def read_bit(self, action: PLCAction, target: PLCTarget) -> tuple[bool, str]:
        if not SNAP7_AVAILABLE or self._client is None:
            return False, "snap7 no disponible"
        if not self._ensure_connected(action.ip, action.rack, action.slot):
            return False, "sin conexión"
        area = action.area.upper()
        if area == "DB" and action.db_number is None:
            return False, "DB no definido"
        try:
            with self._lock:
                if area == "M":
                    raw = self._client.read_area(Areas.MK, 0, target.byte_index, 1)
                elif area == "DB":
                    raw = self._client.read_area(Areas.DB, action.db_number, target.byte_index, 1)
                elif area == "Q":
                    raw = self._client.read_area(Areas.PE, 0, target.byte_index, 1)
                else:
                    raw = self._client.read_area(Areas.PA, 0, target.byte_index, 1)
                value = snap7_util.get_bool(raw, 0, target.bit_index)
                return True, str(int(value))
        except Exception as exc:  # pragma: no cover
            self._disconnect()
            return False, str(exc)

    def read_s7_value(self, ip: str, rack: int, slot: int, s7_address: str, data_type: str) -> object:
        if not SNAP7_AVAILABLE or self._client is None:
            return None
        if not self._ensure_connected(ip, rack, slot):
            return None
        
        try:
            # Parse address using utility
            info = plc_bit_writer.parse_s7_address(s7_address)
            area = info["area"]
            db_number = info["db_number"]
            offset = info["byte_offset"]
            
            # Determine size based on requested type vs inferred type
            d_type = data_type.upper().strip()
            size = 2
            if d_type == "BYTE":
                size = 1
            elif d_type in ("WORD", "INT"):
                size = 2
            elif d_type in ("DWORD", "DINT", "REAL"):
                size = 4
            
            with self._lock:
                if area == Areas.DB:
                    if db_number is None:
                         return None
                    raw = self._client.read_area(Areas.DB, db_number, offset, size)
                elif area == Areas.MK:
                    raw = self._client.read_area(Areas.MK, 0, offset, size)
                elif area == Areas.PE:
                    raw = self._client.read_area(Areas.PE, 0, offset, size)
                elif area == Areas.PA:
                    raw = self._client.read_area(Areas.PA, 0, offset, size)
                else:
                    return None

            if d_type == "BYTE":
                return snap7_util.get_byte(raw, 0)
            elif d_type == "WORD":
                return snap7_util.get_word(raw, 0)
            elif d_type == "DWORD":
                return snap7_util.get_dword(raw, 0)
            elif d_type == "INT":
                return snap7_util.get_int(raw, 0)
            elif d_type == "DINT":
                return snap7_util.get_dint(raw, 0)
            elif d_type == "REAL":
                return snap7_util.get_real(raw, 0)
            
            return None
        except Exception as exc:
            LOGGER.debug("Error leyendo S7 %s desde %s: %s", s7_address, ip, exc)
            self._disconnect()
            return None

    def emit(self, order: PLCOrder) -> None:
        """Compatibilidad con versiones anteriores: registra la orden recibida."""

        self.logger.info(
            "Orden PLC -> bloque=%s action=%s force=%s mensaje=%s",
            order.block_decisions,
            order.action_choice,
            order.force_level,
            order.message,
        )

    def close(self) -> None:
        self._disconnect()


class MockPLCClient(PLCClient):
    def __init__(self, *, logger: logging.Logger | None = None) -> None:  # pragma: no cover - usado en tests
        super().__init__(logger=logger)

    def send_action(self, action: PLCAction) -> tuple[bool, str]:  # noqa: D401 - hereda docstring
        label = f"{action.area}{action.byte_index}.{action.bit_index}"
        self.logger.info("Simulación PLC: set %s -> %s", label, int(action.value))
        return True, f"Simulado {label} -> {int(action.value)}"


class DecisionEngine:
    """Combina snapshot + overrides para construir la orden final."""

    def __init__(self) -> None:
        self._overrides = OperatorOverrides()
        self._last_order: Optional[PLCOrder] = None
        self._block_release_time: float = 0.0
        self._last_forced_level: str = MANUAL_LEVELS[0]

    def set_overrides(self, overrides: OperatorOverrides) -> None:
        self._overrides = overrides
        if not overrides.block_enabled:
            self._block_release_time = 0.0
        if overrides.manual_enabled and overrides.manual_level:
            self._last_forced_level = overrides.manual_level

    def get_overrides(self) -> OperatorOverrides:
        return self._overrides

    def evaluate(self, snapshot: SnapshotState | None) -> PLCOrder:
        order = PLCOrder(
            trigger_active=False,
            block_decisions=False,
            action_choice="ignore",
            force_level=None,
            resume_mode="instant",
            resume_delay_sec=0,
            resume_eta_sec=None,
            message="",
        )
        self._last_order = order
        return order

    @property
    def last_order(self) -> Optional[PLCOrder]:
        return self._last_order

    def build_idle_order(self) -> PLCOrder:
        return PLCOrder(
            trigger_active=False,
            block_decisions=False,
            action_choice="ignore",
            force_level=None,
            resume_mode="instant",
            resume_delay_sec=0,
            resume_eta_sec=None,
            message="",
        )


class SnapshotWatcher(threading.Thread):
    """Hilo que monitoriza el archivo snapshot."""

    def __init__(self, path: Path, interval: float, callback: Callable[[SnapshotState | None], None]) -> None:
        super().__init__(daemon=True)
        self.path = Path(path)
        self.interval = interval
        self.callback = callback
        self._stop_event = threading.Event()
        self._last_mtime: Optional[float] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # noqa: D401 - descripción heredada
        while not self._stop_event.is_set():
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self.callback(None)
                time.sleep(self.interval)
                continue

            if self._last_mtime is None or stat.st_mtime != self._last_mtime:
                self._last_mtime = stat.st_mtime
                snapshot: SnapshotState | None
                try:
                    with self.path.open("r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    snapshot = SnapshotState.from_payload(payload)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("No se pudo leer snapshot: %s", exc)
                    snapshot = None
                self.callback(snapshot)
            time.sleep(self.interval)


class PLCConditionPoller(threading.Thread):
    """Hilo que mantiene las lecturas de bits PLC actualizadas con caching compartido."""

    def __init__(
        self,
        plc_client: "PLCClient",
        *,
        poll_interval: float,
        ttl: float,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._plc_client = plc_client
        self._poll_interval = max(0.05, float(poll_interval))
        self._ttl = max(0.1, float(ttl))
        self._logger = logger or LOGGER
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pending: set[tuple] = set()
        self._cache: dict[tuple, tuple[float, bool | None]] = {}
        self._last_known: dict[tuple, tuple[float, bool | None]] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # noqa: D401 - descripción heredada
        while not self._stop_event.is_set():
            self._poll_once()
            self._expire()
            time.sleep(self._poll_interval)

    def request(self, action: "PLCAction") -> bool | None:
        key = self._build_key(action)
        now = time.monotonic()
        with self._lock:
            record = self._cache.get(key)
            if record is None:
                fallback = self._last_known.get(key)
                self._pending.add(key)
                if fallback is None:
                    return None
                return fallback[1]
            ts, value = record
            if now - ts > self._ttl:
                self._pending.add(key)
            return value

    def _poll_once(self) -> None:
        snapshot: list[tuple] = []
        with self._lock:
            if not self._pending:
                return
            snapshot = [key for key in self._pending]
            self._pending.clear()

        for key in snapshot:
            ip, rack, slot, area, db_number, byte_index, bit_index = key
            action = PLCAction(
                preset_id="",
                ip=ip,
                rack=rack,
                slot=slot,
                area=area,
                db_number=db_number,
                byte_index=byte_index,
                bit_index=bit_index,
                value=False,
                tag="",
                targets=(),
            )
            target = PLCTarget(byte_index, bit_index, True)
            try:
                success, raw = self._plc_client.read_bit(action, target)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("Lectura PLC (poller) falló: %s", exc)
                self._store(key, None)
                continue
            if not success:
                self._logger.debug(
                    "Lectura PLC fallida (poller %s%s.%s): %s",
                    area,
                    byte_index,
                    bit_index,
                    raw,
                )
                self._store(key, None)
                continue
            value_str = str(raw).strip().lower()
            actual = value_str in {"1", "true", "on", "yes"}
            self._store(key, actual)

    def _store(self, key: tuple, value: bool | None) -> None:
        with self._lock:
            record = (time.monotonic(), value)
            self._cache[key] = record
            self._last_known[key] = record

    def _expire(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [key for key, (ts, _) in self._cache.items() if now - ts > self._ttl]
            for key in expired:
                self._cache.pop(key, None)
            stale_limit = max(self._ttl * 5, self._ttl)
            very_old = [key for key, (ts, _) in self._last_known.items() if now - ts > stale_limit]
            for key in very_old:
                self._last_known.pop(key, None)

    def _build_key(self, action: "PLCAction") -> tuple:
        return (
            action.ip,
            action.rack,
            action.slot,
            action.area,
            action.db_number,
            action.byte_index,
            action.bit_index,
        )


class SendToPLCService:
    _PLC_CACHE_MISS: ClassVar[object] = object()
    _ACTION_NOT_FOUND: ClassVar[object] = object()

    """Servicio en segundo plano encargado de evaluar y emitir órdenes."""

    def __init__(
        self,
        snapshot_path: Path | str | None = SNAPSHOT_PATH,
        *,
        config_path: Path | str | None = CONFIG_PATH,
        persist_snapshots: Optional[bool] = None,
        rules_path: Path | str | None = RULES_PATH,
        capture_dir: Path | str | None = None,
    ) -> None:
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.config_path = Path(config_path) if config_path else None
        self.persist_snapshots = bool(self.snapshot_path) if persist_snapshots is None else bool(persist_snapshots)
        self.rules_path = Path(rules_path) if rules_path else None
        self.capture_dir = Path(capture_dir) if capture_dir else None

        LOGGER.info(
            "Inicializando SendToPLCService (snapshot=%s, config=%s, rules=%s, capture_dir=%s)",
            self.snapshot_path,
            self.config_path,
            self.rules_path,
            self.capture_dir,
        )
        if self.snapshot_path is not None and not self.snapshot_path.exists():
            LOGGER.debug("La ruta de snapshot todavía no existe: %s", self.snapshot_path)
        if self.config_path is not None and not self.config_path.exists():
            LOGGER.debug("La configuración sendToPLC no existe, se creará: %s", self.config_path)
        if self.rules_path is not None and not self.rules_path.exists():
            LOGGER.debug("El fichero de reglas no existe, se generará si es necesario: %s", self.rules_path)

        self.plc_client = PLCClient()
        self._snapshot: SnapshotState | None = None
        self._base_metadata: dict[str, object] = {}
        self._last_metadata: dict[str, object] = {}
        self._plc_condition_cache: dict[tuple, tuple[float, bool | None]] = {}
        self._plc_condition_cache_ttl = PLC_CONDITION_CACHE_TTL_SEC
        self._plc_numeric_cache: dict[tuple, tuple[float, object]] = {}
        self._plc_condition_action_cache: dict[tuple, object] = {}
        self._plc_poller = PLCConditionPoller(
            self.plc_client,
            poll_interval=PLC_CONDITION_POLL_INTERVAL_SEC,
            ttl=PLC_CONDITION_CACHE_TTL_SEC,
        )
        self._vision_diag_recent: deque[dict[str, object]] = deque(maxlen=200)
        self._vision_diag_counters: Counter[str] = Counter()
        self._vision_diag_lock = threading.Lock()
        try:
            self.rule_engine = RuleEngine(
                rules_path=self.rules_path,
                plc_condition_provider=self._evaluate_plc_condition,
                vision_diag_callback=self._handle_vision_diag,
            )
        except TypeError:
            LOGGER.debug("RuleEngine no acepta proveedor PLC; se usará configuración por defecto")
            self.rule_engine = RuleEngine(rules_path=self.rules_path)
        self._rule_evaluation: RuleEvaluation | None = None
        self._rule_effects = RuleEffects()
        self._muted_rules: dict[str, float] = {}
        self.engine = DecisionEngine()
        self._order: PLCOrder | None = None
        self.profile_manager = ProfileManager(self.config_path)
        active_rules_payload = self.profile_manager.get_active_rules()
        if active_rules_payload:
            rules: list[Rule] = []
            for item in active_rules_payload:
                if isinstance(item, dict):
                    try:
                        rules.append(Rule.from_dict(item))
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Regla ignorada por formato inválido en perfil activo '%s': %s",
                            self.profile_manager.active_profile_id,
                            exc,
                        )
            if rules:
                self.rule_engine.update_rules(rules)
        else:
            existing_rules = [rule.to_dict() for rule in self.rule_engine.get_rules()]
            if existing_rules:
                self.profile_manager.replace_profile_rules(self.profile_manager.active_profile_id, existing_rules)
                self.profile_manager.save_if_dirty()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watcher = (
            SnapshotWatcher(self.snapshot_path, SNAPSHOT_POLL_INTERVAL_SEC, self._on_snapshot)
            if self.snapshot_path is not None
            else None
        )
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._subscribers: list[Callable[[SnapshotState | None, PLCOrder], None]] = []
        self._event_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=32)
        self._rule_check_states: dict[str, dict[str, object]] = {}
        self._snapshot_cooldowns: dict[str, float] = {}
        if self.capture_dir is not None:
            try:
                self.capture_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudo crear carpeta de capturas: %s", exc)
                self.capture_dir = None

    def start(self) -> None:
        if self._watcher is not None and not self._watcher.is_alive():
            self._watcher.start()
        if not self._loop_thread.is_alive():
            self._stop_event.clear()
            self._loop_thread.start()
        if not self._plc_poller.is_alive():
            self._plc_poller.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._watcher is not None:
            self._watcher.stop()
            if self._watcher.is_alive():
                self._watcher.join(timeout=1.0)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
        if self._plc_poller.is_alive():
            self._plc_poller.stop()
            self._plc_poller.join(timeout=1.0)
        logging.info("SendToPLCService detenido.")

    def _on_snapshot(self, snapshot: SnapshotState | None) -> None:
        with self._lock:
            self._snapshot = snapshot
        if snapshot is None:
            LOGGER.debug("Snapshot recibido: ninguno (se reprograma)")
            return
        summary = _summarize_snapshot(snapshot)
        LOGGER.debug("Snapshot recibido: %s", summary)

    # ...
    def push_snapshot(self, payload: SnapshotState | dict | None, *, persist: Optional[bool] = None) -> None:
        """Permite inyectar un snapshot directamente sin depender de archivos."""

        if payload is None:
            snapshot = None
        elif isinstance(payload, SnapshotState):
            snapshot = payload
        elif isinstance(payload, dict):
            snapshot = SnapshotState.from_payload(payload)
        else:
            raise TypeError("payload debe ser dict, SnapshotState o None")

        with self._lock:
            self._snapshot = snapshot
            summary = _summarize_snapshot(snapshot) if snapshot is not None else {"snapshot": None}
            LOGGER.debug("Procesando snapshot inyectado: %s", summary)
            evaluation = self._evaluate_rules(snapshot)
            self._rule_evaluation = evaluation
            self._rule_effects = evaluation.effects
            self._apply_muted_triggers_effects(evaluation.effects)

        persist_snapshot = self.persist_snapshots if persist is None else bool(persist)
        if persist_snapshot and self.snapshot_path is not None and snapshot is not None:
            try:
                self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                with self.snapshot_path.open("w", encoding="utf-8") as fh:
                    json.dump(snapshot.raw, fh, ensure_ascii=False, indent=2)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudo persistir snapshot: %s", exc)

        self._notify_state()

    def _handle_vision_diag(self, payload: dict[str, object]) -> None:
        reason = str(payload.get("reason", "")).strip().lower()
        with self._vision_diag_lock:
            self._vision_diag_recent.append(payload)
            count = 0
            if reason:
                self._vision_diag_counters[reason] += 1
                count = self._vision_diag_counters[reason]
        if reason and reason not in {"", "ok"} and count in {1, 5, 20, 50}:
            LOGGER.info(
                "Diagnóstico visión: reason=%s ocurrencias=%s últimos=%s",
                reason,
                count,
                {
                    "class": payload.get("class_name"),
                    "count": payload.get("actual_count"),
                    "min_count": payload.get("min_count"),
                    "metric": payload.get("metric"),
                    "min_area": payload.get("min_area"),
                    "conf_avg": payload.get("conf_avg"),
                    "min_conf": payload.get("min_conf"),
                },
            )

    def set_overrides(self, overrides: OperatorOverrides) -> None:
        normalized = OperatorOverrides(
            trigger_class=overrides.trigger_class.strip(),
            block_enabled=overrides.block_enabled,
            action_choice=_cast_action(overrides.action_choice),
            resume_mode=_cast_resume(overrides.resume_mode),
            resume_delay_sec=max(1, int(overrides.resume_delay_sec)),
            manual_enabled=overrides.manual_enabled,
            manual_level=overrides.manual_level if overrides.manual_level in MANUAL_LEVELS else MANUAL_LEVELS[0],
        )
        with self._lock:
            self.engine.set_overrides(normalized)
            if self.profile_manager.update_active_overrides(normalized):
                self.profile_manager.save_if_dirty()

    def get_overrides(self) -> OperatorOverrides:
        with self._lock:
            return self.engine.get_overrides()

    def get_state(self) -> tuple[SnapshotState | None, PLCOrder | None]:
        with self._lock:
            return self._snapshot, self._order

    def get_pending_events(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        while True:
            try:
                items.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def get_model_metadata(self) -> dict[str, object]:
        with self._lock:
            target = self._last_metadata or self._base_metadata
            return copy.deepcopy(target)

    def update_model_metadata(self, metadata: dict | None) -> None:
        sanitized = _sanitize_metadata(metadata)
        with self._lock:
            self._base_metadata = copy.deepcopy(sanitized)
            self._last_metadata = copy.deepcopy(sanitized)
        classes = sanitized.get("classes")
        if isinstance(classes, (list, tuple, set)):
            self.rule_engine.set_available_classes(classes)
        self._notify_state()

    def _model_metadata_with_rules(self) -> dict[str, object]:
        with self._lock:
            # Optimización: Solo copiar lo necesario
            data = self._base_metadata.copy() if isinstance(self._base_metadata, dict) else {}
            # Si queremos ser muy estrictos con la inmutabilidad de los valores internos,
            # podríamos necesitar deepcopy, pero para la UI un shallow copy del dict suele bastar 
            # ya que solo leemos y el servicio reemplaza el dict entero en update_model_metadata.
            data["muted_rules"] = self._muted_rules_summary()
            return data

    def _broadcast_metadata(self, metadata: dict[str, object]) -> None:
        with self._lock:
            self._last_metadata = copy.deepcopy(metadata)

    def subscribe(self, callback: Callable[[SnapshotState | None, PLCOrder], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    def _notify(self, snapshot: SnapshotState | None, order: PLCOrder) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for listener in listeners:
            try:
                listener(snapshot, copy.deepcopy(order))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Listener SendToPLC falló: %s", exc)

    def _notify_state(self) -> None:
        with self._lock:
            snapshot = self._snapshot
            order = self._order
        detector_ready = is_detector_started()
        if detector_ready:
            evaluation: RuleEvaluation | None = None
            if order is None:
                evaluation = self._evaluate_rules(snapshot)
                self._apply_muted_triggers_effects(evaluation.effects)
                order = self.engine.evaluate(snapshot)
                trigger_active = bool(getattr(evaluation, "matches", None))
                if trigger_active:
                    order.trigger_active = True
                order = self._apply_rule_effects_to_order(order, evaluation.effects)
                self._apply_effects_side_actions(evaluation.effects, trigger_active=trigger_active)
                with self._lock:
                    self._order = order
                    self._rule_evaluation = evaluation
                    self._rule_effects = evaluation.effects
        else:
            order = self._manually_notify_idle()
        metadata = self._model_metadata_with_rules()
        if detector_ready and isinstance(order, PLCOrder):
            active_eval = evaluation if evaluation is not None else self._rule_evaluation
            if active_eval:
                metadata["active_rules"] = [
                    str(getattr(match, "rule_id", ""))
                    for match in active_eval.matches
                    if getattr(match, "rule_id", None)
                ]
        self._broadcast_metadata(metadata)
        self._notify(snapshot, order)

    def _drain_events(self) -> None:
        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    def _enqueue_event(self, event: dict[str, object]) -> None:
        if not event:
            return
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                LOGGER.debug("Cola de eventos llena, evento descartado: %s", event.get("type"))

    def _apply_effects_side_actions(self, effects: RuleEffects, *, trigger_active: bool) -> None:
        if not effects.overlay_messages and not effects.snapshot_requests:
            return
        timestamp = datetime.utcnow().isoformat()
        for message in effects.overlay_messages:
            payload = {
                "text": str(message.get("text", "")),
                "color": str(message.get("color", "#ffbc00")),
                "duration_ms": int(message.get("duration_ms", 4000) or 4000),
            }
            if "opacity" in message:
                payload["opacity"] = float(message.get("opacity", 0.8))
            self._enqueue_event({
                "type": "overlay_message",
                "ts": timestamp,
                "payload": payload,
            })

        for request in effects.snapshot_requests:
            if not self._should_take_snapshot(request, trigger_active=trigger_active):
                continue
            file_path = self._build_capture_filepath(request.get("label"))
            if not file_path:
                continue
            payload = {
                "file_path": file_path,
                "label": request.get("label", ""),
                "annotate": bool(request.get("annotate", False)),
            }
            self._enqueue_event({
                "type": "snapshot_request",
                "ts": timestamp,
                "payload": payload,
            })
            self._register_snapshot_event(request)

    def _build_capture_filepath(self, label: object) -> str | None:
        if self.capture_dir is None:
            return None
        try:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudo preparar carpeta de capturas: %s", exc)
            return None
        suffix = ""
        if isinstance(label, str):
            suffix = self._sanitize_label(label)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{stamp}.png" if not suffix else f"{stamp}_{suffix}.png"
        return str(self.capture_dir / filename)

    @staticmethod
    def _sanitize_label(label: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
        cleaned = cleaned.strip("._-")
        return cleaned[:40]

    def _snapshot_key(self, request: dict[str, object]) -> str:
        label = str(request.get("label", "")) if isinstance(request.get("label"), str) else ""
        key = label.strip()
        return key or "__default__"

    def _should_take_snapshot(self, request: dict[str, object], *, trigger_active: bool) -> bool:
        if not _ensure_bool(request.get("enabled", True), default=True):
            return False
        if _ensure_bool(request.get("require_trigger", True), default=True) and not trigger_active:
            return False
        cooldown_sec = _ensure_float(request.get("cooldown_sec"))
        if cooldown_sec is None or cooldown_sec <= 0:
            return True
        key = self._snapshot_key(request)
        last = self._snapshot_cooldowns.get(key)
        now = time.monotonic()
        if last is not None and now - last < cooldown_sec:
            return False
        return True

    def _register_snapshot_event(self, request: dict[str, object]) -> None:
        cooldown_sec = _ensure_float(request.get("cooldown_sec"))
        if cooldown_sec is None or cooldown_sec <= 0:
            return
        key = self._snapshot_key(request)
        self._snapshot_cooldowns[key] = time.monotonic()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._expire_plc_condition_cache()
            self._expire_plc_numeric_cache()
            self._expire_muted_rules()
            with self._lock:
                snapshot = self._snapshot
            if is_detector_started():
                evaluation = self._evaluate_rules(snapshot)
                self._apply_muted_triggers_effects(evaluation.effects)
                trigger_active = bool(getattr(evaluation, "matches", None))
                order = self.engine.evaluate(snapshot)
                if trigger_active:
                    order.trigger_active = True
                order = self._apply_rule_effects_to_order(order, evaluation.effects)
                self._apply_effects_side_actions(evaluation.effects, trigger_active=trigger_active)
                with self._lock:
                    self._order = order
                    self._rule_evaluation = evaluation
                    self._rule_effects = evaluation.effects
                try:
                    self.plc_client.emit(order)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Fallo al emitir orden al PLC: %s", exc)
                metadata = self._model_metadata_with_rules()
                metadata["active_rules"] = [
                    str(getattr(match, "rule_id", "")) for match in evaluation.matches if getattr(match, "rule_id", None)
                ]
                self._broadcast_metadata(metadata)
                self._notify(snapshot, order)
            else:
                idle_order = self._manually_notify_idle()
                metadata = self._model_metadata_with_rules()
                self._broadcast_metadata(metadata)
                self._notify(snapshot, idle_order)
            try:
                time.sleep(SNAPSHOT_POLL_INTERVAL_SEC)
            except Exception:
                self._notify(snapshot, order)

    def _manually_notify_idle(self) -> PLCOrder:
        idle_order = self.engine.build_idle_order()
        with self._lock:
            self._order = idle_order
            self._rule_evaluation = None
            self._rule_effects = RuleEffects()
        self._drain_events()
        return idle_order

    def _evaluate_rules(self, snapshot: SnapshotState | None) -> RuleEvaluation:
        summary = _summarize_snapshot(snapshot) if snapshot is not None else {"snapshot": None}
        LOGGER.debug("Evaluando reglas con snapshot: %s", summary)
        try:
            if hasattr(self.rule_engine, "set_muted_rules"):
                try:
                    self.rule_engine.set_muted_rules(self._active_muted_rule_ids())
                except Exception:
                    pass
            evaluation = self.rule_engine.process(snapshot)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Fallo al evaluar reglas: %s", exc)
            evaluation = RuleEvaluation(effects=RuleEffects(), matches=[])
        LOGGER.debug(
            "Resultado evaluación: matches=%s efectos=%s",
            [getattr(rule, "rule_id", "?") for rule in evaluation.matches],
            _effects_to_dict(evaluation.effects),
        )
        return evaluation

    def _evaluate_plc_condition(self, condition: dict[str, object]) -> bool:
        mode = str(condition.get("plc_mode", "bit")).strip().lower()
        if mode == "numeric":
            return self._evaluate_plc_numeric(condition)

        LOGGER.debug("Evaluando condición PLC cruda: %s", condition)
        normalized = _normalize_plc_condition(condition)
        LOGGER.debug("Condición PLC normalizada: %s", normalized)
        try:
            action = self._resolve_plc_condition_action(normalized)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Condición PLC inválida: %s", exc)
            return False
        if action is None:
            return False

        cache_key = self._build_plc_cache_key(action)
        cached = self._get_cached_plc_value(cache_key)
        if cached is not self._PLC_CACHE_MISS:
            if cached is None:
                return False
            expected = bool(normalized.get("expected_value", True))
            return cached is expected

        async_value = self._plc_poller.request(action)
        if async_value is None:
            return False

        self._store_plc_cache_value(cache_key, async_value)
        expected = bool(normalized.get("expected_value", True))
        return bool(async_value) is expected

    def _evaluate_plc_numeric(self, condition: dict[str, object]) -> bool:
        # 1. Resolver conexión y parámetros de forma centralizada
        try:
            action = self._resolve_plc_condition_action(condition)
        except Exception as exc:
            LOGGER.warning("No se pudo resolver conexión para numérico: %s", exc)
            return False
            
        if action is None or not action.ip:
            return False

        address = str(condition.get("address", "")).strip()
        dtype = str(condition.get("data_type", "WORD")).strip()
        if not address:
            return False

        # 2. Consultar caché de valor crudo (para evitar múltiples lecturas S7 iguales en el mismo ciclo)
        cache_key = (action.ip, action.rack, action.slot, address, dtype)
        now = time.monotonic()
        if cache_key in self._plc_numeric_cache:
            ts, cached_val = self._plc_numeric_cache[cache_key]
            if now - ts < PLC_NUMERIC_CACHE_TTL_SEC:
                val = cached_val
            else:
                val = self.plc_client.read_s7_value(action.ip, action.rack, action.slot, address, dtype)
                self._plc_numeric_cache[cache_key] = (now, val)
        else:
            val = self.plc_client.read_s7_value(action.ip, action.rack, action.slot, address, dtype)
            self._plc_numeric_cache[cache_key] = (now, val)

        if val is None:
            return False

        # 3. Comparar
        operator = str(condition.get("operator", "=")).strip()
        val1_str = str(condition.get("value1", "0"))
        
        try:
            current_val = float(val)
            ref_val1 = float(val1_str)
        except (ValueError, TypeError):
             return False

        result = False
        if operator == "=":
            result = abs(current_val - ref_val1) < 0.000001
        elif operator == ">":
            result = current_val > ref_val1
        elif operator == "<":
            result = current_val < ref_val1
        elif operator == ">=":
            result = current_val >= ref_val1
        elif operator == "<=":
            result = current_val <= ref_val1
        elif operator == "!=":
            result = abs(current_val - ref_val1) > 0.000001
        elif operator == "between":
            val2_str = str(condition.get("value2", "0"))
            try:
                ref_val2 = float(val2_str)
                low = min(ref_val1, ref_val2)
                high = max(ref_val1, ref_val2)
                result = low <= current_val <= high
            except ValueError:
                return False
        
        LOGGER.debug("PLC Numeric: %s %s (%s) %s -> %s", address, val, dtype, operator, result)
        return result


    def _resolve_plc_condition_action(self, condition: dict[str, object]) -> PLCAction | None:
        cache_key = self._build_plc_action_cache_key(condition)
        cached_action = self._plc_condition_action_cache.get(cache_key)
        if cached_action is self._ACTION_NOT_FOUND:
            LOGGER.debug("Acción PLC cacheada como no encontrada para %s", condition)
            return None
        if isinstance(cached_action, PLCAction):
            LOGGER.debug("Acción PLC recuperada de caché: %s -> %s", condition, cached_action)
            return cached_action

        preset_id = condition.get("preset_id")
        payload: dict[str, object]
        if preset_id:
            preset = self.profile_manager.get_plc_preset(str(preset_id))
            payload = preset.to_payload() if preset is not None else {}
            LOGGER.debug("Preset PLC cargado (%s): %s", preset_id, payload)
        else:
            preset = None
            payload = {}

        def _choose(first: object, second: object, *, default: object = None) -> object:
            return first if first is not None else (second if second is not None else default)

        expected_value = condition.get("expected_value")
        if expected_value is None and "expected" in condition:
            expected_value = condition.get("expected")
        if expected_value is None and "value" in condition:
            expected_value = condition.get("value")
        if expected_value is None:
            expected_value = True

        payload.update(
            {
                "preset_id": preset_id,
                "ip": payload.get("ip") or condition.get("ip"),
                "rack": _choose(payload.get("rack"), condition.get("rack"), default=0),
                "slot": _choose(payload.get("slot"), condition.get("slot"), default=2),
                "area": payload.get("area") or condition.get("area", "M"),
                "db_number": _choose(payload.get("db_number"), condition.get("db_number")),
                "byte_index": _choose(payload.get("byte_index"), condition.get("byte_index"), default=0),
                "bit_index": _choose(payload.get("bit_index"), condition.get("bit_index"), default=0),
                "value": _choose(payload.get("value"), expected_value, default=True),
                "tag": payload.get("tag") or condition.get("label", ""),
            }
        )

        LOGGER.debug("Payload PLC para resolver condición: %s", payload)

        try:
            action = self.profile_manager.resolve_plc_action(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudo resolver preset/IP para condición PLC: %s", exc)
            self._plc_condition_action_cache[cache_key] = self._ACTION_NOT_FOUND
            return None
        if not action.ip:
            LOGGER.warning(
                "Condición PLC sin IP definida (preset=%s area=%s byte=%s bit=%s)",
                payload.get("preset_id", ""),
                action.area,
                action.byte_index,
                action.bit_index,
            )
            self._plc_condition_action_cache[cache_key] = self._ACTION_NOT_FOUND
            return None

        self._plc_condition_action_cache[cache_key] = action
        return action

    def _build_plc_cache_key(self, action: PLCAction) -> tuple:
        return (
            action.ip,
            action.rack,
            action.slot,
            action.area,
            action.db_number,
            action.byte_index,
            action.bit_index,
        )

    def _build_plc_action_cache_key(self, condition: dict[str, object]) -> tuple:
        preset_id = str(condition.get("preset_id", "")).strip()
        ip = str(condition.get("ip", "")).strip()
        try:
            rack = int(condition.get("rack", 0) or 0)
        except (TypeError, ValueError):
            rack = 0
        try:
            slot = int(condition.get("slot", 2) or 0)
        except (TypeError, ValueError):
            slot = 2
        area = str(condition.get("area", "M")).strip().upper()
        try:
            db_number = int(condition.get("db_number"))
        except (TypeError, ValueError):
            db_number = None
        try:
            byte_index = int(condition.get("byte_index", 0) or 0)
        except (TypeError, ValueError):
            byte_index = 0
        try:
            bit_index = int(condition.get("bit_index", 0) or 0)
        except (TypeError, ValueError):
            bit_index = 0
        return (preset_id, ip, rack, slot, area, db_number, byte_index, bit_index)

    def _get_cached_plc_value(self, cache_key: tuple) -> bool | None | object:
        record = self._plc_condition_cache.get(cache_key)
        if record is None:
            return self._PLC_CACHE_MISS
        ts, value = record
        ttl = self._plc_condition_cache_ttl
        if ttl > 0:
            now = time.monotonic()
            if now - ts > ttl:
                self._plc_condition_cache.pop(cache_key, None)
                return self._PLC_CACHE_MISS
        return value

    def _store_plc_cache_value(self, cache_key: tuple, value: bool | None) -> None:
        self._plc_condition_cache[cache_key] = (time.monotonic(), value)

    def _expire_plc_condition_cache(self) -> None:
        if not self._plc_condition_cache:
            return
        ttl = self._plc_condition_cache_ttl
        if ttl <= 0:
            return
        now = time.monotonic()
        expired = [key for key, (ts, _) in self._plc_condition_cache.items() if now - ts > ttl]
        for key in expired:
            self._plc_condition_cache.pop(key, None)

    def _expire_plc_numeric_cache(self) -> None:
        if not self._plc_numeric_cache:
            return
        ttl = PLC_NUMERIC_CACHE_TTL_SEC
        if ttl <= 0:
            return
        now = time.monotonic()
        expired = [key for key, (ts, _) in self._plc_numeric_cache.items() if now - ts > ttl]
        for key in expired:
            self._plc_numeric_cache.pop(key, None)

    def _expire_muted_rules(self) -> None:
        if not self._muted_rules:
            return
        now = time.monotonic()
        expired = [rid for rid, (ts, _) in self._muted_rules.items() if ts > 0 and now >= ts]
        for rid in expired:
            self._muted_rules.pop(rid, None)

    def _active_muted_rule_ids(self) -> set[str]:
        self._expire_muted_rules()
        return set(self._muted_rules.keys())

    def _apply_muted_triggers_effects(self, effects: RuleEffects) -> None:
        muted = getattr(effects, "muted_triggers", []) or []
        if not muted:
            return
        now = time.monotonic()
        for item in muted:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id", "")).strip()
            if not rule_id:
                continue
            duration_raw = item.get("duration_sec")
            try:
                duration = max(0, int(duration_raw)) if duration_raw is not None else 0
            except Exception:
                duration = 0
            label = str(item.get("label", "")).strip()
            expires_at = now + duration if duration > 0 else 0.0
            self._muted_rules[rule_id] = (expires_at, label or rule_id)

    def _muted_rules_summary(self) -> list[dict[str, object]]:
        self._expire_muted_rules()
        now = time.monotonic()
        summary: list[dict[str, object]] = []
        for rule_id, (expires_at, label) in self._muted_rules.items():
            remaining = max(0.0, expires_at - now) if expires_at > 0 else 0.0
            summary.append({
                "rule_id": rule_id,
                "label": label or rule_id,
                "remaining_sec": remaining,
            })
        return summary

    def _apply_rule_effects_to_order(self, order: PLCOrder, effects: RuleEffects) -> PLCOrder:
        order.trigger_active = bool(order.trigger_active or effects.snapshot_requests)
        messages: list[str] = [order.message] if order.message else []
        if effects.forced_level:
            order.force_level = effects.forced_level
            order.block_decisions = True
            if effects.forced_level not in messages:
                messages.append(f"Regla -> forzar nivel {effects.forced_level}")
        if effects.blocked_classes:
            blocked = ", ".join(sorted(effects.blocked_classes))
            messages.append(f"Regla -> bloquear clases: {blocked}")
        muted = self._muted_rules_summary()
        if muted:
            muted_labels = ", ".join(item.get("label", item.get("rule_id", "")) or "" for item in muted)
            messages.append(f"Regla -> triggers deshabilitados: {muted_labels}")
        if getattr(effects, "resume_level", None):
            order.force_level = effects.resume_level
            messages.append(f"Regla -> reanudar nivel {effects.resume_level}")
        if getattr(effects, "plc_orders", None):
            for action_payload in effects.plc_orders:
                try:
                    plc_action = self.profile_manager.resolve_plc_action(action_payload)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("No se pudo resolver acción PLC: %s", exc)
                    order.plc_feedback.append(f"Resolver PLC falló: {exc}")
                    continue
                success, info = self._send_plc_action(plc_action)
                status = "OK" if success else "ERROR"
                feedback = f"PLC ({plc_action.tag or plc_action.area}): {status} - {info}"
                order.plc_feedback.append(feedback)
                if not success:
                    messages.append(f"Regla -> fallo envío PLC: {info}")
        order.message = "; ".join(filter(None, messages))
        return order

    def _send_plc_action(self, action: PLCAction) -> tuple[bool, str]:
        try:
            return self.plc_client.send_action(action)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Excepción enviando acción PLC (%s): %s", action.area, exc)
            return False, f"Excepción: {exc}"

    def get_rules_payload(self) -> list[dict[str, object]]:
        with self._lock:
            return [copy.deepcopy(rule) for rule in self.profile_manager.get_active_rules()]

    def replace_rules_from_payload(self, items: Iterable[dict[str, object]]) -> None:
        rules_payload: list[dict[str, object]] = []
        rules: list[Rule] = []
        for item in items:
            if isinstance(item, dict):
                rules_payload.append(copy.deepcopy(item))
                try:
                    rules.append(Rule.from_dict(item))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Regla ignorada por formato inválido: %s", exc)
        with self._lock:
            self.profile_manager.replace_profile_rules(self.profile_manager.active_profile_id, rules_payload)
            self.profile_manager.save_if_dirty()
            self._muted_rules = {}
        self.rule_engine.update_rules(rules)
        self._notify_state()

    def get_rule_evaluation(self) -> RuleEvaluation | None:
        with self._lock:
            return copy.deepcopy(self._rule_evaluation)

    # ------------------------------------------------------------------
    # Gestión de perfiles (expuesta para la UI)
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[dict[str, object]]:
        with self._lock:
            return self.profile_manager.list_profiles_payload()

    def set_active_profile(self, profile_id: str) -> dict[str, object]:
        with self._lock:
            profile = self.profile_manager.set_active_profile(profile_id)
            overrides = profile.overrides
            self.engine.set_overrides(copy.deepcopy(overrides))
            rules_payload = profile.rules
            rules: list[Rule] = []
            for item in rules_payload:
                if isinstance(item, dict):
                    try:
                        rules.append(Rule.from_dict(item))
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Regla ignorada por formato inválido en perfil '%s': %s", profile_id, exc)
            self.rule_engine.update_rules(rules)
            self._rule_evaluation = None
            self._rule_effects = RuleEffects()
            self._order = None
            self._muted_rules = {}
            self.profile_manager.save_if_dirty()
            payload = profile.to_payload(include_id=True)
        self._notify_state()
        return payload
    def create_profile(
        self,
        name: str,
        base_profile_id: str | None = None,
        *,
        copy_rules: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            profile = self.profile_manager.create_profile(name, base_profile_id)
            if not copy_rules and profile.rules:
                profile.rules = []
                self.profile_manager.replace_profile_rules(profile.profile_id, [])
            profile_payload = profile.to_payload(include_id=True)
            self.profile_manager.save_if_dirty()
        return profile_payload

    def delete_profile(self, profile_id: str) -> None:
        with self._lock:
            self.profile_manager.delete_profile(profile_id)
            overrides = self.profile_manager.get_active_overrides()
            self.engine.set_overrides(copy.deepcopy(overrides))
            rules_payload = self.profile_manager.get_active_rules()
            rules: list[Rule] = []
            for item in rules_payload:
                if isinstance(item, dict):
                    try:
                        rules.append(Rule.from_dict(item))
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Regla ignorada por formato invalido en perfil '%s': %s",
                            self.profile_manager.active_profile_id,
                            exc,
                        )
            self.rule_engine.update_rules(rules)
            self._rule_evaluation = None
            self._rule_effects = RuleEffects()
            self._order = None
            self._muted_rules = {}
            self.profile_manager.save_if_dirty()
        self._notify_state()

    def rename_profile(self, profile_id: str, new_name: str) -> dict[str, object]:
        with self._lock:
            profile = self.profile_manager.rename_profile(profile_id, new_name)
            self.profile_manager.save_if_dirty()
            return profile.to_payload(include_id=True)

    def replace_profile_actions(self, profile_id: str, actions: Iterable[dict[str, object]]) -> None:
        with self._lock:
            self.profile_manager.replace_profile_actions(profile_id, actions)
            self.profile_manager.save_if_dirty()

    def replace_profile_conditions(self, profile_id: str, conditions: Iterable[dict[str, object]]) -> None:
        with self._lock:
            self.profile_manager.replace_profile_conditions(profile_id, conditions)
            self.profile_manager.save_if_dirty()

    # ------------------------------------------------------------------
    # Gestión de presets PLC (expuesta para la UI)
    # ------------------------------------------------------------------

    def list_plc_presets(self) -> list[dict[str, object]]:
        with self._lock:
            return self.profile_manager.list_plc_presets()

    def upsert_plc_preset(self, payload: dict[str, Any]) -> dict[str, object]:
        with self._lock:
            preset = self.profile_manager.upsert_plc_preset(payload)
            self.profile_manager.save_if_dirty()
            return preset.to_payload()

    def delete_plc_preset(self, preset_id: str) -> None:
        with self._lock:
            self.profile_manager.delete_plc_preset(preset_id)
            self.profile_manager.save_if_dirty()


if tk is not None:  # UI disponible

    from tkinter import filedialog
 
    def _handle_mousewheel_event(event: tk.Event, canvas: tk.Canvas, window: tk.Misc) -> None:
        """Helper robusto para gestionar el scroll de forma aislada por ventana."""
        try:
            if not window.winfo_exists():
                return
            if event.widget and str(event.widget.winfo_toplevel()) != str(window.winfo_toplevel()):
                return
            widget = event.widget
            if widget:
                w_class = widget.winfo_class()
                if w_class in ("Listbox", "Text", "Treeview"):
                    return
                p_id = widget.winfo_parent()
                if p_id:
                    parent = widget.nametowidget(p_id)
                    if parent.winfo_class() in ("Listbox", "Text", "Treeview") or "scrolled" in str(parent).lower():
                        return
        except Exception:
            return
        delta = int(-1 * (event.delta / 120))
        canvas.yview_scroll(delta, "units")

    class SendToPLCWindow:
        """Ventana Tkinter que permite inspeccionar y ajustar el servicio."""

        _profile_section_bg = "#e8f0ff"
        _profile_section_border = "#4b7ae5"
        _decision_section_bg = "#e6f7f0"
        _decision_section_border = "#2f9c73"
        _rules_section_bg = "#fff1e3"
        _rules_section_border = "#e08e3c"
        _active_rule_bg = "#7be181"
        _rule_state_labels: dict[str, str] = {
            "pending": "Pendiente",
            "active": "Disparando",
            "idle": "En espera",
            "muted": "Pausada",
            "disabled": "Desactivada",
        }
        _rule_state_colors: dict[str, str] = {
            "pending": "#b8860b",
            "active": "#1f7a35",
            "idle": "#1a1a1a",
            "muted": "#6c6c6c",
            "disabled": "#cc0000",
        }

        def __init__(
            self,
            master: tk.Misc,
            service: SendToPLCService,
            *,
            on_close: Callable[[], None] | None = None,
        ) -> None:
            self.service = service
            self.on_close_callback = on_close
            self.window = tk.Toplevel(master)
            self.window.title("sendToPLC - Supervisión")
            self.window.transient(master)  # Evitar icono extra en barra de tareas
            self.window.protocol("WM_DELETE_WINDOW", self._on_close)
            self.window.state("zoomed")
            self.window.minsize(1100, 760)
            self.window.resizable(True, True)
            self.window.columnconfigure(0, weight=1)
            self.window.rowconfigure(0, weight=1)

            self.decision_summary = tk.StringVar(value="Sin decisiones aún")
            self.resume_eta_text = tk.StringVar(value="")
            self.plc_feedback_text = tk.StringVar(value="")
            self.muted_status_text = tk.StringVar(value="")

            self._available_classes: tuple[str, ...] = tuple()
            self._model_info = {}
            self._updating_ui = False
            self._unsubscribe: Callable[[], None] | None = None

            self._profiles_cache: dict[str, dict[str, object]] = {}
            self._profile_ids: list[str] = []
            self._profile_selecting = False
            self._current_profile_id: Optional[str] = None
            self._rule_states: dict[str, str] = {}
            self._active_rule_ids: set[str] = set()
            self._muted_rule_ids: set[str] = set()
            self._rules_monitoring_ready = False
            self._selected_rule_id: Optional[str] = None
            self._rules_sort_col: str | None = None
            self._rules_sort_desc = False
            self._rules_by_id_cache: dict[str, dict[str, object]] = {} # Cache para búsqueda rápida O(1)
            self._rules_total_count = 0
            self._rules_paused_count = 0
            if InfoIcon is not None and init_tooltips is not None:
                try:
                    tooltips_path = get_resource_path(
                        os.path.join("..", "config", "tooltips.json"),
                        is_config=True,
                    )
                    init_tooltips(tooltips_path)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("No se pudo inicializar tooltips: %s", exc)
            self._build_ui()
            self._load_profiles()
            self._unsubscribe = self.service.subscribe(self._on_service_update)
            self._apply_model_metadata(self.service.get_model_metadata())
            self._load_rules_from_service()

        # ------------------------------------------------------------------
        def _build_ui(self) -> None:
            outer = ttk.Frame(self.window)
            outer.grid(row=0, column=0, sticky="nsew")
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(0, weight=1)

            canvas = tk.Canvas(outer, highlightthickness=0)
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.configure(background=self.window.cget("background"))

            main = ttk.Frame(canvas, padding=10)
            self._main_canvas = canvas
            self._main_canvas_window = canvas.create_window((0, 0), window=main, anchor="nw")
            main.bind("<Configure>", self._on_main_frame_configure)
            canvas.bind("<Configure>", self._on_main_canvas_configure)
 
            # El scroll ahora se gestiona de forma global y robusta para evitar conflictos
            self.window.bind_all("<MouseWheel>", lambda e: _handle_mousewheel_event(e, canvas, self.window))
            self.window.bind("<Destroy>", lambda _: self.window.unbind_all("<MouseWheel>"))

            main.columnconfigure(0, weight=1)
            main.rowconfigure(0, weight=1)

            container = ttk.Frame(main)
            container.grid(row=0, column=0, sticky="nsew")
            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=0)
            container.rowconfigure(1, weight=0)
            container.rowconfigure(2, weight=1)

            profiles_wrapper = tk.Frame(
                container,
                background=self._profile_section_bg,
                highlightbackground=self._profile_section_border,
                highlightcolor=self._profile_section_border,
                highlightthickness=2,
                bd=0,
            )
            profiles_wrapper.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
            profiles_wrapper.columnconfigure(0, weight=1)
            tk.Label(
                profiles_wrapper,
                text="Perfiles",
                background=self._profile_section_bg,
                foreground=self._profile_section_border,
                font=("TkDefaultFont", 11, "bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="we", padx=10, pady=(8, 0))
            profiles_body = ttk.Frame(profiles_wrapper, padding=(10, 8, 10, 10))
            profiles_body.grid(row=1, column=0, sticky="nsew")
            profiles_body.columnconfigure(0, weight=1)

            self._build_profiles_ui(profiles_body)

            status_wrapper = tk.Frame(
                container,
                background=self._decision_section_bg,
                highlightbackground=self._decision_section_border,
                highlightcolor=self._decision_section_border,
                highlightthickness=2,
                bd=0,
            )
            status_wrapper.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
            status_wrapper.columnconfigure(0, weight=1)
            tk.Label(
                status_wrapper,
                text="Última decisión",
                background=self._decision_section_bg,
                foreground=self._decision_section_border,
                font=("TkDefaultFont", 11, "bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="we", padx=10, pady=(8, 0))
            status_body = ttk.Frame(status_wrapper, padding=(10, 8, 10, 10))
            status_body.grid(row=1, column=0, sticky="nsew")
            status_body.columnconfigure(1, weight=1)
            accent = tk.Frame(status_body, background=self._decision_section_border, width=4, height=1)
            accent.grid(row=0, column=0, rowspan=4, sticky="nsw", padx=(0, 10), pady=2)
            ttk.Label(status_body, textvariable=self.decision_summary, justify="left", wraplength=960).grid(
                row=0, column=1, sticky="we", pady=(2, 4)
            )
            ttk.Label(status_body, textvariable=self.resume_eta_text, foreground="#2f3f3f").grid(
                row=1, column=1, sticky="w", pady=(0, 4)
            )
            ttk.Label(
                status_body,
                textvariable=self.muted_status_text,
                foreground="#c06800",
                justify="left",
                wraplength=960,
            ).grid(row=2, column=1, sticky="we", pady=(0, 4))
            ttk.Label(
                status_body,
                textvariable=self.plc_feedback_text,
                foreground=self._decision_section_border,
                justify="left",
                wraplength=960,
            ).grid(row=3, column=1, sticky="we")

            rules_wrapper = tk.Frame(
                container,
                background=self._rules_section_bg,
                highlightbackground=self._rules_section_border,
                highlightcolor=self._rules_section_border,
                highlightthickness=2,
                bd=0,
            )
            rules_wrapper.grid(row=2, column=0, sticky="nsew")
            rules_wrapper.columnconfigure(0, weight=1)
            rules_wrapper.rowconfigure(1, weight=1)
            tk.Label(
                rules_wrapper,
                text="Reglas acción-reacción",
                background=self._rules_section_bg,
                foreground=self._rules_section_border,
                font=("TkDefaultFont", 11, "bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="we", padx=10, pady=(8, 0))
            rules_body = ttk.Frame(rules_wrapper, padding=(10, 8, 10, 10))
            rules_body.grid(row=1, column=0, sticky="nsew")
            rules_body.columnconfigure(0, weight=1)
            rules_body.rowconfigure(0, weight=1)

            self._build_rules_ui(rules_body)

        # ------------------------------------------------------------------
        def _on_main_frame_configure(self, event: object) -> None:
            if hasattr(self, "_main_canvas"):
                self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all"))

        def _on_main_canvas_configure(self, event: object) -> None:
            if hasattr(self, "_main_canvas") and hasattr(self, "_main_canvas_window"):
                self._main_canvas.itemconfigure(self._main_canvas_window, width=self._main_canvas.winfo_width())

        def _on_main_canvas_enter(self, event: object) -> None:
            if hasattr(self, "_main_canvas"):
                self._main_canvas.bind_all("<MouseWheel>", self._on_main_mousewheel)

        def _on_main_canvas_leave(self, event: object) -> None:
            if hasattr(self, "_main_canvas"):
                self._main_canvas.unbind_all("<MouseWheel>")

        def _on_main_mousewheel(self, event: tk.Event) -> None:
            if not hasattr(self, "_main_canvas"):
                return

            # 1. Verificar que el evento pertenece a esta ventana (o sus hijos)
            try:
                if not self.window.winfo_exists():
                    return
                # Comprobar si el widget que disparó el evento está en nuestro toplevel
                if event.widget and str(event.widget.winfo_toplevel()) != str(self.window):
                    return
            except Exception:
                return

            # 2. Si el scroll ocurre sobre un widget con scroll propio, ignoramos
            widget = event.widget
            if widget:
                try:
                    w_class = widget.winfo_class()
                    if w_class in ("Listbox", "Text", "Treeview"):
                        return
                    # Comprobar si el padre es un contenedor de scroll (como ScrolledText)
                    p_id = widget.winfo_parent()
                    if p_id:
                        parent = widget.nametowidget(p_id)
                        if parent.winfo_class() in ("Listbox", "Text", "Treeview") or "scrolled" in str(parent).lower():
                            return
                except Exception:
                    pass

            delta = int(-1 * (event.delta / 120))
            self._main_canvas.yview_scroll(delta, "units")

        # ------------------------------------------------------------------
        def _on_service_update(self, snapshot: SnapshotState | None, order: PLCOrder) -> None:
            if not self.window.winfo_exists():
                return
            metadata = self.service.get_model_metadata()
            self.window.after(0, lambda: self._update_ui(snapshot, order, metadata))

        def _update_ui(self, snapshot: SnapshotState | None, order: PLCOrder, metadata: dict[str, object]) -> None:
            if snapshot is not None:
                self._update_class_list(tuple(sorted(snapshot.classes_short.keys())))
            self.decision_summary.set(order.message)
            if order.resume_eta_sec is not None:
                self.resume_eta_text.set(f"Reanudación automática en ~{order.resume_eta_sec:.1f}s")
            else:
                self.resume_eta_text.set("")
            muted_rules = metadata.get("muted_rules") if isinstance(metadata, dict) else None
            if isinstance(muted_rules, list) and muted_rules:
                parts: list[str] = []
                for item in muted_rules:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label", "") or item.get("rule_id", ""))
                    remaining = item.get("remaining_sec")
                    if remaining is not None:
                        try:
                            parts.append(f"{label} ({float(remaining):.1f}s)")
                        except Exception:
                            parts.append(str(label))
                    else:
                        parts.append(str(label))
                self.muted_status_text.set(f"Triggers deshabilitados: {', '.join(parts)}")
            else:
                self.muted_status_text.set("")
            if hasattr(order, "plc_feedback") and order.plc_feedback:
                feedback_lines = [f"  • {line}" for line in order.plc_feedback]
                self.plc_feedback_text.set("\n".join(feedback_lines))
            else:
                self.plc_feedback_text.set("")
            self._apply_model_metadata(metadata)
            self._update_rules_status()
            self._highlight_active_rule(metadata)

        def _shade_color(self, color: str, factor: float = 0.9) -> str:
            hex_color = color.lstrip("#")
            if len(hex_color) != 6:
                return color
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            except ValueError:
                return color
            r = max(0, min(255, int(r * factor)))
            g = max(0, min(255, int(g * factor)))
            b = max(0, min(255, int(b * factor)))
            return f"#{r:02x}{g:02x}{b:02x}"

        def _create_colored_button(
            self,
            parent: tk.Misc,
            text: str,
            command: Callable[[], None],
            *,
            base_color: str,
            fg: str = "#1f1f1f",
        ) -> tk.Button:
            darker = self._shade_color(base_color, 0.85)
            btn = tk.Button(
                parent,
                text=text,
                command=command,
                bg=base_color,
                fg=fg,
                activebackground=darker,
                activeforeground=fg,
                relief="flat",
                bd=1,
                highlightbackground=base_color,
                padx=12,
                pady=4,
                cursor="hand2",
            )
            return btn

        def _build_profiles_ui(self, main: ttk.Frame) -> None:
            main.columnconfigure(0, weight=0)
            main.columnconfigure(1, weight=1)
            main.columnconfigure(2, weight=0)
            main.columnconfigure(3, weight=0)

            ttk.Label(main, text="Perfil activo:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
            if InfoIcon is not None:
                InfoIcon(main, "sendToPLC.profiles.active").grid(row=0, column=0, sticky="e", padx=(0, 2), pady=4)
            self.combo_profiles = ttk.Combobox(main, state="readonly", values=(), width=28)
            self.combo_profiles.grid(row=0, column=1, sticky="we", pady=4)
            self.combo_profiles.bind("<<ComboboxSelected>>", lambda *_: self._on_profile_selected())

            self.btn_manage_plc_presets = self._create_colored_button(
                main,
                "Presets PLC",
                self._on_manage_plc_presets,
                base_color="#dfe7ff",
                fg="#25306a",
            )
            self.btn_manage_plc_presets.grid(row=0, column=2, sticky="e", padx=4, pady=4)

            write_btn = self._create_colored_button(
                main,
                "Write/Read",
                self._on_open_plc_bit_writer,
                base_color="#dff6ff",
                fg="#0b4c63",
            )
            write_btn.grid(row=0, column=3, sticky="e", padx=4, pady=4)

            button_bar = ttk.Frame(main)
            button_bar.grid(row=1, column=0, columnspan=4, sticky="we", padx=0, pady=(6, 0))
            for col in range(4):
                button_bar.columnconfigure(col, weight=1)

            self._create_colored_button(
                button_bar,
                "Nuevo",
                self._on_profile_new,
                base_color="#ffe3c2",
            ).grid(row=0, column=0, sticky="we", padx=4, pady=4)
            self._create_colored_button(
                button_bar,
                "Duplicar",
                self._on_profile_duplicate,
                base_color="#ffefc2",
            ).grid(row=0, column=1, sticky="we", padx=4, pady=4)
            self._create_colored_button(
                button_bar,
                "Renombrar",
                self._on_profile_rename,
                base_color="#e7f0ff",
            ).grid(row=0, column=2, sticky="we", padx=4, pady=4)
            self._create_colored_button(
                button_bar,
                "🗑 Eliminar",
                self._on_profile_delete,
                base_color="#ffe0dd",
                fg="#7a1d17",
            ).grid(row=0, column=3, sticky="we", padx=4, pady=4)

        def _update_class_list(self, classes: tuple[str, ...]) -> None:
            if classes == self._available_classes:
                return
            self._available_classes = classes

        def _apply_model_metadata(self, metadata: dict[str, object]) -> None:
            if not metadata or metadata == self._model_info:
                return
            self._model_info = metadata
            classes_raw = metadata.get("classes") if isinstance(metadata, dict) else None
            if isinstance(classes_raw, (list, tuple, set)):
                classes_iter: Sequence[object] = classes_raw  # type: ignore[assignment]
            else:
                classes_iter = ()
            classes_tuple = tuple(sorted({str(c) for c in classes_iter if c}))
            self._update_class_list(classes_tuple)
            muted = metadata.get("muted_rules") if isinstance(metadata, dict) else None
            if isinstance(muted, list):
                self._muted_rule_ids = {str(item.get("rule_id", "")) for item in muted if isinstance(item, dict)}
            self._refresh_rules_actions()

        def _ensure_latest_classes(self) -> tuple[str, ...]:
            metadata = self.service.get_model_metadata()
            classes_raw = metadata.get("classes") if isinstance(metadata, dict) else None
            if isinstance(classes_raw, (list, tuple, set)):
                classes_tuple = tuple(sorted({str(c) for c in classes_raw if c}))
            else:
                classes_tuple = ()
            self._update_class_list(classes_tuple)
            return classes_tuple

        def _highlight_active_rule(self, metadata: dict[str, object]) -> None:
            active_ids: set[str] = set()
            matches = metadata.get("active_rules") if isinstance(metadata, dict) else None
            monitoring_ready = isinstance(matches, (list, tuple))
            if monitoring_ready:
                active_ids = {str(rid) for rid in matches if rid}
            muted_meta = metadata.get("muted_rules") if isinstance(metadata, dict) else None
            muted_ids: set[str] = set()
            if isinstance(muted_meta, list):
                muted_ids = {
                    str(item.get("rule_id", ""))
                    for item in muted_meta
                    if isinstance(item, dict) and item.get("rule_id")
                }

            # Optimización: solo actualizar si hay cambios reales en los estados
            changed = (
                monitoring_ready != getattr(self, "_rules_monitoring_ready", None)
                or active_ids != getattr(self, "_active_rule_ids", None)
                or muted_ids != getattr(self, "_muted_rule_ids", None)
            )

            self._rules_monitoring_ready = monitoring_ready
            self._active_rule_ids = active_ids
            self._muted_rule_ids = muted_ids

            if changed:
                self._apply_rule_states()

        def _on_close(self) -> None:
            if messagebox and not messagebox.askokcancel("Cerrar", "¿Cerrar la ventana de sendToPLC?"):
                return
            if self._unsubscribe:
                self._unsubscribe()
                self._unsubscribe = None
            if callable(self.on_close_callback):
                self.on_close_callback()
            if self.window.winfo_exists():
                self.window.destroy()

        # ------------------------------------------------------------------
        # Gestión de perfiles, acciones y condiciones
        # ------------------------------------------------------------------

        def _load_profiles(self) -> None:
            try:
                payload = self.service.list_profiles()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudieron obtener perfiles: %s", exc)
                payload = []

            self._profiles_cache.clear()
            self._profile_ids.clear()
            self._current_profile_id = None

            entries: list[tuple[str, str]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                profile_id = str(item.get("profile_id", ""))
                if not profile_id:
                    continue
                self._profiles_cache[profile_id] = item
                self._profile_ids.append(profile_id)
                label = str(item.get("name", profile_id))
                entries.append((profile_id, label))
                if bool(item.get("is_active")):
                    self._current_profile_id = profile_id

            if entries and self._current_profile_id is None:
                self._current_profile_id = entries[0][0]

            labels = tuple(label for _, label in entries)
            self._profile_selecting = True
            try:
                self.combo_profiles.configure(values=labels)
                if self._current_profile_id and self._current_profile_id in self._profile_ids:
                    idx = self._profile_ids.index(self._current_profile_id)
                    self.combo_profiles.current(idx)
                elif labels:
                    self.combo_profiles.current(0)
                    self._current_profile_id = entries[0][0]
                else:
                    self.combo_profiles.set("")
            finally:
                self._profile_selecting = False

            self._load_profile_details()
            # Actualizar estado del botón de presets según si hay perfiles
            try:
                if hasattr(self, "btn_manage_plc_presets"):
                    self.btn_manage_plc_presets.configure(state=("normal" if self._profile_ids else "disabled"))
            except Exception:
                pass

        def _load_profile_details(self) -> None:
            profile = self._profiles_cache.get(self._current_profile_id or "")
            if not isinstance(profile, dict):
                self._rules_payload = []
                self._refresh_rules_list()
                self._update_rules_status()
                return

            overrides_dict = profile.get("overrides") if isinstance(profile.get("overrides"), dict) else {}
            self._apply_overrides(_overrides_from_dict(overrides_dict))
            rules_list = profile.get("rules") if isinstance(profile.get("rules"), list) else []
            self._rules_payload = [copy.deepcopy(rule) for rule in rules_list if isinstance(rule, dict)]
            self._refresh_rules_list()
            self._update_rules_status()

        def _apply_overrides(self, overrides: OperatorOverrides) -> None:
            # La interfaz reducida ya no expone controles de overrides; se conserva por compatibilidad.
            self._updating_ui = True
            try:
                self._current_overrides = overrides
            finally:
                self._updating_ui = False

        def _on_profile_selected(self) -> None:
            if self._profile_selecting:
                return
            idx = self.combo_profiles.current()
            if idx < 0 or idx >= len(self._profile_ids):
                return
            new_profile_id = self._profile_ids[idx]
            if new_profile_id == self._current_profile_id:
                return
            try:
                payload = self.service.set_active_profile(new_profile_id)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Perfiles", f"No se pudo activar el perfil.\nDetalle: {exc}")
                return
            self._current_profile_id = str(payload.get("profile_id", new_profile_id))
            self._profiles_cache[self._current_profile_id] = payload
            self._load_profile_details()
            self._ensure_latest_classes()
            self._load_rules_from_service()

        def _ask_text(self, title: str, prompt: str, initial: str = "") -> str | None:
            if simpledialog is None:
                return None
            return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self.window)

        def _on_profile_new(self) -> None:
            name = self._ask_text("Nuevo perfil", "Nombre del perfil:", "Perfil")
            if not name:
                return
            base_id = self._current_profile_id
            try:
                profile = self.service.create_profile(name.strip(), base_id, copy_rules=False)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Perfiles", f"No se pudo crear el perfil.\nDetalle: {exc}")
                return
            profile_id = str(profile.get("profile_id"))
            if profile_id:
                self._profiles_cache[profile_id] = profile
                try:
                    payload = self.service.set_active_profile(profile_id)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("No se pudo activar el nuevo perfil '%s': %s", profile_id, exc)
                else:
                    self._profiles_cache[profile_id] = payload
            self._load_profiles()
            try:
                self.btn_manage_plc_presets.configure(state="normal")
            except Exception:
                pass

        def _on_profile_duplicate(self) -> None:
            if not self._current_profile_id:
                messagebox.showinfo("Perfiles", "No hay perfil activo para duplicar.")
                return
            info = self._profiles_cache.get(self._current_profile_id, {})
            base_name = str(info.get("name", "Perfil"))
            name = self._ask_text("Duplicar perfil", "Nombre para la copia:", f"{base_name} (copia)")
            if not name:
                return
            try:
                profile = self.service.create_profile(name.strip(), self._current_profile_id, copy_rules=True)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Perfiles", f"No se pudo duplicar el perfil.\nDetalle: {exc}")
                return
            self._profiles_cache[str(profile.get("profile_id"))] = profile
            self._load_profiles()
            try:
                self.btn_manage_plc_presets.configure(state="normal")
            except Exception:
                pass

        def _on_profile_rename(self) -> None:
            if not self._current_profile_id:
                messagebox.showinfo("Perfiles", "Selecciona un perfil para renombrar.")
                return
            info = self._profiles_cache.get(self._current_profile_id, {})
            name = self._ask_text("Renombrar perfil", "Nuevo nombre:", str(info.get("name", "Perfil")))
            if not name:
                return
            try:
                profile = self.service.rename_profile(self._current_profile_id, name.strip())
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Perfiles", f"No se pudo renombrar el perfil.\nDetalle: {exc}")
                return
            self._profiles_cache[self._current_profile_id] = profile
            self._load_profiles()
            try:
                self.btn_manage_plc_presets.configure(state="normal" if self._profile_ids else "disabled")
            except Exception:
                pass

        def _on_profile_delete(self) -> None:
            if not self._current_profile_id:
                return
            info = self._profiles_cache.get(self._current_profile_id, {})
            name = str(info.get("name", self._current_profile_id))
            if not messagebox or not messagebox.askyesno("🗑 Eliminar perfil", f"¿🗑 Eliminar el perfil '{name}'?"):
                return
            try:
                self.service.delete_profile(self._current_profile_id)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Perfiles", f"No se pudo eliminar el perfil.\nDetalle: {exc}")
                return
            self._current_profile_id = None
            self._load_profiles()
            try:
                self.btn_manage_plc_presets.configure(state="disabled" if not self._profile_ids else "normal")
            except Exception:
                pass

        def _on_manage_plc_presets(self) -> None:
            try:
                dialog = _PlcPresetsDialog(self.window, self.service)
                dialog.show()
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Presets PLC", f"No se pudo abrir la gestión de presets.\nDetalle: {exc}")

        def _on_open_plc_bit_writer(self) -> None:
            try:
                script_path = Path(__file__).with_name("plc_bit_writer.py")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Write/Read PLC", f"No se pudo resolver la ruta del script.\nDetalle: {exc}")
                return

            if not script_path.is_file():
                messagebox.showerror(
                    "Write/Read PLC",
                    f"No se encontró el archivo '{script_path.name}'.",
                )
                return

            try:
                subprocess.Popen([sys.executable, str(script_path)], cwd=str(script_path.parent))
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Write/Read PLC", f"No se pudo abrir el script.\nDetalle: {exc}")

        def destroy(self) -> None:
            if self._unsubscribe:
                self._unsubscribe()
                self._unsubscribe = None
            if self.window.winfo_exists():
                self.window.destroy()

        # ------------------------------------------------------------------
        # Panel de reglas acción-reacción
        # ------------------------------------------------------------------

        def _build_rules_ui(self, main: ttk.Frame) -> None:
            style = ttk.Style(self.window)
            style.configure("RulesAccent.TButton", padding=(8, 4), background=self._rules_section_border, foreground="#ffffff")
            style.map("RulesAccent.TButton", background=[("active", "#c9772d")], foreground=[("disabled", "#f6e1c7")])
            style.configure("RulesGhost.TButton", padding=(8, 4))
            style.configure("RulesSearch.TEntry", foreground="#1a1a1a")
            style.configure("RulesSearchPlaceholder.TEntry", foreground="#7a7a7a")
            style.configure("Rules.Treeview", background="#ffffff", fieldbackground="#ffffff", rowheight=30, relief="flat", font=("TkDefaultFont", 10))
            style.map(
                "Rules.Treeview",
                background=[("selected", "#3478f6"), ("active", "#e8f0ff")],
                foreground=[("selected", "#ffffff")],
            )
            style.configure(
                "Rules.Treeview.Heading",
                background="#f0f0f0",
                foreground="#1a1a1a",
                font=("TkDefaultFont", 9, "bold"),
                relief="raised",
            )
            style.map(
                "Rules.Treeview.Heading",
                background=[("active", "#e5e5e5")],
            )

            main.columnconfigure(0, weight=1)
            main.columnconfigure(1, weight=0)
            main.rowconfigure(1, weight=1)

            filter_bar = ttk.Frame(main)
            filter_bar.grid(row=0, column=0, columnspan=2, sticky="we", padx=6, pady=(6, 0))
            filter_bar.columnconfigure(1, weight=1)

            ttk.Label(filter_bar, text="Buscar:").grid(row=0, column=0, sticky="w", padx=(0, 6))
            if InfoIcon is not None:
                InfoIcon(filter_bar, "sendToPLC.rules.search").grid(row=0, column=0, sticky="e", padx=(0, 2))
            self._rules_search_placeholder = "Buscar (nombre, clase, accion...)"
            self._rules_search_var = tk.StringVar()
            self.entry_rules_search = ttk.Entry(
                filter_bar,
                textvariable=self._rules_search_var,
                width=40,
                style="RulesSearchPlaceholder.TEntry",
            )
            self.entry_rules_search.grid(row=0, column=1, sticky="we")
            self.entry_rules_search.bind("<FocusIn>", self._on_rules_search_focus_in)
            self.entry_rules_search.bind("<FocusOut>", self._on_rules_search_focus_out)

            ttk.Label(filter_bar, text="Filtro:").grid(row=0, column=2, sticky="w", padx=(8, 6))
            if InfoIcon is not None:
                InfoIcon(filter_bar, "sendToPLC.rules.filter").grid(row=0, column=2, sticky="e", padx=(0, 2))
            self._rules_filter_var = tk.StringVar(value="Todas")
            self.combo_rules_filter = ttk.Combobox(
                filter_bar,
                textvariable=self._rules_filter_var,
                values=("Todas", "Disparando ahora", "Pausadas", "Silenciadas", "Pendientes"),
                state="readonly",
                width=18,
            )
            self.combo_rules_filter.grid(row=0, column=3, sticky="w")
            # self.combo_rules_filter.bind("<<ComboboxSelected>>", lambda *_: self._refresh_rules_list()) # Redundante con trace
            self._rules_filter_var.trace_add("write", lambda *_: self._refresh_rules_list())

            ttk.Button(filter_bar, text="Limpiar", command=self._on_rules_clear, style="RulesGhost.TButton").grid(
                row=0, column=4, sticky="e", padx=(8, 0)
            )

            self._rules_search_placeholder_active = False
            self._set_rules_search_placeholder()
            self._rules_search_var.trace_add("write", lambda *_: self._on_rules_search_change())

            self.tree_rules = ttk.Treeview(
                main,
                columns=("now", "on", "name", "class", "condition", "priority", "actions"),
                show="headings",
                height=10,
                selectmode="browse",
                style="Rules.Treeview",
            )
            # Style priority: the last configured tag has the highest precedence.
            self.tree_rules.tag_configure("row_even", background="#ffffff")
            self.tree_rules.tag_configure("row_odd", background="#f9f9f9")
            self.tree_rules.tag_configure("rule_pending", foreground="#b8860b")
            self.tree_rules.tag_configure("rule_muted", foreground="#6c6c6c")
            self.tree_rules.tag_configure("rule_disabled", foreground="#d32f2f")
            # The active rule should have the highest priority for both bg and fg
            self.tree_rules.tag_configure("active_rule", background="#e8f5e9", foreground="#2e7d32")
            self.tree_rules.heading("now", text="Ahora", command=lambda c="now": self._sort_treeview(c))
            self.tree_rules.heading("on", text="ON", command=lambda c="on": self._sort_treeview(c))
            self.tree_rules.heading("name", text="Nombre", command=lambda c="name": self._sort_treeview(c))
            self.tree_rules.heading("class", text="Clase", command=lambda c="class": self._sort_treeview(c))
            self.tree_rules.heading("condition", text="Condicion", command=lambda c="condition": self._sort_treeview(c))
            self.tree_rules.heading("priority", text="Prioridad", command=lambda c="priority": self._sort_treeview(c))
            self.tree_rules.heading("actions", text="Acciones", command=lambda c="actions": self._sort_treeview(c))
            self.tree_rules.column("now", width=60, anchor="center", stretch=False)
            self.tree_rules.column("on", width=55, anchor="center", stretch=False)
            self.tree_rules.column("name", width=170, anchor="w")
            self.tree_rules.column("class", width=120, anchor="w")
            self.tree_rules.column("condition", width=150, anchor="w")
            self.tree_rules.column("priority", width=70, anchor="center", stretch=False)
            self.tree_rules.column("actions", width=220, anchor="w")
            self.tree_rules.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=6)

            scrollbar = ttk.Scrollbar(main, orient="vertical", command=self.tree_rules.yview)
            scrollbar.grid(row=1, column=1, sticky="ns", pady=6, padx=(0, 6))
            self.tree_rules.configure(yscrollcommand=scrollbar.set)

            self.tree_rules.bind("<<TreeviewSelect>>", self._on_rule_select)
            self.tree_rules.bind("<Button-1>", self._on_rules_click)
            self.tree_rules.bind("<Double-1>", self._on_rule_double_click)

            button_bar = ttk.Frame(main)
            button_bar.grid(row=2, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 6))
            button_bar.columnconfigure((0, 1, 2, 3, 4), weight=1)

            # Usar tk.Button para asegurar que los colores de fondo se vean en Windows
            tk.Button(button_bar, text="➕ Nueva", command=self._on_rule_add, bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white", relief="flat", padx=10, pady=5, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, padx=3, sticky="we")
            tk.Button(button_bar, text="📝 Editar", command=self._on_rule_edit, bg="#1976d2", fg="white", activebackground="#1565c0", activeforeground="white", relief="flat", padx=10, pady=5).grid(row=0, column=1, padx=3, sticky="we")
            tk.Button(button_bar, text="👯 Duplicar", command=self._on_rule_duplicate, bg="#607d8b", fg="white", activebackground="#455a64", activeforeground="white", relief="flat", padx=10, pady=5).grid(row=0, column=2, padx=3, sticky="we")
            tk.Button(button_bar, text="🗑 Eliminar", command=self._on_rule_delete, bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white", relief="flat", padx=10, pady=5).grid(row=0, column=3, padx=3, sticky="we")
            tk.Button(button_bar, text="📕 Exportar PDF", command=self._on_rule_export_pdf, bg="#f5f5f5", relief="flat", padx=10, pady=5).grid(row=0, column=4, padx=3, sticky="we")

            self.rule_status_var = tk.StringVar(value="Reglas: 0 | Disparando: 0 | Pausadas: 0")
            ttk.Label(main, textvariable=self.rule_status_var, foreground=self._rules_section_border).grid(
                row=3, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 4)
            )

            preview_frame = ttk.LabelFrame(main, text="Vista previa")
            preview_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 6))
            preview_frame.columnconfigure(0, weight=1)

            preview_widget: tk.Text
            if ScrolledText is not None:
                preview_widget = ScrolledText(preview_frame, height=6, wrap="word")
            else:
                preview_widget = tk.Text(preview_frame, height=6, wrap="word")
            preview_widget.configure(state="disabled", background="#ffffff", relief="solid", borderwidth=1)
            preview_widget.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
            self.rule_preview_text = preview_widget
            self._rule_preview_empty = "Selecciona una regla para ver detalles."
            self._set_rule_preview_text(self._rule_preview_empty)

            self._rules_payload: list[dict[str, object]] = []
            self._rules_by_id_cache = {}

        def _get_rules_payload_with_cache(self) -> list[dict[str, object]]:
            return self._rules_payload

        def _update_rules_payload_cache(self, payload: list[dict[str, object]]) -> None:
            self._rules_payload = payload
            self._rules_by_id_cache = {str(r.get("rule_id")): r for r in payload if isinstance(r, dict)}
            self._rules_total_count = len(payload)
            self._rules_paused_count = sum(1 for r in payload if not bool(r.get("enabled", True)))

        def _load_rules_from_service(self) -> None:
            try:
                payload = self.service.get_rules_payload()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudieron obtener reglas del servicio: %s", exc)
                payload = []
            self._update_rules_payload_cache(payload or [])
            self._refresh_rules_list()

        def _refresh_rules_list(self) -> None:
            self._rule_states = {}
            existing_selection = self._selected_rule_id
            self._rules_display_cache = {}
            rules_view: list[dict[str, object]] = []
            for item in self.tree_rules.get_children():
                self.tree_rules.delete(item)
            for rule in self._rules_payload:
                rule_id = str(rule.get("rule_id", uuid.uuid4().hex))
                condition = rule.get("condition", {}) if isinstance(rule.get("condition"), dict) else {}
                class_name = str(condition.get("class_name", ""))
                priority = int(rule.get("priority", 0))
                actions_summary = self._summarize_actions(rule.get("actions"))
                condition_summary = self._summarize_condition(condition)
                state_key = self._resolve_rule_state(rule_id)
                display_name = self._format_rule_name(rule, state_key)
                if state_key == "active":
                    now_value = "● SI"
                else:
                    now_value = "  NO" # Espacio para alinear con el círculo
                on_value = "ON" if bool(rule.get("enabled", True)) else "OFF"
                self._rule_states[rule_id] = state_key
                entry = {
                    "rule_id": rule_id,
                    "rule": rule,
                    "state_key": state_key,
                    "display_name": display_name,
                    "class_name": class_name,
                    "condition_summary": condition_summary,
                    "priority": priority,
                    "actions_summary": actions_summary,
                    "now_value": now_value,
                    "on_value": on_value,
                }
                self._rules_display_cache[rule_id] = entry
                rules_view.append(entry)

            query = self._get_rules_search_query()
            filter_label = self._get_rules_filter_key()
            visible_view: list[dict[str, object]] = []
            for entry in rules_view:
                if filter_label == "Pausadas":
                    rule = entry.get("rule")
                    if isinstance(rule, dict) and bool(rule.get("enabled", True)):
                        continue
                elif filter_label == "Silenciadas":
                    if str(entry.get("rule_id", "")) not in self._muted_rule_ids:
                        continue
                elif filter_label == "Disparando ahora":
                    if entry.get("state_key") != "active":
                        continue
                elif filter_label == "Pendientes":
                    if entry.get("state_key") != "pending":
                        continue
                if query:
                    haystack = " ".join(
                        str(part)
                        for part in (
                            entry.get("display_name", ""),
                            entry.get("class_name", ""),
                            entry.get("condition_summary", ""),
                            entry.get("actions_summary", ""),
                        )
                    ).lower()
                    if query not in haystack:
                        continue
                visible_view.append(entry)

            visible_view = self._sort_rules_view(visible_view)

            available_ids: set[str] = set()
            for idx, entry in enumerate(visible_view):
                rule_id = str(entry.get("rule_id", ""))
                available_ids.add(rule_id)
                row_tag = "row_even" if idx % 2 == 0 else "row_odd"
                tag = self._get_rule_tag(str(entry.get("state_key", "")))
                self.tree_rules.insert(
                    "",
                    "end",
                    iid=rule_id,
                    values=(
                        entry.get("now_value", ""),
                        entry.get("on_value", ""),
                        entry.get("display_name", ""),
                        entry.get("class_name", ""),
                        entry.get("condition_summary", ""),
                        entry.get("priority", 0),
                        entry.get("actions_summary", ""),
                    ),
                    tags=(row_tag, tag) if tag else (row_tag,),
                )
            if existing_selection and existing_selection not in available_ids:
                self._selected_rule_id = None
            if self._selected_rule_id and self._selected_rule_id not in available_ids:
                self._selected_rule_id = None
            try:
                self.tree_rules.selection_remove(self.tree_rules.selection())
            except Exception:
                pass
            if self._selected_rule_id and self._selected_rule_id in available_ids:
                try:
                    self.tree_rules.selection_set(self._selected_rule_id)
                    self.tree_rules.focus(self._selected_rule_id)
                except Exception:
                    self._selected_rule_id = None
            self._update_rule_preview()
            self._update_rules_status()

        def _get_rules_search_query(self) -> str:
            if getattr(self, "_rules_search_placeholder_active", False):
                return ""
            query = (self._rules_search_var.get() if hasattr(self, "_rules_search_var") else "").strip()
            if not query or query == getattr(self, "_rules_search_placeholder", ""):
                return ""
            return query.lower()

        def _get_rules_filter_key(self) -> str:
            label = (self._rules_filter_var.get() if hasattr(self, "_rules_filter_var") else "Todas").strip()
            return label or "Todas"

        def _sort_treeview(self, column: str) -> None:
            if column == self._rules_sort_col:
                self._rules_sort_desc = not self._rules_sort_desc
            else:
                self._rules_sort_col = column
                self._rules_sort_desc = column in {"priority", "now", "on"}
            self._refresh_rules_list()

        def _sort_rules_view(self, rules_view: list[dict[str, object]]) -> list[dict[str, object]]:
            column = self._rules_sort_col
            if not column:
                return rules_view

            def _key(entry: dict[str, object]) -> object:
                if column == "priority":
                    return int(entry.get("priority", 0))
                if column == "now":
                    return 1 if entry.get("state_key") == "active" else 0
                if column == "on":
                    return 1 if entry.get("on_value") == "ON" else 0
                if column == "name":
                    return str(entry.get("display_name", "")).lower()
                if column == "class":
                    return str(entry.get("class_name", "")).lower()
                if column == "condition":
                    return str(entry.get("condition_summary", "")).lower()
                if column == "actions":
                    return str(entry.get("actions_summary", "")).lower()
                return str(entry.get(column, "")).lower()

            return sorted(rules_view, key=_key, reverse=self._rules_sort_desc)

        def _get_rule_tag(self, state_key: str) -> str:
            if state_key == "active":
                return "active_rule"
            if state_key == "pending":
                return "rule_pending"
            if state_key == "muted":
                return "rule_muted"
            if state_key == "disabled":
                return "rule_disabled"
            return ""

        def _on_rules_search_focus_in(self, event: object | None = None) -> None:
            if getattr(self, "_rules_search_placeholder_active", False):
                self._clear_rules_search_placeholder()

        def _on_rules_search_focus_out(self, event: object | None = None) -> None:
            if not self._rules_search_var.get().strip():
                self._set_rules_search_placeholder()

        def _on_rules_search_change(self) -> None:
            if getattr(self, "_updating_ui", False):
                return
            self._refresh_rules_list()

        def _set_rules_search_placeholder(self) -> None:
            self._rules_search_placeholder_active = True
            self._rules_search_var.set(self._rules_search_placeholder)
            try:
                self.entry_rules_search.configure(style="RulesSearchPlaceholder.TEntry")
            except Exception:
                pass

        def _clear_rules_search_placeholder(self) -> None:
            self._rules_search_placeholder_active = False
            self._rules_search_var.set("")
            try:
                self.entry_rules_search.configure(style="RulesSearch.TEntry")
            except Exception:
                pass

        def _on_rules_clear(self) -> None:
            if hasattr(self, "combo_rules_filter"):
                self.combo_rules_filter.set("Todas")
            if hasattr(self, "_rules_filter_var"):
                self._rules_filter_var.set("Todas")
            self._set_rules_search_placeholder()
            self._refresh_rules_list()

        def _find_rule_payload(self, rule_id: str) -> dict[str, object] | None:
            # Búsqueda O(1) vía cache
            return self._rules_by_id_cache.get(str(rule_id))

        def _toggle_rule_enabled(self, rule_id: str) -> None:
            rule = self._find_rule_payload(rule_id)
            if rule is None:
                return
            rule["enabled"] = not bool(rule.get("enabled", True))
            self._persist_rules()

        def _on_rules_click(self, event: object) -> str | None:
            if event is None or not hasattr(event, "x") or not hasattr(event, "y"):
                return None
            row_id = self.tree_rules.identify_row(event.y)
            if not row_id:
                return None
            col = self.tree_rules.identify_column(event.x)
            if col == "#2":
                try:
                    self.tree_rules.selection_set(row_id)
                except Exception:
                    pass
                self._selected_rule_id = row_id
                self._toggle_rule_enabled(row_id)
                return "break"
            return None

        def _on_rule_double_click(self, event: object | None = None) -> None:
            if event is None or not hasattr(event, "x") or not hasattr(event, "y"):
                return
            row_id = self.tree_rules.identify_row(event.y)
            if not row_id:
                return
            col = self.tree_rules.identify_column(event.x)
            if col == "#2":
                return
            try:
                self.tree_rules.selection_set(row_id)
                self.tree_rules.focus(row_id)
            except Exception:
                pass
            self._selected_rule_id = row_id
            self._on_rule_edit()

        def _set_rule_preview_text(self, text: str) -> None:
            try:
                self.rule_preview_text.configure(state="normal")
                self.rule_preview_text.delete("1.0", "end")
                self.rule_preview_text.insert("1.0", text)
                self.rule_preview_text.configure(state="disabled")
            except Exception:
                pass

        def _update_rule_preview(self) -> None:
            rule_id = self._selected_rule_id
            if not rule_id:
                self._set_rule_preview_text(self._rule_preview_empty)
                return
            rule = self._find_rule_payload(rule_id)
            if not rule:
                self._set_rule_preview_text(self._rule_preview_empty)
                return
            state_key = self._resolve_rule_state(rule_id)
            self._set_rule_preview_text(self._build_rule_preview_text(rule, state_key))

        def _build_rule_preview_text(self, rule: dict[str, object], state_key: str) -> str:
            name = self._format_rule_name(rule, state_key)
            enabled = bool(rule.get("enabled", True))
            now_value = "SI" if state_key == "active" else "NO"
            on_value = "ON" if enabled else "OFF"
            condition = rule.get("condition", {}) if isinstance(rule.get("condition"), dict) else {}
            class_name = str(condition.get("class_name", ""))
            condition_summary = self._summarize_condition(condition)
            actions_summary = self._summarize_actions(rule.get("actions"))
            actions_lines = [part.strip() for part in actions_summary.split(";") if part.strip()]
            if not actions_lines:
                actions_lines = ["(sin acciones)"]

            lines = [
                f"Nombre: {name}",
                f"ON: {on_value} | Ahora: {now_value}",
                f"Clase: {class_name or '-'}",
                f"Condicion: {condition_summary}",
                "Acciones:",
            ]
            lines.extend(f"- {line}" for line in actions_lines)
            return "\n".join(lines)

        def _format_rule_name(self, rule: dict[str, object], state_key: str | None = None) -> str:
            name = str(rule.get("name", "Regla"))
            if state_key == "disabled" or not bool(rule.get("enabled", True)):
                return f"{name} (desactivada)"
            if state_key == "muted":
                return f"{name} (pausada)"
            if state_key == "pending":
                return f"{name} (pendiente)"
            return name

        def _summarize_condition(self, condition: dict[str, object]) -> str:
            if not isinstance(condition, dict):
                return "-"
            kind = str(condition.get("kind") or CONDITION_KIND_VISION).strip().lower()
            
            if kind == CONDITION_KIND_PLC_BIT:
                label = str(condition.get("label") or condition.get("tag", "")).strip()
                area = str(condition.get("area", "M")).upper()
                byte_idx = condition.get("byte_index")
                bit_idx = condition.get("bit_index")
                expected = "1 (activo)" if _ensure_bool(condition.get("expected_value"), default=True) else "0 (inactivo)"
                addr = f"{area}{byte_idx}.{bit_idx}" if byte_idx is not None and bit_idx is not None else ""
                desc = f"Señal de entrada: '{label}'" if label else "Señal de entrada PLC"
                if addr:
                    desc += f" en {addr}"
                return f"SI {desc} esta en {expected}"

            # Regla (Meta-regla)
            if kind == CONDITION_KIND_RULE:
                target_label = str(condition.get("label", "")).strip()
                rid = str(condition.get("rule_id", ""))
                if not target_label:
                    target_label = rid[:8]
                min_f = condition.get("min_firings", 1)
                max_f = condition.get("max_firings")
                win = condition.get("window_sec", 60)
                
                count_str = f"{min_f}"
                if max_f:
                    count_str += f"-{max_f}"
                    
                return f"SI regla '{target_label}' dispara {count_str} veces en {win}s"

            # PLC Numérico
            if condition.get("plc_mode") == "numeric":
                addr = str(condition.get("address", "")).strip()
                op = str(condition.get("operator", "="))
                val1 = str(condition.get("value1", ""))
                return f"Lectura PLC ({addr}) es {op} {val1}"

            # Visión
            if bool(condition.get("detection_only")):
                return "Simple detección de presencia"
            
            parts: list[str] = []
            min_count = condition.get("min_count")
            max_count = condition.get("max_count")
            window_sec = condition.get("window_sec")
            min_conf = condition.get("min_conf")
            
            # Conteo y Lógica Humana
            try:
                mc = int(min_count) if min_count not in (None, "") else None
                mxc = int(max_count) if max_count not in (None, "") else None
                
                if mc == 0 and mxc == 0:
                    parts.append("Superficie libre (Sin detecciones)")
                elif mc == 1 and mxc in (None, ""):
                    parts.append("Presencia al menos puntual")
                elif mc is not None and mxc is not None and mc == mxc:
                    parts.append(f"Exactamente {mc} unidades")
                elif mc is not None:
                    txt = f"Al menos {mc} unidades"
                    if mxc is not None: txt += f" (máximo {mxc})"
                    parts.append(txt)
            except Exception:
                if min_count: parts.append(f"Min: {min_count}")
                if max_count: parts.append(f"Max: {max_count}")

            if window_sec:
                parts.append(f"observado durante {window_sec}s")
            
            if min_conf:
                try:
                    parts.append(f"(Confianza > {float(min_conf)*100:.0f}%)")
                except Exception:
                    pass

            # Sectores
            sector = condition.get("sector")
            sector_mode = condition.get("sector_mode", "aggregate")
            if sector is not None:
                if isinstance(sector, list):
                    s_list = [str(s + 1 if isinstance(s, int) else s) for s in sector]
                    parts.append(f"en sector/es {', '.join(s_list)}")
                else:
                    s_val = sector + 1 if isinstance(sector, int) else sector
                    parts.append(f"en sector {s_val}")
            
            if sector_mode == "any":
                parts.append("[Evaluado carril por carril]")

            return " | ".join(parts) if parts else "Cualquier objeto detectado"

        def _on_rule_select(self, event: object | None = None) -> None:
            selection = self.tree_rules.selection()
            if selection:
                self._selected_rule_id = selection[0]
                try:
                    self.tree_rules.focus(self._selected_rule_id)
                except Exception:
                    pass
            else:
                self._selected_rule_id = None
            self._update_rule_preview()

        def _resolve_rule_state(
            self,
            rule_id: str,
            *,
            monitoring_ready: bool | None = None,
            active_ids: set[str] | None = None,
        ) -> str:
            ready = self._rules_monitoring_ready if monitoring_ready is None else monitoring_ready
            matches = self._active_rule_ids if active_ids is None else active_ids
            if not ready:
                return "pending"
            for rule in self._rules_payload:
                if str(rule.get("rule_id")) == str(rule_id):
                    if not bool(rule.get("enabled", True)):
                        return "disabled"
                    break
            if str(rule_id) in getattr(self, "_muted_rule_ids", set()):
                return "muted"
            return "active" if str(rule_id) in matches else "idle"

        def _apply_rule_states(self) -> None:
            # Si hay un filtro o una búsqueda activa, comprobamos si el cambio de estado
            # afecta a qué reglas deben ser visibles.
            filter_key = self._get_rules_filter_key()
            search_query = self._get_rules_search_query()
            
            if filter_key != "Todas" or search_query:
                # Calculamos qué reglas DEBERÍAN verse ahora
                current_items = set(self.tree_rules.get_children(""))
                new_visible_ids = set()
                
                # Reutilizamos parte de la lógica de _refresh_rules_list de forma optimizada
                for rule in self._rules_payload:
                    rule_id = str(rule.get("rule_id", ""))
                    state_key = self._resolve_rule_state(rule_id)
                    
                    # Filtro de categoría
                    if filter_key == "Pausadas" and bool(rule.get("enabled", True)): continue
                    if filter_key == "Silenciadas" and rule_id not in self._muted_rule_ids: continue
                    if filter_key == "Disparando ahora" and state_key != "active": continue
                    if filter_key == "Pendientes" and state_key != "pending": continue
                    
                    # Filtro de búsqueda
                    if search_query:
                        entry = self._rules_display_cache.get(rule_id)
                        if not entry: # Si no está en cache, forzamos un refresh total y salimos
                            self._refresh_rules_list()
                            return
                        haystack = " ".join(str(part) for part in (
                            entry.get("display_name", ""),
                            entry.get("class_name", ""),
                            entry.get("condition_summary", ""),
                            entry.get("actions_summary", ""),
                        )).lower()
                        if search_query not in haystack:
                            continue
                    
                    new_visible_ids.add(rule_id)
                
                # Si el conjunto de reglas visibles ha cambiado (alguien entró o salió del filtro),
                # entonces sí hacemos el refresh pesado.
                if new_visible_ids != current_items:
                    self._refresh_rules_list()
                    return

            # Caso base u optimizado: los mismos elementos siguen visibles, solo actualizamos sus valores in-place.
            for item in self.tree_rules.get_children(""):
                state_key = self._resolve_rule_state(item)
                self._rule_states[item] = state_key
                rule = self._find_rule_payload(item)
                
                if state_key == "active":
                    now_value = "● SI"
                else:
                    now_value = "  NO"
                
                on_value = "ON" if rule and bool(rule.get("enabled", True)) else "OFF"
                display_name = self._format_rule_name(rule, state_key) if rule else str(item)
                
                # Solo actualizar si el valor ha cambiado (Tkinter Treeview es sensible a escrituras)
                curr = self.tree_rules.item(item, "values")
                if curr:
                     # El orden en Columnas es: (now, on, name, class, condition, priority, actions)
                     if curr[0] != now_value or curr[1] != on_value or curr[2] != display_name:
                        self.tree_rules.set(item, "now", now_value)
                        self.tree_rules.set(item, "on", on_value)
                        self.tree_rules.set(item, "name", display_name)
                
                row_tag = "row_even" if self.tree_rules.index(item) % 2 == 0 else "row_odd"
                tag = self._get_rule_tag(state_key)
                new_tags = (row_tag, tag) if tag else (row_tag,)
                if self.tree_rules.item(item, "tags") != new_tags:
                    self.tree_rules.item(item, tags=new_tags)
            
            self._update_rule_preview()
            self._update_rules_status()

        def _refresh_rules_actions(self) -> None:
            # No acción directa necesaria, pero se puede actualizar tooltips en el futuro
            pass

        def _on_rule_add(self) -> None:
            available = self._ensure_latest_classes()
            if not available:
                messagebox.showinfo("Reglas", "Aún no hay clases disponibles. Selecciona los modelos en el detector.")
                return
            editor = _RuleEditorDialog(self.window, available, service=self.service)
            result = editor.show()
            if result is None:
                return

            # REGLA DE DOMINIO: Si el modo es "Por Sector" (any) y hay múltiples sectores,
            # dividimos la regla en N reglas individuales (una por sector).
            condition = result.get("condition", {})
            sector_mode = condition.get("sector_mode")
            sector_val = condition.get("sector")
            kind = condition.get("kind")

            if kind == CONDITION_KIND_VISION and sector_mode == "any" and isinstance(sector_val, list) and len(sector_val) > 1:
                base_name = str(result.get("name", "Regla"))
                for s in sector_val:
                    new_rule = copy.deepcopy(result)
                    # Ajustar nombre
                    new_rule["name"] = f"{base_name} (S{s})"
                    # Ajustar condición
                    new_cond = new_rule["condition"]
                    new_cond["sector"] = int(s)
                    # El modo sigue siendo "any" o "aggregate", da igual para 1 sector, 
                    # pero "aggregate" es más semántico para single sector.
                    new_cond["sector_mode"] = "aggregate"  
                    
                    # Generar nuevo ID para evitar colisiones
                    new_rule["rule_id"] = uuid.uuid4().hex
                    
                    self._rules_payload.append(new_rule)
            else:
                self._rules_payload.append(result)

            self._persist_rules()

        def _on_rule_edit(self) -> None:
            rule = self._get_selected_rule()
            if rule is None:
                messagebox.showinfo("Reglas", "Selecciona una regla para editar.")
                return
            available = self._ensure_latest_classes()
            editor = _RuleEditorDialog(self.window, available, rule, service=self.service)
            result = editor.show()
            if result is None:
                return
            for idx, existing in enumerate(self._rules_payload):
                if str(existing.get("rule_id")) == str(result.get("rule_id")):
                    self._rules_payload[idx] = result
                    break
            self._persist_rules()

        def _on_rule_duplicate(self) -> None:
            rule = self._get_selected_rule()
            if rule is None:
                messagebox.showinfo("Reglas", "Selecciona una regla para duplicar.")
                return
            duplicate = copy.deepcopy(rule)
            duplicate["rule_id"] = uuid.uuid4().hex
            duplicate["name"] = f"{duplicate.get('name', 'Regla')} (copia)"
            self._rules_payload.append(duplicate)
            self._persist_rules()

        def _on_rule_delete(self) -> None:
            rule = self._get_selected_rule()
            if rule is None:
                messagebox.showinfo("Reglas", "Selecciona una regla para eliminar.")
                return
            if messagebox and not messagebox.askyesno(
                "🗑 Eliminar regla", f"¿🗑 Eliminar la regla '{rule.get('name', 'Regla')}'?"
            ):
                return
            self._rules_payload = [r for r in self._rules_payload if r.get("rule_id") != rule.get("rule_id")]
            self._persist_rules()

        def _get_selected_rule(self) -> dict[str, object] | None:
            rule_id = self._selected_rule_id
            if not rule_id:
                selection = self.tree_rules.selection()
                if selection:
                    rule_id = selection[0]
            if not rule_id:
                return None
            for rule in self._rules_payload:
                if str(rule.get("rule_id")) == str(rule_id):
                    return copy.deepcopy(rule)
            return None

        def _persist_rules(self) -> None:
            try:
                self.service.replace_rules_from_payload(self._rules_payload)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("No se pudieron guardar las reglas: %s", exc)
            finally:
                self._load_rules_from_service()

        def _summarize_actions(self, actions: object) -> str:
            if not isinstance(actions, list) or not actions:
                return "(Sin acciones)"
            summaries: list[str] = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind")
                params = action.get("params") if isinstance(action.get("params"), dict) else {}
                
                if kind in {"block_classes", "mute_triggers", "block_triggers"}:
                    triggers = params.get("triggers", [])
                    labels: list[str] = []
                    if isinstance(triggers, (list, tuple, set)) and triggers:
                        for item in triggers:
                            if isinstance(item, dict):
                                labels.append(str(item.get("label", "")) or str(item.get("rule_id", "")))
                            else:
                                labels.append(str(item))
                    else:
                        classes = params.get("classes", [])
                        if isinstance(classes, (list, tuple, set)):
                            labels = [str(c) for c in classes]
                    
                    target_desc = ", ".join(labels) or "activaciones"
                    duration = params.get("duration_sec")
                    msg = f"IGNORAR temporalmente: {target_desc}"
                    if duration:
                        msg += f" durante {duration}s"
                    summaries.append(msg)
                
                elif kind == "force_manual_level":
                    level = params.get("level")
                    if isinstance(level, str):
                        summaries.append(f"FORZAR modo de nivel: {level}")
                
                elif kind == "resume_level":
                    level = params.get("level")
                    if isinstance(level, str):
                        summaries.append(f"REANUDAR nivel anterior ({level})")
                
                elif kind == "show_message":
                    text = str(params.get("text", "")).strip()
                    if text:
                        summaries.append(f"NOTIFICAR en pantalla: '{text}'")
                    else:
                        summaries.append("Mostrar aviso visual de detección")

                elif kind == "take_snapshot":
                    label = str(params.get("label", "")).strip()
                    desc = f"GUARDAR FOTO técnica '{label}'" if label else "GUARDAR captura de imagen"
                    summaries.append(desc)

                elif kind == "send_plc":
                    preset_id = str(params.get("preset_id", ""))
                    preset_name = ""
                    if hasattr(self, "_plc_presets_cache"):
                        for preset in getattr(self, "_plc_presets_cache", []):
                            if str(preset.get("preset_id")) == preset_id:
                                preset_name = str(preset.get("name", ""))
                                break
                    
                    area = str(params.get("area", "M")).upper()
                    targets = params.get("targets")
                    loc_desc = f" [{preset_name}]" if preset_name else ""
                    
                    if isinstance(targets, list) and targets:
                        items_desc: list[str] = []
                        for item in targets:
                            if not isinstance(item, dict): continue
                            byte_idx = item.get("byte_index")
                            bit_idx = item.get("bit_index")
                            val = 1 if _ensure_bool(item.get("value", params.get("value", True))) else 0
                            if byte_idx is not None and bit_idx is not None:
                                items_desc.append(f"{area}{byte_idx}.{bit_idx} -> {val}")
                        
                        if items_desc:
                            summaries.append(f"COMUNICAR al PLC{loc_desc}: " + ", ".join(items_desc))
                            continue
                    
                    byte_idx = params.get("byte_index", params.get("byte"))
                    bit_idx = params.get("bit_index", params.get("bit"))
                    val = 1 if _ensure_bool(params.get("value", True)) else 0
                    if byte_idx is not None and bit_idx is not None:
                        summaries.append(f"COMUNICAR al PLC{loc_desc}: {area}{byte_idx}.{bit_idx} -> {val}")
                    else:
                        summaries.append(f"COMUNICAR al PLC{loc_desc}")
                
                else:
                    summaries.append(str(kind).upper())
            
            return "; ".join(summaries) if summaries else "(Sin acciones configuradas)"

        def _on_rule_export_pdf(self) -> None:
            if FPDF is None:
                if messagebox:
                    messagebox.showerror(
                        "Exportar PDF",
                        "No se puede exportar porque la librería 'fpdf' no está instalada.\n"
                        "Instálala con 'pip install fpdf2' y vuelve a intentarlo.",
                    )
                return

            self._load_rules_from_service()
            active_rules = [rule for rule in self._rules_payload if bool(rule.get("enabled", True))]
            if not active_rules:
                if messagebox:
                    messagebox.showinfo("Exportar PDF", "No hay reglas activas en este perfil para exportar.")
                return

            profile_id = self._current_profile_id or ""
            profile_info = self._profiles_cache.get(profile_id, {}) if hasattr(self, "_profiles_cache") else {}
            profile_name = str(profile_info.get("name", profile_id or "Perfil"))
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", profile_name).strip("_") or "perfil"
            default_filename = f"reglas_{safe_name}.pdf"

            filepath = filedialog.asksaveasfilename(
                parent=self.window,
                title="💾 Guardar reglas como PDF",
                defaultextension=".pdf",
                filetypes=[("Documento PDF", "*.pdf"), ("Todos los archivos", "*.*")],
                initialfile=default_filename,
            )
            if not filepath:
                return

            try:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_left_margin(15)
                pdf.set_right_margin(15)
                pdf.add_page()
                pdf.set_title(f"Configuracion de Reglas - {profile_name}")

                def _txt(text: object) -> str:
                    raw = str(text) if text is not None else ""
                    # ASCII clean for Latin-1 encoding
                    cleaned = raw.replace("•", "- ").replace("→", "->").replace("●", "[*]")
                    cleaned = cleaned.replace("©", "(c)").replace("™", "(tm)")
                    try:
                        return cleaned.encode("latin-1", "replace").decode("latin-1")
                    except Exception:  # noqa: BLE001
                        return cleaned

                content_width = max(40, pdf.w - pdf.l_margin - pdf.r_margin)

                def _get_tree_summary(node: dict[str, Any], depth: int = 0) -> str:
                    if not isinstance(node, dict):
                        return "-"
                    ntype = str(node.get("type", "")).lower()
                    neg = "NO " if node.get("negated") else ""
                    
                    if ntype == "group":
                        op_raw = str(node.get("operator", "and")).upper()
                        op_map = {"AND": " Y ademas ", "OR": " O bien "}
                        op = op_map.get(op_raw, f" {op_raw} ")
                        
                        children = node.get("children", [])
                        if not children:
                            return f"{neg}(Sin criterios configurados)"
                        summaries = [_get_tree_summary(c, depth + 1) for c in children if isinstance(c, dict)]
                        inner = op.join(summaries)
                        return f"{neg}({inner})"
                    
                    if ntype == "condition" or "condition" in node:
                        cond = node.get("condition")
                        if isinstance(cond, dict):
                            summary = self._summarize_condition(cond)
                            # Clarificadores para el árbol
                            if " entrada:" in summary: summary = summary.replace("Señal de entrada: ", "PLC: ")
                            return f"{neg}{summary}"
                    return "-"

                # Estilo de cabecera de página
                pdf.set_font("Helvetica", "B", 20)
                pdf.set_text_color(31, 139, 75)
                pdf.cell(content_width, 15, _txt("ESPECIFICACION DE REGLAS DE CONTROL"), ln=True, align="C")
                
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(content_width, 8, _txt(f"Perfil seleccionado: {profile_name}"), ln=True, align="C")
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(content_width, 6, _txt(f"Documento generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M:%S')}"), ln=True, align="C")
                pdf.ln(10)

                sorted_rules = sorted(active_rules, key=lambda r: int(r.get("priority", 0)), reverse=True)
                for idx, rule in enumerate(sorted_rules, start=1):
                    name = rule.get("name", f"Regla {idx}")
                    description = str(rule.get("description", "")).strip()

                    # CABECERA DE REGLA
                    pdf.set_fill_color(240, 240, 240)
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(content_width, 10, _txt(f" {idx}. OBJETIVO: {name.upper()}"), ln=True, fill=True)
                    
                    # DESCRIPCION HUMANA
                    if description:
                        pdf.set_font("Helvetica", "", 10)
                        pdf.set_text_color(60, 60, 60)
                        pdf.set_x(pdf.l_margin + 5)
                        clean_desc = description.replace("**", "").replace("*", "").strip()
                        pdf.multi_cell(content_width - 8, 5, _txt(clean_desc))
                        pdf.ln(1)

                    # LOGICA DE DISPARO
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(29, 55, 97)
                    pdf.set_x(pdf.l_margin + 5)
                    pdf.cell(25, 7, _txt("CRITERIOS:"), border=0)
                    
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(0, 0, 0)
                    
                    tree = rule.get("condition_tree")
                    condition = rule.get("condition") if isinstance(rule.get("condition"), dict) else {}
                    
                    if tree and isinstance(tree, dict) and tree.get("children"):
                        logic_text = _get_tree_summary(tree)
                        if logic_text.startswith("("): logic_text = logic_text[1:-1]
                        # Indentar criterios complejos
                        pdf.set_x(pdf.l_margin + 30)
                        pdf.multi_cell(content_width - 32, 5, _txt(logic_text))
                    else:
                        pdf.set_x(pdf.l_margin + 30)
                        pdf.cell(0, 7, _txt(self._summarize_condition(condition)), ln=True)

                    # OBJETO/CLASE (si aplica)
                    class_name = condition.get("class_name", "")
                    if class_name:
                        pdf.set_font("Helvetica", "B", 8)
                        pdf.set_text_color(100, 100, 100)
                        pdf.set_x(pdf.l_margin + 30)
                        pdf.cell(0, 5, _txt(f"[Clase Vision: {class_name}]"), ln=True)

                    # RESPUESTA DEL SISTEMA
                    pdf.ln(1)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(106, 44, 0)
                    pdf.set_x(pdf.l_margin + 5)
                    pdf.cell(25, 7, _txt("RESPUESTA:"), border=0)
                    
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(0, 0, 0)
                    
                    actions_summary = self._summarize_actions(rule.get("actions"))
                    actions_list = [a.strip() for a in actions_summary.split(";") if a.strip()]
                    
                    pdf.set_x(pdf.l_margin + 30)
                    if not actions_list or actions_summary == "(Sin acciones)":
                        pdf.cell(0, 7, _txt("No se han definido acciones para esta regla."), ln=True)
                    else:
                        curr_y = pdf.get_y()
                        prefix = "- "
                        for i, action in enumerate(actions_list):
                            pdf.set_x(pdf.l_margin + 30)
                            pdf.multi_cell(content_width - 32, 5, _txt(f"{prefix}{action}"))
                    
                    pdf.ln(5)
                    # Linea de separacion elegante
                    pdf.set_draw_color(230, 230, 230)
                    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
                    pdf.ln(4)

                    if pdf.get_y() > 255:
                        pdf.add_page()

                pdf.output(filepath)
            except Exception as exc:  # noqa: BLE001
                if messagebox:
                    messagebox.showerror(
                        "Exportar PDF",
                        f"No se pudo guardar el PDF.\nDetalle: {exc}",
                    )
                return

            if messagebox:
                messagebox.showinfo(
                    "Exportar PDF",
                    f"Se exportaron {len(active_rules)} reglas activas a:\n{filepath}",
                )

        def _update_rules_status(self) -> None:
            total_rules = self._rules_total_count
            paused_rules = self._rules_paused_count
            if self._rules_monitoring_ready:
                active_rules = len(self._active_rule_ids.intersection(self._rules_by_id_cache.keys()))
            else:
                active_rules = 0
            new_text = f"Reglas: {total_rules} | Disparando: {active_rules} | Pausadas: {paused_rules}"
            if self.rule_status_var.get() != new_text:
                self.rule_status_var.set(new_text)

# ----------------------------------------------------------------------
# Diálogos auxiliares
# ----------------------------------------------------------------------

class _RuleEditorDialog:
    _condition_kind_labels: dict[str, str] = {
        CONDITION_KIND_VISION: "Detección de visión",
        CONDITION_KIND_PLC_BIT: "Lectura de PLC",
        CONDITION_KIND_RULE: "Disparo de regla",
    }
    _operator_options: tuple[tuple[str, str], ...] = (
        ("and", "Se cumplen todas (Y)"),
        ("or", "Se cumple cualquiera (O)"),
    )
    _info_section_bg = "#e3f6e0"
    _info_section_border = "#3a8b4b"
    _trigger_section_bg = "#d7e9ff"
    _trigger_section_border = "#5c82d1"
    _reaction_section_bg = "#ffe6d1"
    _reaction_section_border = "#d78233"

    def __init__(
        self,
        master: tk.Misc,
        available_classes: Sequence[str],
        payload: dict[str, object] | None = None,
        *,
        service: "SendToPLCService | None" = None,
    ) -> None:
        self.available_classes = tuple(available_classes)
        self._orig_payload = copy.deepcopy(payload) if payload else None
        self.result: dict[str, object] | None = None
        self._service = service
        self._orig_actions_cache: list[dict[str, object]] = []
        self._snapshot_dialog_payload: dict[str, object] | None = None
        self._message_dialog_payload: dict[str, object] | None = None
        self._plc_action_payload: dict[str, object] | None = None
        self._plc_presets_cache: list[dict[str, object]] = self._fetch_plc_presets()
        self._condition_kinds_by_label = {label: kind for kind, label in self._condition_kind_labels.items()}
        (
            self._plc_preset_labels,
            self._plc_preset_by_label,
            self._plc_preset_label_by_id,
        ) = self._build_preset_options()
        self._operator_label_to_value = {label: value for value, label in self._operator_options}
        self._operator_value_to_label = {value: label for value, label in self._operator_options}
        self._mute_trigger_choices: list[tuple[str, str]] = []
        self._selected_mute_triggers: set[str] = set()

        self.var_condition_kind = tk.StringVar(value=CONDITION_KIND_VISION)
        self.var_condition_kind_label = tk.StringVar(value=self._condition_kind_labels[CONDITION_KIND_VISION])

        # Campos condición visión
        self.var_class = tk.StringVar()
        self.var_detection_only = tk.BooleanVar(value=False)
        self.var_min_count = tk.IntVar(value=1)
        self.var_max_count = tk.StringVar(value="")
        self.var_min_area = tk.StringVar()
        self.var_max_area = tk.StringVar()
        self.var_area_unit = tk.StringVar(value="px")
        self.var_min_conf = tk.StringVar()
        self.var_window_sec = tk.IntVar(value=int(WINDOW_SHORT_SEC))
        self.var_sector = tk.StringVar()  # NUEVO: sector(es) separados por coma
        
        # Filtro de sectores (selector visual)
        self.var_sector_filter_mode = tk.StringVar(value="all")  # "all" o "selected"
        self.var_sector_eval_mode = tk.StringVar(value="aggregate")  # "aggregate" o "any"
        self._sector_checkboxes: dict[int, tk.BooleanVar] = {}
        self._available_sectors: list[int] = list(range(1, 11))  # Sectores 1-10 por defecto

        # Campos condición PLC
        self.var_plc_condition_preset = tk.StringVar()
        self.var_plc_condition_ip = tk.StringVar()
        self.var_plc_condition_area = tk.StringVar(value="M")
        self.var_plc_condition_rack = tk.IntVar(value=0)
        self.var_plc_condition_slot = tk.IntVar(value=2)
        self.var_plc_condition_db = tk.StringVar()
        self.var_plc_condition_byte = tk.IntVar(value=0)
        self.var_plc_condition_bit = tk.IntVar(value=0)
        self.var_plc_condition_expected = tk.StringVar(value="1")
        self.var_plc_condition_label = tk.StringVar()
        self.var_plc_manual_enabled = tk.BooleanVar(value=not self._plc_preset_labels)
        
        # Nuevos campos para PLC Numérico
        self.var_plc_type = tk.StringVar(value="bit")  # "bit" o "numeric"
        self.var_plc_numeric_address = tk.StringVar()  # ej: DB6.DBW0
        self.var_plc_numeric_type = tk.StringVar(value="WORD")
        self.var_plc_operator = tk.StringVar(value="=")
        self.var_plc_value1 = tk.StringVar()
        self.var_plc_value2 = tk.StringVar()
        
        self._plc_manual_row: int | None = None

        # Campos condición de Regla (meta-regla)
        self.var_rule_target = tk.StringVar()  # ID de regla a monitorear
        self.var_rule_min_firings = tk.IntVar(value=1)
        self.var_rule_max_firings = tk.StringVar(value="")  # "" = sin límite
        self.var_rule_window_sec = tk.IntVar(value=60)
        self.var_rule_debounce_ms = tk.IntVar(value=0)  # 0 = sin debounce
        self.var_rule_cooldown_sec = tk.IntVar(value=0)  # 0 = sin cooldown
        self.var_rule_label = tk.StringVar()
        self._rule_target_choices: list[tuple[str, str]] = []  # [(rule_id, label), ...]

        # Estado del constructor de condiciones
        self._condition_tree: dict[str, object] = _make_condition_group("and")
        self._editing_node_id: str | None = None
        self.var_root_operator = tk.StringVar(value="and")
        self.var_root_negated = tk.BooleanVar(value=False)
        self.var_condition_summary = tk.StringVar(value="Sin condiciones")
        self.var_condition_hint = tk.StringVar(value="Añade condiciones con los botones inferiores.")
        self._trigger_placeholder_active = False
        self._blink_save_job: str | None = None
        self._blink_state = False
        self._held_button: tk.Button | None = None

        self.window = tk.Toplevel(master)
        self.window.title("Editor de regla")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(True, True)
        self.window.geometry("820x790")
        self.window.minsize(800, 520)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_ui()
        self._populate_from_payload(self._orig_payload)

    def show(self) -> dict[str, object] | None:
        self.window.wait_window()
        return self.result

    def _build_preset_options(self) -> tuple[tuple[str, ...], dict[str, dict[str, object]], dict[str, str]]:
        labels: list[str] = []
        by_label: dict[str, dict[str, object]] = {}
        label_by_id: dict[str, str] = {}
        seen: set[str] = set()
        for item in self._plc_presets_cache:
            if not isinstance(item, dict):
                continue
            preset_id = str(item.get("preset_id", "")).strip()
            display = str(item.get("name", "")).strip()
            label = display or preset_id
            if not label:
                continue
            base_label = label
            counter = 1
            while label in seen:
                counter += 1
                label = f"{base_label} ({counter})"
            seen.add(label)
            labels.append(label)
            by_label[label] = item
            if preset_id:
                label_by_id[preset_id] = label
        return tuple(labels), by_label, label_by_id

    def _build_ui(self) -> None:
        container = ttk.Frame(self.window)
        container.grid(row=0, column=0, sticky="nsew")
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        frame = ttk.Frame(canvas, padding=12)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(event):
            # 1. Verificar que el evento pertenece a esta ventana
            try:
                if not self.window.winfo_exists():
                    return
                if event.widget and str(event.widget.winfo_toplevel()) != str(self.window):
                    return
            except Exception:
                return

            # 2. Impedir que el scroll se propague si estamos sobre un widget interno con scroll propio
            widget = event.widget
            if widget:
                try:
                    w_class = widget.winfo_class()
                    # Bloquear solo si es un widget de scroll (Lista o Texto)
                    if w_class in ("Listbox", "Text", "Treeview"):
                        return
                    # También si el padre o abuelo es un contenedor de scroll (como ScrolledText)
                    parent_id = widget.winfo_parent()
                    if parent_id:
                        parent = widget.nametowidget(parent_id)
                        if parent.winfo_class() in ("Listbox", "Text", "Treeview") or "scrolled" in str(parent).lower():
                            return
                except Exception:
                    pass

            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Usamos el despachador robusto compartido
        self.window.bind_all("<MouseWheel>", lambda e: _handle_mousewheel_event(e, canvas, self.window))
        self.window.bind("<Destroy>", lambda _: self.window.unbind_all("<MouseWheel>"))

        row = 0
        info_wrapper = tk.Frame(
            frame,
            background=self._info_section_bg,
            highlightbackground=self._info_section_border,
            highlightcolor=self._info_section_border,
            highlightthickness=2,
            bd=0,
        )
        info_wrapper.grid(row=row, column=0, columnspan=2, sticky="nsew")
        info_wrapper.columnconfigure(0, weight=1)

        info_header = tk.Label(
            info_wrapper,
            text="Datos generales de la regla",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
            background=self._info_section_bg,
            foreground="#1f6632",
        )
        info_header.grid(row=0, column=0, sticky="we", padx=12, pady=(10, 0))

        info_frame = tk.Frame(info_wrapper, background=self._info_section_bg)
        info_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))
        info_frame.columnconfigure(1, weight=1)

        self._label_with_info(info_frame, 0, 0, "Nombre:", "sendToPLC.rule_info.name", sticky="w")
        self.var_name = tk.StringVar(value="Nueva regla")
        ttk.Entry(info_frame, textvariable=self.var_name).grid(row=0, column=1, sticky="we", padx=(6, 0))

        self._label_with_info(
            info_frame,
            1,
            0,
            "Habilitada:",
            "sendToPLC.rule_info.enabled",
            sticky="w",
            pady=(6, 0),
        )
        self.var_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(info_frame, variable=self.var_enabled).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        self._label_with_info(
            info_frame,
            2,
            0,
            "Prioridad:",
            "sendToPLC.rule_info.priority",
            sticky="w",
            pady=(6, 0),
        )
        self.var_priority = tk.IntVar(value=0)
        ttk.Spinbox(info_frame, from_=-100, to=100, textvariable=self.var_priority, width=6).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        self._label_with_info(
            info_frame,
            3,
            0,
            "Descripción:",
            "sendToPLC.rule_info.description",
            sticky="nw",
            pady=(6, 0),
        )
        if ScrolledText is not None:
            self.txt_description = ScrolledText(info_frame, height=4, wrap="word")
        else:
            self.txt_description = tk.Text(info_frame, height=4, wrap="word")  # type: ignore[assignment]
        self.txt_description.grid(row=3, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))
        info_frame.rowconfigure(3, weight=1)

        row += 1
        ttk.Separator(frame).grid(row=row, column=0, columnspan=2, sticky="we", pady=10)
        row += 1

        # NUEVO: Resumen visible de la regla
        summary_wrapper = tk.Frame(
            frame,
            background="#FFF8E1",
            highlightbackground="#FFE082",
            highlightcolor="#FFE082",
            highlightthickness=2,
            bd=0,
        )
        summary_wrapper.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        summary_wrapper.columnconfigure(0, weight=1)

        self.var_rule_summary = tk.StringVar(value="Esta regla: (añade condiciones y acciones)")
        self.lbl_rule_summary = tk.Label(
            summary_wrapper,
            textvariable=self.var_rule_summary,
            anchor="w",
            wraplength=860,
            justify="left",
            font=("Segoe UI", 10, "bold"),
            background="#FFF8E1",
            foreground="#5D4037",
            padx=12,
            pady=8,
        )
        self.lbl_rule_summary.grid(row=0, column=0, sticky="we")

        row += 1

        trigger_wrapper = tk.Frame(
            frame,
            background=self._trigger_section_bg,
            highlightbackground=self._trigger_section_border,
            highlightcolor=self._trigger_section_border,
            highlightthickness=2,
            bd=0,
        )
        trigger_wrapper.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        trigger_wrapper.columnconfigure(0, weight=1)
        trigger_wrapper.rowconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

        trigger_header = tk.Label(
            trigger_wrapper,
            text="Condiciones que deben cumplirse",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
            background=self._trigger_section_bg,
            foreground="#1d3761",
        )
        trigger_header.grid(row=0, column=0, sticky="we", padx=12, pady=(10, 0))

        triggers_frame = tk.Frame(trigger_wrapper, background=self._trigger_section_bg)
        triggers_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))
        triggers_frame.columnconfigure(1, weight=1)

        builder_container = ttk.Frame(triggers_frame)
        builder_container.grid(row=0, column=0, columnspan=2, sticky="nsew")
        builder_container.columnconfigure(0, weight=1)
        self._build_condition_builder(builder_container, 0)

        builder_end_row = getattr(self, "_condition_builder_last_row", 0)
        trigger_controls_row = builder_end_row + 1

        self._trigger_content_holder = ttk.Frame(triggers_frame)
        self._trigger_content_holder.grid(row=trigger_controls_row, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self._trigger_content_holder.columnconfigure(0, weight=1)
        
        # NUEVO: Actionable Empty State (Reemplaza al canvas de rayas)
        self._trigger_placeholder_frame = tk.Frame(
            self._trigger_content_holder,
            bg="#fdfdfd",
            padx=20,
            pady=30,
            highlightbackground="#e0e0e0",
            highlightthickness=1
        )
        self._trigger_placeholder_frame.grid(row=0, column=0, sticky="nsew", pady=10)
        self._trigger_placeholder_frame.columnconfigure(0, weight=1)

        tk.Label(
            self._trigger_placeholder_frame,
            text="⚠️ Aún no has añadido condiciones",
            font=("Segoe UI", 11, "bold"),
            bg="#fdfdfd",
            fg="#d32f2f"
        ).pack(pady=(0, 4))
        
        tk.Label(
            self._trigger_placeholder_frame,
            text="Empieza creando un trigger para que esta regla pueda ejecutarse:",
            font=("Segoe UI", 9),
            bg="#fdfdfd",
            fg="#666666"
        ).pack(pady=(0, 15))

        empty_btn_frame = tk.Frame(self._trigger_placeholder_frame, bg="#fdfdfd")
        empty_btn_frame.pack()
        
        tk.Button(
            empty_btn_frame,
            text="🔭 Añadir visión",
            command=lambda: self._on_condition_add(CONDITION_KIND_VISION),
            bg="#BBDEFB",
            padx=15,
            pady=5,
            relief="raised"
        ).pack(side="left", padx=10)

        tk.Button(
            empty_btn_frame,
            text="🔌 Añadir PLC",
            command=lambda: self._on_condition_add(CONDITION_KIND_PLC_BIT),
            bg="#BBDEFB",
            padx=15,
            pady=5,
            relief="raised"
        ).pack(side="left", padx=10)

        tk.Button(
            empty_btn_frame,
            text="📋 Añadir Regla",
            command=lambda: self._on_condition_add(CONDITION_KIND_RULE),
            bg="#E1BEE7",  # Color lila para diferenciarlo
            padx=15,
            pady=5,
            relief="raised"
        ).pack(side="left", padx=10)

        self._trigger_content_holder.rowconfigure(0, weight=1)

        self._build_condition_frames(self._trigger_content_holder)
        self._toggle_condition_frames()
        self._show_trigger_placeholder(True)

        row += 1

        reaction_wrapper = tk.Frame(
            frame,
            background=self._reaction_section_bg,
            highlightbackground=self._reaction_section_border,
            highlightcolor=self._reaction_section_border,
            highlightthickness=2,
            bd=0,
        )
        reaction_wrapper.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        reaction_wrapper.columnconfigure(0, weight=1)
        reaction_wrapper.rowconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

        reaction_header = tk.Label(
            reaction_wrapper,
            text="Reacciones automáticas a ejecutar",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
            background=self._reaction_section_bg,
            foreground="#6a2c00",
        )
        reaction_header.grid(row=0, column=0, sticky="we", padx=12, pady=(10, 0))

        reactions_frame = tk.Frame(reaction_wrapper, background=self._reaction_section_bg)
        reactions_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))
        reactions_frame.columnconfigure(1, weight=1)
        reactions_frame.rowconfigure(0, weight=1)

        self.var_block = tk.BooleanVar(value=False)
        self._check_with_info(
            reactions_frame,
            0,
            0,
            "Deshabilitar trigger(s):",
            self.var_block,
            "sendToPLC.actions.mute_triggers",
            sticky="w",
            command=self._on_mute_toggle,
        )
        block_frame = ttk.Frame(reactions_frame)
        block_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        block_frame.columnconfigure(0, weight=1)
        block_frame.rowconfigure(0, weight=1)
        self.list_block_triggers = tk.Listbox(
            block_frame,
            selectmode="extended",
            exportselection=False,
            height=4,
        )
        self.list_block_triggers.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(block_frame, orient="vertical", command=self.list_block_triggers.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.list_block_triggers.configure(yscrollcommand=scroll.set)
        self.list_block_triggers.bind("<<ListboxSelect>>", self._on_mute_selection_change)
        controls = ttk.Frame(block_frame)
        controls.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))
        controls.columnconfigure(0, weight=1)
        ttk.Button(controls, text="🔄 Refrescar", command=self._refresh_mute_triggers).grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="🧹 Limpiar selección", command=self._clear_mute_selection).grid(
            row=0, column=1, sticky="e", padx=(6, 0)
        )

        self._label_with_info(
            reactions_frame,
            1,
            0,
            "Duración bloqueo (s):",
            "sendToPLC.actions.mute_duration",
            sticky="w",
            pady=(6, 0),
        )
        self.var_block_duration = tk.IntVar(value=0)
        ttk.Spinbox(reactions_frame, from_=0, to=3600, textvariable=self.var_block_duration, width=8).grid(
            row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Separator(reactions_frame, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="we", pady=(10, 6)
        )

        self._label_with_info(
            reactions_frame,
            3,
            0,
            "Acciones extra (tras condición):",
            "sendToPLC.actions.extra_actions",
            sticky="w",
            columnspan=2,
        )
        buttons_frame = ttk.Frame(reactions_frame)
        buttons_frame.grid(row=4, column=0, columnspan=2, sticky="we", pady=(6, 0))
        for col in range(3):
            buttons_frame.columnconfigure(col, weight=1)

        self.btn_add_snapshot = ttk.Button(buttons_frame, text="Añadir captura", command=self._on_rule_add_snapshot)
        self.btn_add_snapshot.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")

        self.btn_add_message = ttk.Button(buttons_frame, text="Añadir mensaje", command=self._on_rule_add_message)
        self.btn_add_message.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")

        self.btn_add_plc = ttk.Button(buttons_frame, text="Añadir acción PLC", command=self._on_rule_add_plc)
        self.btn_add_plc.grid(row=0, column=2, padx=4, pady=4, sticky="nsew")

        row += 1
        button_bar = ttk.Frame(frame)
        button_bar.grid(row=row, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(button_bar, text="❌ Cancelar", command=self._on_cancel).grid(row=0, column=0, padx=4)
        ttk.Button(button_bar, text="💾 Guardar", command=self._on_accept).grid(row=0, column=1, padx=4)

        self._refresh_mute_triggers()

    def _get_enabled_triggers(self) -> list[tuple[str, str]]:
        if self._service is not None:
            try:
                rules = self._service.get_rules_payload()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("No se pudieron cargar triggers habilitados: %s", exc)
                rules = []
            seen: set[str] = set()
            result: list[tuple[str, str]] = []
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                if not bool(rule.get("enabled", True)):
                    continue
                rule_id = str(rule.get("rule_id", ""))
                name = str(rule.get("name", "")) or "Regla"
                label = f"{name} [{rule_id[:6]}]" if rule_id else name
                if rule_id and rule_id in seen:
                    continue
                seen.add(rule_id)
                result.append((rule_id or name, label))
            return result
        return [(str(item), str(item)) for item in self.available_classes]

    def _refresh_mute_triggers(self) -> None:
        self._mute_trigger_choices = self._get_enabled_triggers()
        self.list_block_triggers.delete(0, tk.END)
        for _, label in self._mute_trigger_choices:
            self.list_block_triggers.insert(tk.END, label)
        self._restore_mute_selection()

    def _restore_mute_selection(self) -> None:
        try:
            self.list_block_triggers.selection_clear(0, tk.END)
        except Exception:
            return
        for idx, (rule_id, _) in enumerate(self._mute_trigger_choices):
            if rule_id in self._selected_mute_triggers:
                self.list_block_triggers.selection_set(idx)

    def _on_mute_toggle(self) -> None:
        if self.var_block.get():
            self._refresh_mute_triggers()

    def _on_mute_selection_change(self, *_: object) -> None:
        selection = self.list_block_triggers.curselection()
        selected: set[str] = set()
        for idx in selection:
            try:
                rule_id = self._mute_trigger_choices[int(idx)][0]
                if rule_id:
                    selected.add(rule_id)
            except Exception:
                continue
        self._selected_mute_triggers = selected

    def _clear_mute_selection(self) -> None:
        self._selected_mute_triggers = set()
        try:
            self.list_block_triggers.selection_clear(0, tk.END)
        except Exception:
            pass

    def _get_selected_mute_triggers(self) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        for rule_id, label in self._mute_trigger_choices:
            if rule_id in self._selected_mute_triggers:
                selected.append({"rule_id": rule_id, "label": label})
        return selected

    def _add_info_icon(self, parent: tk.Widget, row: int, column: int, key: str) -> None:
        if InfoIcon is not None:
            icon = InfoIcon(parent, key)
            icon.grid(row=row, column=column, sticky="e", padx=(0, 2), pady=(6 if row > 0 else 0, 0))
            return
        lbl = ttk.Label(parent, text="i", foreground="#2196F3", cursor="hand2")
        # El padding horizontal ayuda a separarlo del texto del label pero cerca para indicar que es de él
        lbl.grid(row=row, column=column, sticky="e", padx=(0, 2), pady=(6 if row > 0 else 0, 0))
        _Tooltip(lbl, key)

    def _label_with_info(
        self,
        parent: tk.Widget,
        row: int,
        column: int,
        text: str,
        key: str,
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
            InfoIcon(frame, key).pack(side="left", padx=(4, 0))
        else:
            lbl = ttk.Label(frame, text="i", foreground="#2196F3", cursor="hand2")
            lbl.pack(side="left", padx=(4, 0))
            _Tooltip(lbl, key)
        return frame

    def _check_with_info(
        self,
        parent: tk.Widget,
        row: int,
        column: int,
        text: str,
        variable: tk.Variable,
        key: str,
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
            InfoIcon(frame, key).pack(side="left", padx=(4, 0))
        else:
            lbl = ttk.Label(frame, text="i", foreground="#2196F3", cursor="hand2")
            lbl.pack(side="left", padx=(4, 0))
            _Tooltip(lbl, key)
        return frame

    def _build_condition_frames(self, parent: ttk.Frame) -> None:
        row = 6
        self._condition_frames_row = row

        # Condición de visión -------------------------------------------------
        self.frame_condition_vision = ttk.Labelframe(parent, text="Condición de visión")
        self.frame_condition_vision.grid(row=row, column=0, columnspan=2, sticky="nsew")
        self.frame_condition_vision.columnconfigure(1, weight=1)

        self._label_with_info(
            self.frame_condition_vision,
            0,
            0,
            "Clase condición:",
            "sendToPLC.vision.class_condition",
            sticky="w",
        )
        self.combo_class = ttk.Combobox(
            self.frame_condition_vision,
            textvariable=self.var_class,
            values=("", *self.available_classes),
            state="readonly",
        )
        self.combo_class.grid(row=0, column=1, sticky="we", padx=(6, 0))

        # Checkbox para modo simple (solo detectar presencia)
        self._check_with_info(
            self.frame_condition_vision,
            1,
            0,
            "Solo detectar presencia (ignora filtros avanzados)",
            self.var_detection_only,
            "sendToPLC.vision.detection_only",
            sticky="w",
            pady=(8, 4),
            columnspan=2,
            command=self._on_detection_only_toggle,
        )

        # Frame contenedor para los campos avanzados (se oculta si detection_only está activo)
        self.frame_vision_advanced = ttk.Frame(self.frame_condition_vision)
        self.frame_vision_advanced.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.frame_vision_advanced.columnconfigure(1, weight=1)

        self._label_with_info(
            self.frame_vision_advanced,
            0,
            0,
            "Mín. apariciones:",
            "sendToPLC.vision.min_count",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Spinbox(
            self.frame_vision_advanced,
            from_=0,
            to=1000,
            textvariable=self.var_min_count,
            width=6,
        ).grid(row=0, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        self._label_with_info(
            self.frame_vision_advanced,
            1,
            0,
            "Máx. apariciones (vacío = sin límite):",
            "sendToPLC.vision.max_count",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Entry(self.frame_vision_advanced, textvariable=self.var_max_count, width=10).grid(
            row=1,
            column=1,
            sticky="w",
            padx=(6, 0),
            pady=(6, 0),
        )

        self._label_with_info(
            self.frame_vision_advanced,
            2,
            0,
            "Área mínima:",
            "sendToPLC.vision.min_area",
            sticky="w",
            pady=(6, 0),
        )
        frame_area_min = ttk.Frame(self.frame_vision_advanced)
        frame_area_min.grid(row=2, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        frame_area_min.columnconfigure(0, weight=1)
        ttk.Entry(frame_area_min, textvariable=self.var_min_area, width=10).grid(row=0, column=0, sticky="we")

        self._label_with_info(
            self.frame_vision_advanced,
            3,
            0,
            "Área máxima:",
            "sendToPLC.vision.max_area",
            sticky="w",
            pady=(6, 0),
        )
        frame_area_max = ttk.Frame(self.frame_vision_advanced)
        frame_area_max.grid(row=3, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        frame_area_max.columnconfigure(0, weight=1)
        ttk.Entry(frame_area_max, textvariable=self.var_max_area, width=10).grid(row=0, column=0, sticky="we")

        self._label_with_info(
            self.frame_vision_advanced,
            4,
            0,
            "Unidad área:",
            "sendToPLC.vision.area_unit",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Combobox(
            self.frame_vision_advanced,
            textvariable=self.var_area_unit,
            values=("px", "cm"),
            state="readonly",
            width=8,
        ).grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        self._label_with_info(
            self.frame_vision_advanced,
            5,
            0,
            "Conf. mínima:",
            "sendToPLC.vision.min_conf",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Entry(self.frame_vision_advanced, textvariable=self.var_min_conf, width=10).grid(
            row=5,
            column=1,
            sticky="w",
            padx=(6, 0),
            pady=(6, 0),
        )

        self._label_with_info(
            self.frame_vision_advanced,
            6,
            0,
            "Ventana de análisis (s):",
            "sendToPLC.vision.window_sec",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Spinbox(
            self.frame_vision_advanced,
            from_=1,
            to=600,
            textvariable=self.var_window_sec,
            width=8,
        ).grid(row=6, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        # NUEVO: Campo de sector(es) - acepta número único o lista separada por comas
        self._label_with_info(
            self.frame_vision_advanced,
            7,
            0,
            "Sector(es):",
            "sendToPLC.vision.sector",
            sticky="w",
            pady=(6, 0),
        )
        sector_frame = ttk.Frame(self.frame_vision_advanced)
        sector_frame.grid(row=7, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        ttk.Entry(sector_frame, textvariable=self.var_sector, width=15).grid(row=0, column=0, sticky="w")
        ttk.Label(sector_frame, text="(vacío=todos, ej: 3 o 1,2,3)", foreground="gray").grid(
            row=0, column=1, sticky="w", padx=(6, 0)
        )

        # Condición PLC -------------------------------------------------------
        self.frame_condition_plc = ttk.Labelframe(parent, text="Condición PLC")
        self.frame_condition_plc.grid(row=row, column=0, columnspan=2, sticky="nsew")
        self.frame_condition_plc.columnconfigure(1, weight=1)

        row_plc = 0
        self._label_with_info(
            self.frame_condition_plc,
            row_plc,
            0,
            "Preset PLC:",
            "sendToPLC.plc.preset",
            sticky="w",
        )
        self.combo_plc_condition_preset = ttk.Combobox(
            self.frame_condition_plc,
            textvariable=self.var_plc_condition_preset,
            values=self._plc_preset_labels,
            state="readonly" if self._plc_preset_labels else "normal",
        )
        self.combo_plc_condition_preset.grid(row=row_plc, column=1, sticky="we", padx=(6, 0))
        self.combo_plc_condition_preset.bind("<<ComboboxSelected>>", self._on_plc_preset_change)

        row_plc += 1
        self.chk_plc_manual = self._check_with_info(
            self.frame_condition_plc,
            row_plc,
            0,
            "Configurar manualmente (sin preset)",
            self.var_plc_manual_enabled,
            "sendToPLC.plc.manual_toggle",
            sticky="w",
            pady=(6, 0),
            columnspan=2,
            command=self._toggle_plc_manual_section,
        )
        if not self._plc_preset_labels:
            self.chk_plc_manual.state(["disabled"])

        row_plc += 1
        self.frame_plc_manual = ttk.Frame(self.frame_condition_plc)
        self.frame_plc_manual.columnconfigure(1, weight=1)
        self._label_with_info(
            self.frame_plc_manual,
            0,
            0,
            "IP (opcional):",
            "sendToPLC.plc.ip_optional",
            sticky="w",
        )
        ttk.Entry(self.frame_plc_manual, textvariable=self.var_plc_condition_ip).grid(
            row=0, column=1, sticky="we", padx=(6, 0)
        )
        self._label_with_info(
            self.frame_plc_manual,
            1,
            0,
            "Área:",
            "sendToPLC.plc.connection_params",
            sticky="w",
            pady=(6, 0),
        )
        self.combo_plc_condition_area = ttk.Combobox(
            self.frame_plc_manual,
            textvariable=self.var_plc_condition_area,
            values=("M", "DB", "Q", "I"),
            state="readonly",
            width=6,
        )
        self.combo_plc_condition_area.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(self.frame_plc_manual, text="Rack:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            self.frame_plc_manual,
            from_=0,
            to=10,
            textvariable=self.var_plc_condition_rack,
            width=6,
        ).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(self.frame_plc_manual, text="Slot:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            self.frame_plc_manual,
            from_=0,
            to=10,
            textvariable=self.var_plc_condition_slot,
            width=6,
        ).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(self.frame_plc_manual, text="DB (si área DB):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.entry_plc_condition_db = ttk.Entry(self.frame_plc_manual, textvariable=self.var_plc_condition_db)
        self.entry_plc_condition_db.grid(row=4, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        self._plc_manual_row = row_plc
        self.frame_plc_manual.grid(row=row_plc, column=0, columnspan=2, sticky="nsew", pady=(6, 0))

        row_plc += 1
        self._frame_plc_byte_label = ttk.Frame(self.frame_condition_plc)
        self._frame_plc_byte_label.grid(row=row_plc, column=0, sticky="w", pady=(6, 0))
        self.lbl_plc_byte = ttk.Label(self._frame_plc_byte_label, text="Byte:")
        self.lbl_plc_byte.pack(side="left")
        if InfoIcon is not None:
            InfoIcon(self._frame_plc_byte_label, "sendToPLC.plc.byte_bit_value").pack(side="left", padx=(6, 0))
        inline_frame = ttk.Frame(self.frame_condition_plc)
        inline_frame.grid(row=row_plc, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self.spin_plc_byte = ttk.Spinbox(
            inline_frame,
            from_=0,
            to=65535,
            textvariable=self.var_plc_condition_byte,
            width=6,
        )
        self.spin_plc_byte.grid(row=0, column=0, sticky="w")
        self.lbl_plc_bit = ttk.Label(inline_frame, text="Bit (0-7):")
        self.lbl_plc_bit.grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.spin_plc_bit = ttk.Spinbox(
            inline_frame,
            from_=0,
            to=7,
            textvariable=self.var_plc_condition_bit,
            width=4,
        )
        self.spin_plc_bit.grid(row=0, column=2, sticky="w", padx=(4, 0))
        
        self.lbl_plc_expected = ttk.Label(inline_frame, text="Valor:")
        self.lbl_plc_expected.grid(row=0, column=3, sticky="w", padx=(12, 0))

        self.frame_plc_expected = ttk.Frame(inline_frame)
        self.frame_plc_expected.grid(row=0, column=4, sticky="w", padx=(6, 0))
        ttk.Radiobutton(self.frame_plc_expected, text="1", variable=self.var_plc_condition_expected, value="1").grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Radiobutton(self.frame_plc_expected, text="0", variable=self.var_plc_condition_expected, value="0").grid(row=0, column=1)

        row_plc += 1
        self._label_with_info(
            self.frame_condition_plc,
            row_plc,
            0,
            "Etiqueta opcional:",
            "sendToPLC.plc.label_optional",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Entry(self.frame_condition_plc, textvariable=self.var_plc_condition_label).grid(
            row=row_plc, column=1, sticky="we", padx=(6, 0), pady=(6, 0)
        )

        self._toggle_plc_manual_section()
        
        # Selector de Tipo (Bit / Numérico)
        row_plc += 1
        ttk.Separator(self.frame_condition_plc).grid(row=row_plc, column=0, columnspan=2, sticky="we", pady=10)
        row_plc += 1
        self._label_with_info(
            self.frame_condition_plc,
            row_plc,
            0,
            "Modo de lectura:",
            "sendToPLC.plc.read_mode",
            sticky="w",
            pady=(6, 0),
        )
        frame_plc_type = ttk.Frame(self.frame_condition_plc)
        frame_plc_type.grid(row=row_plc, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Radiobutton(frame_plc_type, text="Bit (Digital)", variable=self.var_plc_type, value="bit", command=self._toggle_plc_type_ui).grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(frame_plc_type, text="Numérico (Analógico)", variable=self.var_plc_type, value="numeric", command=self._toggle_plc_type_ui).grid(row=0, column=1)

        # UI para PLC Numérico
        row_plc += 1
        self.frame_plc_numeric = ttk.Frame(self.frame_condition_plc)
        self.frame_plc_numeric.grid(row=row_plc, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self.frame_plc_numeric.columnconfigure(1, weight=1)
        
        # Dirección S7
        self._label_with_info(
            self.frame_plc_numeric,
            0,
            0,
            "Dirección (S7):",
            "sendToPLC.plc.numeric_address",
            sticky="w",
        )
        ttk.Entry(self.frame_plc_numeric, textvariable=self.var_plc_numeric_address).grid(row=0, column=1, sticky="we", padx=(6, 0))
        ttk.Label(self.frame_plc_numeric, text="Ej: DB6.DBW0, MW100, DB6.DBD4", foreground="gray").grid(row=1, column=1, sticky="w", padx=(6, 0))

        # Tipo de Dato
        self._label_with_info(
            self.frame_plc_numeric,
            2,
            0,
            "Tipo de dato:",
            "sendToPLC.plc.numeric_type",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Combobox(self.frame_plc_numeric, textvariable=self.var_plc_numeric_type, values=PLC_NUMERIC_TYPES, state="readonly", width=10).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        # Operador
        self._label_with_info(
            self.frame_plc_numeric,
            3,
            0,
            "Condición:",
            "sendToPLC.plc.numeric_operator",
            sticky="w",
            pady=(6, 0),
        )
        frame_op = ttk.Frame(self.frame_plc_numeric)
        frame_op.grid(row=3, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        
        self.combo_plc_operator = ttk.Combobox(frame_op, textvariable=self.var_plc_operator, values=("=", ">", "<", ">=", "<=", "!=", "between"), state="readonly", width=8)
        self.combo_plc_operator.pack(side="left")
        self.combo_plc_operator.bind("<<ComboboxSelected>>", self._on_plc_operator_change)
        
        self.entry_plc_value1 = ttk.Entry(frame_op, textvariable=self.var_plc_value1, width=10)
        self.entry_plc_value1.pack(side="left", padx=(6, 0))
        
        self.lbl_plc_and = ttk.Label(frame_op, text="y")
        self.entry_plc_value2 = ttk.Entry(frame_op, textvariable=self.var_plc_value2, width=10)
        
        # Inicializar estado UI
        self._toggle_plc_type_ui()


        # Condición Regla (meta-regla) -----------------------------------------
        self.frame_condition_rule = ttk.Labelframe(parent, text="Condición de Regla")
        self.frame_condition_rule.grid(row=row, column=0, columnspan=2, sticky="nsew")
        self.frame_condition_rule.columnconfigure(1, weight=1)

        row_rule = 0
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Regla a monitorear:",
            "sendToPLC.rule_editor.target_rule",
            sticky="w",
        )
        self.combo_rule_target = ttk.Combobox(
            self.frame_condition_rule,
            textvariable=self.var_rule_target,
            state="readonly",
            values=[],
        )
        self.combo_rule_target.grid(row=row_rule, column=1, sticky="we", padx=(6, 0))

        row_rule += 1
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Mín. disparos:",
            "sendToPLC.rule_editor.min_firings",
            sticky="w",
            pady=(6, 0),
        )
        spinbox_min = ttk.Spinbox(
            self.frame_condition_rule,
            from_=1,
            to=1000,
            textvariable=self.var_rule_min_firings,
            width=8,
        )
        spinbox_min.grid(row=row_rule, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        row_rule += 1
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Máx. disparos:",
            "sendToPLC.rule_editor.max_firings",
            sticky="w",
            pady=(6, 0),
        )
        frame_max = ttk.Frame(self.frame_condition_rule)
        frame_max.grid(row=row_rule, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        ttk.Entry(frame_max, textvariable=self.var_rule_max_firings, width=8).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(frame_max, text="(vacío = sin límite)", foreground="gray").grid(
            row=0, column=1, sticky="w", padx=(6, 0)
        )

        row_rule += 1
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Ventana (segundos):",
            "sendToPLC.rule_editor.window_sec",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Spinbox(
            self.frame_condition_rule,
            from_=1,
            to=3600,
            textvariable=self.var_rule_window_sec,
            width=8,
        ).grid(row=row_rule, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        row_rule += 1
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Debounce (ms):",
            "sendToPLC.rule_editor.debounce_ms",
            sticky="w",
            pady=(6, 0),
        )
        frame_debounce = ttk.Frame(self.frame_condition_rule)
        frame_debounce.grid(row=row_rule, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        ttk.Spinbox(
            frame_debounce,
            from_=0,
            to=10000,
            textvariable=self.var_rule_debounce_ms,
            width=8,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame_debounce, text="(0 = sin debounce)", foreground="gray"
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        row_rule += 1
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Cooldown (segundos):",
            "sendToPLC.rule_editor.cooldown_sec",
            sticky="w",
            pady=(6, 0),
        )
        frame_cooldown = ttk.Frame(self.frame_condition_rule)
        frame_cooldown.grid(row=row_rule, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        ttk.Spinbox(
            frame_cooldown,
            from_=0,
            to=3600,
            textvariable=self.var_rule_cooldown_sec,
            width=8,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame_cooldown, text="(0 = sin cooldown)", foreground="gray"
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        row_rule += 1
        self._label_with_info(
            self.frame_condition_rule,
            row_rule,
            0,
            "Etiqueta:",
            "sendToPLC.rule_editor.label",
            sticky="w",
            pady=(6, 0),
        )
        ttk.Entry(self.frame_condition_rule, textvariable=self.var_rule_label).grid(
            row=row_rule, column=1, sticky="we", padx=(6, 0), pady=(6, 0)
        )

        self._condition_frames_last_row = row

    def _show_trigger_placeholder(self, show: bool) -> None:
        self._trigger_placeholder_active = show
        if show:
            self._trigger_placeholder_frame.grid(row=0, column=0, sticky="nsew", pady=10)
            self.frame_condition_vision.grid_remove()
            self.frame_condition_plc.grid_remove()
            self.frame_condition_rule.grid_remove()
            tk.Misc.lift(self._trigger_placeholder_frame)
        else:
            self._trigger_placeholder_frame.grid_remove()
            self._toggle_condition_frames()

    def _ensure_condition_editor_visible(self, kind: str) -> None:
        self._show_trigger_placeholder(False)
        self.var_condition_kind.set(kind)
        self._toggle_condition_frames()

    def _build_condition_builder(self, parent: ttk.Frame, row: int) -> None:
        builder = ttk.Labelframe(parent, text="Condiciones combinadas")
        builder.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        builder.columnconfigure(0, weight=1)

        summary_frame = ttk.Frame(builder)
        summary_frame.grid(row=0, column=0, sticky="we")
        summary_frame.columnconfigure(1, weight=1)
        ttk.Label(summary_frame, text="Resumen actual:").grid(row=0, column=0, sticky="w")
        ttk.Label(summary_frame, textvariable=self.var_condition_summary, wraplength=520).grid(
            row=0, column=1, sticky="we", padx=(6, 0)
        )
        ttk.Label(summary_frame, textvariable=self.var_condition_hint, foreground="#555555", wraplength=520).grid(
            row=1, column=0, columnspan=2, sticky="we", pady=(4, 0)
        )

        root_frame = ttk.Frame(builder)
        root_frame.grid(row=2, column=0, sticky="we", pady=(6, 0))
        ttk.Label(root_frame, text="Unión principal:").grid(row=0, column=0, sticky="w")
        self.combo_root_operator = ttk.Combobox(
            root_frame,
            state="readonly",
            values=[label for _, label in self._operator_options],
            width=28,
        )
        self.combo_root_operator.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self._add_info_icon(root_frame, 0, 2, "sendToPLC.conditions.root_operator")

        self.combo_root_operator.bind("<<ComboboxSelected>>", self._on_root_operator_change)
        self._sync_root_operator_label()
        ttk.Checkbutton(
            root_frame,
            text="Negar resultado",
            variable=self.var_root_negated,
            command=self._update_condition_summary,
        ).grid(row=0, column=3, sticky="w", padx=(12, 0))
        self._add_info_icon(root_frame, 0, 4, "sendToPLC.conditions.negated")

        list_frame = ttk.Frame(builder)
        list_frame.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.condition_listbox = tk.Listbox(list_frame, height=6, exportselection=False)
        self.condition_listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.condition_listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.condition_listbox.configure(yscrollcommand=list_scroll.set)
        self.condition_listbox.bind("<<ListboxSelect>>", self._on_condition_selected)

        buttons_frame = ttk.Frame(builder)
        buttons_frame.grid(row=4, column=0, sticky="we", pady=(6, 0))
        for col in range(7):
            buttons_frame.columnconfigure(col, weight=1)

        # Estilos de botones
        self._btn_style_add = {"bg": "#BBDEFB", "activebackground": "#90CAF9"}
        self._btn_style_delete = {"bg": "#FFCDD2", "activebackground": "#EF9A9A"}
        self._btn_style_save = {"bg": "#E0E0E0", "activebackground": "#BDBDBD"}
        
        self._btn_add_vision = tk.Button(
            buttons_frame,
            text="🔭 Añadir visión",
            command=lambda: self._on_condition_add(CONDITION_KIND_VISION),
            **self._btn_style_add,
            relief="raised",
            bd=1
        )
        self._btn_add_vision.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        self._btn_add_plc = tk.Button(
            buttons_frame,
            text="🔌 Añadir PLC",
            command=lambda: self._on_condition_add(CONDITION_KIND_PLC_BIT),
            **self._btn_style_add,
            relief="raised",
            bd=1
        )
        self._btn_add_plc.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        self._btn_add_rule = tk.Button(
            buttons_frame,
            text="📋 Añadir Regla",
            command=lambda: self._on_condition_add(CONDITION_KIND_RULE),
            bg="#E1BEE7",
            activebackground="#CE93D8",
            relief="raised",
            bd=1
        )
        self._btn_add_rule.grid(row=0, column=2, padx=2, pady=2, sticky="nsew")

        self._btn_edit = tk.Button(
            buttons_frame,
            text="📝 Editar",
            command=self._on_condition_edit_click,
            **self._btn_style_add,
            relief="raised",
            bd=1
        )
        self._btn_edit.grid(row=0, column=3, padx=2, pady=2, sticky="nsew")

        self._btn_save_condition = tk.Button(
            buttons_frame,
            text="💾 Guardar",
            command=lambda: self._on_btn_click(self._btn_save_condition, self._btn_style_save, "#757575", self._on_condition_save),
            **self._btn_style_save,
            relief="raised",
            bd=1
        )
        self._btn_save_condition.grid(row=0, column=4, padx=2, pady=2, sticky="nsew")

        self._btn_cancel_condition = tk.Button(
            buttons_frame,
            text="❌ Cancelar",
            command=lambda: self._on_btn_click(self._btn_cancel_condition, self._btn_style_delete, "#E57373", self._on_condition_cancel),
            bg="#FFCDD2",
            activebackground="#EF9A9A",
            relief="raised",
            bd=1
        )
        self._btn_cancel_condition.grid(row=0, column=5, padx=2, pady=2, sticky="nsew")

        self._btn_delete = tk.Button(
            buttons_frame,
            text="🗑 Eliminar",
            command=lambda: self._on_btn_click(self._btn_delete, self._btn_style_delete, "#D32F2F", self._on_condition_delete),
            **self._btn_style_delete,
            relief="raised",
            bd=1
        )
        self._btn_delete.grid(row=0, column=6, padx=2, pady=2, sticky="nsew")

        self._btn_toggle_negated = tk.Button(
            buttons_frame,
            text="🔄 Alternar NO",
            command=self._on_condition_toggle_negated,
            bg="#F5F5F5",
            relief="raised",
            bd=1
        )
        self._btn_toggle_negated.grid(row=1, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")

        self._btn_up = tk.Button(
            buttons_frame,
            text="🔼 Subir",
            command=self._on_condition_up,
            bg="#F5F5F5",
            relief="raised",
            bd=1
        )
        self._btn_up.grid(row=1, column=4, padx=2, pady=2, sticky="nsew")

        self._btn_down = tk.Button(
            buttons_frame,
            text="🔽 Bajar",
            command=self._on_condition_down,
            bg="#F5F5F5",
            relief="raised",
            bd=1
        )
        self._btn_down.grid(row=1, column=5, padx=2, pady=2, sticky="nsew")

        # NUEVO: Sección de filtro de sectores
        sector_wrapper = ttk.Labelframe(builder, text="Sectores a tener en cuenta")
        sector_wrapper.grid(row=5, column=0, sticky="we", pady=(10, 0))
        sector_wrapper.columnconfigure(1, weight=1)

        # Radio buttons para modo
        mode_frame = ttk.Frame(sector_wrapper)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=(2, 4))
        ttk.Radiobutton(
            mode_frame,
            text="Todos los sectores",
            variable=self.var_sector_filter_mode,
            value="all",
            command=self._on_sector_mode_change,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Solo sectores seleccionados:",
            variable=self.var_sector_filter_mode,
            value="selected",
            command=self._on_sector_mode_change,
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        # Frame para checkboxes de sectores
        self._sector_checkboxes_frame = ttk.Frame(sector_wrapper)
        self._sector_checkboxes_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._build_sector_checkboxes()

        # NUEVO: Radio buttons para modo de evaluación de sectores
        eval_frame = ttk.Frame(sector_wrapper)
        eval_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 4))
        
        lbl_eval = ttk.Label(eval_frame, text="Cómo contar detecciones:")
        lbl_eval.grid(row=0, column=0, sticky="w", padx=(0, 4))
        
        # Icono de ayuda para evaluación
        lbl_help_eval = ttk.Label(eval_frame, text="ℹ️", foreground="#2196F3", cursor="hand2")
        lbl_help_eval.grid(row=0, column=1, sticky="w", padx=(0, 8))
        lbl_help_eval.bind("<Button-1>", lambda _: messagebox.showinfo("Ayuda: Evaluación de sectores", 
            "• Sumar sectores: Suma total de detecciones en todos los sectores seleccionados.\n"
            "• Por sector: Evalúa la condición para cada sector de forma independiente.\n\n"
            "Ejemplo 'Mín. 2':\n"
            "- Sumar: 1 en S1 + 1 en S2 = 2 → CUMPLE.\n"
            "- Por sector: Evalúa si S1 tiene 2 O si S2 tiene 2. Aquí → NO CUMPLE."))

        
        # Radio Sumar sectores
        rb_aggregate = ttk.Radiobutton(
            eval_frame,
            text="Sumar sectores",
            variable=self.var_sector_eval_mode,
            value="aggregate",
        )
        rb_aggregate.grid(row=0, column=1, sticky="w")
        
        # Radio Por sector (OR)
        rb_any = ttk.Radiobutton(
            eval_frame,
            text="Por sector",
            variable=self.var_sector_eval_mode,
            value="any",
        )
        rb_any.grid(row=0, column=2, sticky="w", padx=(12, 0))
        
        # Textos de ayuda/ejemplos
        help_frame = ttk.Frame(sector_wrapper)
        help_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(
            help_frame,
            text='Sumar: S1=2 + S2=1 = 3 detecciones  •  Por sector: S1=2 y S2=1 NO suma, evalúa cada uno',
            font=("Segoe UI", 8),
            foreground="#666666",
        ).grid(row=0, column=0, sticky="w")

        self._condition_builder_last_row = row + 5

    def _build_sector_checkboxes(self) -> None:
        """Construye los checkboxes para seleccionar sectores específicos."""
        # Limpiar checkboxes existentes
        for widget in self._sector_checkboxes_frame.winfo_children():
            widget.destroy()
        self._sector_checkboxes.clear()

        # Crear checkboxes para cada sector disponible
        col = 0
        for sector_id in self._available_sectors:
            var = tk.BooleanVar(value=False)
            self._sector_checkboxes[sector_id] = var
            cb = ttk.Checkbutton(
                self._sector_checkboxes_frame,
                text=f"S{sector_id}",
                variable=var,
                command=self._sync_sector_to_textfield,
            )
            cb.grid(row=0, column=col, padx=(0, 4), sticky="w")
            col += 1

        # Estado inicial: deshabilitado si modo es "all"
        self._update_sector_checkboxes_state()

    def _on_sector_mode_change(self) -> None:
        """Callback cuando cambia el modo de filtro de sectores."""
        self._update_sector_checkboxes_state()
        self._sync_sector_to_textfield()

    def _update_sector_checkboxes_state(self) -> None:
        """Habilita/deshabilita checkboxes según el modo."""
        mode = self.var_sector_filter_mode.get()
        state = "normal" if mode == "selected" else "disabled"
        for widget in self._sector_checkboxes_frame.winfo_children():
            if isinstance(widget, ttk.Checkbutton):
                widget.configure(state=state)

    def _sync_sector_to_textfield(self) -> None:
        """Sincroniza los checkboxes con el campo de texto de sector."""
        mode = self.var_sector_filter_mode.get()
        if mode == "all":
            self.var_sector.set("")
        else:
            selected = [str(s) for s, var in self._sector_checkboxes.items() if var.get()]
            self.var_sector.set(",".join(selected))

    def _sync_sector_from_textfield(self) -> None:
        """Sincroniza el campo de texto de sector a los checkboxes."""
        text = self.var_sector.get().strip()
        if not text:
            self.var_sector_filter_mode.set("all")
            for var in self._sector_checkboxes.values():
                var.set(False)
        else:
            self.var_sector_filter_mode.set("selected")
            try:
                selected_ids = set()
                for part in text.split(","):
                    part = part.strip()
                    if part:
                        selected_ids.add(int(part))
                for sector_id, var in self._sector_checkboxes.items():
                    var.set(sector_id in selected_ids)
            except ValueError:
                pass
        self._update_sector_checkboxes_state()

    def _get_selected_sectors(self) -> list[int] | None:
        """Obtiene la lista de sectores seleccionados o None si todos."""
        mode = self.var_sector_filter_mode.get()
        if mode == "all":
            return None
        return [s for s, var in self._sector_checkboxes.items() if var.get()]

    def _on_btn_click(self, btn: tk.Button, normal_style: dict, intense_color: str, command: callable) -> None:
        # Efecto visual inmediato
        orig_bg = normal_style.get("bg", "#f0f0f0")
        btn.configure(bg=intense_color)
        self.window.update_idletasks()
        self.window.after(200, lambda: btn.configure(bg=orig_bg))
        
        # Ejecutar comando
        command()

    def _start_blinking_save(self) -> None:
        self._stop_blinking_save()
        self._blink_state = False
        self._blink_save_loop()

    def _stop_blinking_save(self) -> None:
        if self._blink_save_job:
            self.window.after_cancel(self._blink_save_job)
            self._blink_save_job = None
        self._btn_save_condition.configure(bg=self._btn_style_save["bg"])

    def _blink_save_loop(self) -> None:
        if not self.window.winfo_exists():
            return
        
        color1 = "#FFE0B2" # Naranja claro
        color2 = "#FB8C00" # Naranja intenso
        
        current_color = color2 if self._blink_state else color1
        self._btn_save_condition.configure(bg=current_color)
        self._blink_state = not self._blink_state
        
        self._blink_save_job = self.window.after(2000, self._blink_save_loop)

    def _draw_placeholder_stripes(self, event=None) -> None:
        canvas = self._trigger_placeholder_canvas
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        canvas.delete("all")
        
        # Fondo
        canvas.create_rectangle(0, 0, w, h, fill="#f9f9f9", outline="")
        
        # Rayas diagonales rojas suaves
        step = 20
        for i in range(-h, w, step):
            canvas.create_line(i, h, i + h, 0, fill="#ffcdd2", width=2)
            
        # Texto
        canvas.create_text(
            w/2, h/2,
            text="Pulsa en Añadir para empezar",
            fill="#b71c1c",
            font=("TkDefaultFont", 11, "bold")
        )

    def _set_button_hold(self, btn: tk.Button) -> None:
        self._release_button_hold()
        self._held_button = btn
        # Color activo "sunken"
        btn.configure(relief="sunken", bg="#90CAF9")

    def _release_button_hold(self) -> None:
        if self._held_button:
            # Restaurar estilo original (asumimos estilo add)
            self._held_button.configure(relief="raised", bg=self._btn_style_add["bg"])
            self._held_button = None

    def _on_condition_cancel(self) -> None:
        self._stop_blinking_save()
        self._release_button_hold()
        self._editing_node_id = None
        self.condition_listbox.selection_clear(0, tk.END)
        self._show_trigger_placeholder(True)

    def _on_condition_edit_click(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showinfo("📝 Editar", "Selecciona una condición de la lista para editar.")
            return
        
        # Asegurar que tenemos el ID correcto
        children = self._get_tree_children()
        if 0 <= idx < len(children):
            self._editing_node_id = children[idx].get("node_id")
            
        self._set_button_hold(self._btn_edit)
        self._start_blinking_save()
        # La selección ya carga los datos, pero nos aseguramos de mostrar los frames
        self._toggle_condition_frames()
        self._show_trigger_placeholder(False)

    def _sync_root_operator_label(self) -> None:
        current = self.var_root_operator.get()
        label = self._operator_value_to_label.get(current)
        if label:
            self.combo_root_operator.set(label)
        else:
            default_label = next(iter(self._operator_value_to_label.values()), "")
            if default_label:
                self.combo_root_operator.set(default_label)
                self.var_root_operator.set(self._operator_label_to_value.get(default_label, "and"))

    def _ensure_condition_tree(self) -> dict[str, object]:
        if not isinstance(self._condition_tree, dict) or self._condition_tree.get("type") != "group":
            self._condition_tree = _make_condition_group(self.var_root_operator.get())
        children = self._condition_tree.get("children")
        if not isinstance(children, list):
            self._condition_tree["children"] = []
        self._condition_tree["operator"] = self.var_root_operator.get() or "and"
        self._condition_tree["negated"] = bool(self.var_root_negated.get())
        return self._condition_tree

    def _get_tree_children(self) -> list[dict[str, object]]:
        tree = self._ensure_condition_tree()
        return tree.get("children", [])  # type: ignore[return-value]

    def _refresh_condition_list(self) -> None:
        if not hasattr(self, "condition_listbox"):
            return
        self.condition_listbox.delete(0, tk.END)
        children = self._get_tree_children()
        for idx, child in enumerate(children):
            label = _condition_tree_to_text(child)
            self.condition_listbox.insert(tk.END, f"{idx + 1}. {label}")
        if not children:
            self.var_condition_hint.set("No hay condiciones. Usa \"🔭 Añadir visión\" o \"🔌 Añadir PLC\" para comenzar.")
            self.condition_listbox.selection_clear(0, tk.END)
        else:
            self.var_condition_hint.set("Selecciona una condición para editarla o usa los botones para gestionarlas.")

    def _update_condition_summary(self) -> None:
        tree = self._ensure_condition_tree()
        summary = _condition_tree_to_text(tree)
        self.var_condition_summary.set(summary)
        self._update_rule_summary()

    def _update_rule_summary(self) -> None:
        """Actualiza el resumen humano visible: 'Esta regla: Si [X] entonces [Y]'"""
        if not hasattr(self, "var_rule_summary"):
            return
        
        # Obtener condiciones
        tree = self._ensure_condition_tree()
        conditions_text = _condition_tree_to_text(tree)
        if conditions_text in ("Sin condiciones", "(vacío)"):
            conditions_text = "(sin condiciones)"
        
        # Obtener acciones desde las diferentes fuentes
        action_parts = []
        
        # Acciones de la cache original
        if hasattr(self, "_orig_actions_cache"):
            for action in self._orig_actions_cache:
                kind = action.get("kind", "")
                if kind == "show_message":
                    action_parts.append("Mensaje")
                elif kind == "take_snapshot":
                    action_parts.append("Captura")
                elif kind == "mute_triggers":
                    action_parts.append("Silenciar")
                elif kind in ("plc_bit", "send_plc"):
                    label = action.get("params", {}).get("label") or "PLC"
                    action_parts.append(f"PLC:{label}")
        
        # Acciones añadidas desde diálogos
        if getattr(self, "_snapshot_dialog_payload", None):
            if "Captura" not in action_parts:
                action_parts.append("Captura")
        if getattr(self, "_message_dialog_payload", None):
            if "Mensaje" not in action_parts:
                action_parts.append("Mensaje")
        if getattr(self, "_plc_action_payload", None):
            if not any(p.startswith("PLC:") for p in action_parts):
                action_parts.append("PLC")
        
        # Mute triggers
        if hasattr(self, "var_block") and self.var_block.get():
            if "Silenciar" not in action_parts:
                action_parts.append("Silenciar")
        
        actions_text = " + ".join(action_parts) if action_parts else "(sin acciones)"
        
        self.var_rule_summary.set(f"Esta regla: Si {conditions_text} → {actions_text}")

    def _on_condition_up(self) -> None:
        idx = self._get_selected_index()
        if idx is None or idx <= 0:
            return
        children = self._get_tree_children()
        if idx < len(children):
            children[idx], children[idx - 1] = children[idx - 1], children[idx]
            self._refresh_condition_list()
            self._select_condition_index(idx - 1)
            self._update_condition_summary()

    def _on_condition_down(self) -> None:
        idx = self._get_selected_index()
        children = self._get_tree_children()
        if idx is None or idx >= len(children) - 1:
            return
        children[idx], children[idx + 1] = children[idx + 1], children[idx]
        self._refresh_condition_list()
        self._select_condition_index(idx + 1)
        self._update_condition_summary()

    def _get_selected_index(self) -> int | None:
        selection = self.condition_listbox.curselection()
        if not selection:
            return None
        return int(selection[0])

    def _find_node_by_id(self, node_id: str | None) -> tuple[int, dict[str, object]] | tuple[None, None]:
        if not node_id:
            return (None, None)
        for idx, child in enumerate(self._get_tree_children()):
            if child.get("node_id") == node_id:
                return idx, child
        return (None, None)

    def _on_root_operator_change(self, *_: object) -> None:
        label = self.combo_root_operator.get()
        value = self._operator_label_to_value.get(label, "and")
        self.var_root_operator.set(value)
        self._ensure_condition_tree()["operator"] = value
        self._update_condition_summary()

    def _on_condition_add(self, kind: str) -> None:
        if kind == CONDITION_KIND_PLC_BIT:
            btn = self._btn_add_plc
        elif kind == CONDITION_KIND_RULE:
            btn = self._btn_add_rule
        else:
            kind = CONDITION_KIND_VISION
            btn = self._btn_add_vision

        self._set_button_hold(btn)
        self._editing_node_id = None
        self.condition_listbox.selection_clear(0, tk.END)  # 🧹 Limpiar selección para evitar confusión
        self._ensure_condition_editor_visible(kind)
        self._reset_condition_fields()
        
        # Si es regla, actualizar opciones del combobox al abrir
        if kind == CONDITION_KIND_RULE:
             self._update_rule_target_options()
        
        self._start_blinking_save()

    def _on_condition_save(self) -> None:
        self._release_button_hold()
        kind = self.var_condition_kind.get()
        if kind == CONDITION_KIND_PLC_BIT:
            condition = self._collect_condition_plc()
        elif kind == CONDITION_KIND_RULE:
            condition = self._collect_condition_rule()
        else:
            condition = self._collect_condition_vision()
        if condition is None:
            return
        node = _make_condition_leaf(condition)
        tree = self._ensure_condition_tree()
        children = tree["children"]  # type: ignore[index]
        target_index: int | None = None
        if self._editing_node_id:
            idx, original = self._find_node_by_id(self._editing_node_id)
            if idx is not None and original is not None:
                node["node_id"] = original.get("node_id")
                node["negated"] = original.get("negated", False)
                children[idx] = node
                target_index = idx
            else:
                children.append(node)
                target_index = len(children) - 1
        else:
            children.append(node)
            target_index = len(children) - 1
        self._show_trigger_placeholder(True)
        self._editing_node_id = None
        self._refresh_condition_list()
        self._update_condition_summary()
        self._select_condition_index(target_index)
        self._stop_blinking_save()

    def _on_condition_delete(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            return
        children = self._get_tree_children()
        if 0 <= idx < len(children):
            del children[idx]
        self._editing_node_id = None
        self._refresh_condition_list()
        self._update_condition_summary()
        self._show_trigger_placeholder(True)
        self._stop_blinking_save()

    def _on_condition_toggle_negated(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            return
        children = self._get_tree_children()
        if 0 <= idx < len(children):
            children[idx]["negated"] = not bool(children[idx].get("negated"))
        self._refresh_condition_list()
        self._select_condition_index(idx)
        self._update_condition_summary()

    def _on_condition_selected(self, *_: object) -> None:
        idx = self._get_selected_index()
        if idx is None:
            return
        children = self._get_tree_children()
        if idx >= len(children):
            return
        node = children[idx]
        self._editing_node_id = node.get("node_id")
        condition = node.get("condition", {}) if isinstance(node.get("condition"), dict) else {}
        kind = str(condition.get("kind") or CONDITION_KIND_VISION).strip().lower()
        if kind not in {CONDITION_KIND_VISION, CONDITION_KIND_PLC_BIT}:
            kind = CONDITION_KIND_VISION
        self.var_condition_kind.set(kind)
        self._toggle_condition_frames()
        if kind == CONDITION_KIND_PLC_BIT:
            self._populate_plc_condition(condition)
        else:
            self._populate_vision_condition(condition)

    def _prepare_condition_tree_payload(self) -> dict[str, object] | None:
        tree = _clone_condition_tree(self._condition_tree)
        if tree and tree.get("children"):
            tree["operator"] = self.var_root_operator.get() or "and"
            tree["negated"] = bool(self.var_root_negated.get())
            return tree
        kind = self.var_condition_kind.get()
        if kind == CONDITION_KIND_PLC_BIT:
            condition = self._collect_condition_plc()
        else:
            condition = self._collect_condition_vision()
        if condition is None:
            return None
        tree = _wrap_condition_as_tree(condition)
        return tree

    def _toggle_condition_frames(self) -> None:
        if getattr(self, "_trigger_placeholder_active", False):
            self.frame_condition_vision.grid_remove()
            self.frame_condition_plc.grid_remove()
            self.frame_condition_rule.grid_remove()
            return
        kind = self.var_condition_kind.get()
        if kind == CONDITION_KIND_PLC_BIT:
            self.frame_condition_vision.grid_remove()
            self.frame_condition_rule.grid_remove()
            self.frame_condition_plc.grid(row=self._condition_frames_row, column=0, columnspan=2, sticky="nsew")
        elif kind == CONDITION_KIND_RULE:
            self.frame_condition_vision.grid_remove()
            self.frame_condition_plc.grid_remove()
            self.frame_condition_rule.grid(row=self._condition_frames_row, column=0, columnspan=2, sticky="nsew")
        else:
            self.frame_condition_plc.grid_remove()
            self.frame_condition_rule.grid_remove()
            self.frame_condition_vision.grid(row=self._condition_frames_row, column=0, columnspan=2, sticky="nsew")
        self._condition_frames_last_row = self._condition_frames_row

    def _reset_condition_fields(self) -> None:
        # Visión
        self.var_class.set("")
        self.combo_class.set("")
        self.var_detection_only.set(False)
        self._on_detection_only_toggle()
        self.var_min_count.set(1)
        self.var_max_count.set("")
        self.var_min_area.set("")
        self.var_max_area.set("")
        self.var_area_unit.set("px")
        self.var_min_conf.set("")
        self.var_window_sec.set(int(WINDOW_SHORT_SEC))
        self.var_sector.set("")
        self.var_sector_filter_mode.set("all")
        self.var_sector_eval_mode.set("aggregate")
        for var in self._sector_checkboxes.values():
            var.set(False)
        self._update_sector_checkboxes_state()

        # PLC
        self.var_plc_condition_preset.set("")
        if self._plc_preset_labels:
            self.combo_plc_condition_preset.set("")
        self.var_plc_condition_ip.set("")
        self.var_plc_condition_area.set("M")
        self.combo_plc_condition_area.set("M")
        self.var_plc_condition_rack.set(0)
        self.var_plc_condition_slot.set(2)
        self.var_plc_condition_db.set("")
        self.var_plc_condition_byte.set(0)
        self.var_plc_condition_bit.set(0)
        self.var_plc_condition_expected.set("1")
        self.var_plc_condition_label.set("")
        manual_default = not bool(self._plc_preset_labels)
        self.var_plc_manual_enabled.set(manual_default)
        self._toggle_plc_manual_section()

        # Regla
        self.var_rule_target.set("")
        # Limpiar combobox si es posible, pero se repoblará al abrir
        self.combo_rule_target.set("") 
        self.var_rule_min_firings.set(1)
        self.var_rule_max_firings.set("")
        self.var_rule_window_sec.set(60)
        self.var_rule_debounce_ms.set(0)
        self.var_rule_cooldown_sec.set(0)
        self.var_rule_label.set("")

    def _populate_from_payload(self, payload: dict[str, object] | None) -> None:
        self.var_name.set("Nueva regla")
        self.var_enabled.set(True)
        self.var_priority.set(0)
        self.txt_description.delete("1.0", tk.END)
        self.var_condition_kind.set(CONDITION_KIND_VISION)
        self.var_condition_kind_label.set(self._condition_kind_labels[CONDITION_KIND_VISION])
        self._reset_condition_fields()
        self._orig_actions_cache = []
        self._snapshot_dialog_payload = None
        self._message_dialog_payload = None
        self._plc_action_payload = None
        self.var_block.set(False)
        self._selected_mute_triggers = set()
        if hasattr(self, "list_block_triggers"):
            self.list_block_triggers.selection_clear(0, tk.END)
        self.var_block_duration.set(0)
        self._condition_tree = _make_condition_group("and")
        self.var_root_operator.set("and")
        self.var_root_negated.set(False)
        self._sync_root_operator_label()
        self._refresh_condition_list()
        self._update_condition_summary()

        if not payload:
            self._toggle_condition_frames()
            return

        self.var_name.set(str(payload.get("name", "Regla")))
        self.var_enabled.set(bool(payload.get("enabled", True)))
        self.var_priority.set(int(payload.get("priority", 0)))
        description = payload.get("description")
        if isinstance(description, str) and description.strip():
            self.txt_description.insert("1.0", description.strip())

        condition = payload.get("condition") if isinstance(payload.get("condition"), dict) else {}
        kind = str(condition.get("kind") or CONDITION_KIND_VISION).strip().lower()
        if kind not in {CONDITION_KIND_VISION, CONDITION_KIND_PLC_BIT, CONDITION_KIND_RULE}:
            kind = CONDITION_KIND_VISION
        self.var_condition_kind.set(kind)
        self.var_condition_kind_label.set(self._condition_kind_labels[kind])

        if kind == CONDITION_KIND_PLC_BIT:
            self._populate_plc_condition(condition)
        elif kind == CONDITION_KIND_RULE:
            self._populate_rule_condition(condition)
        else:
            self._populate_vision_condition(condition)

        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_kind = str(action.get("kind"))
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            if action_kind in {"block_classes", "mute_triggers", "block_triggers"}:
                targets = params.get("triggers")
                if isinstance(targets, (list, tuple, set)):
                    self.var_block.set(True)
                    self._selected_mute_triggers = {
                        str(item.get("rule_id", "")) if isinstance(item, dict) else str(item)
                        for item in targets
                        if (isinstance(item, dict) and item.get("rule_id")) or isinstance(item, (str, int))
                    }
                    self._restore_mute_selection()
                duration = params.get("duration_sec")
                if duration is not None:
                    try:
                        self.var_block_duration.set(int(duration))
                    except (TypeError, ValueError):
                        self.var_block_duration.set(0)
                continue
            if action_kind == "show_message":
                self._message_dialog_payload = copy.deepcopy(params)
                continue
            if action_kind == "take_snapshot":
                self._snapshot_dialog_payload = copy.deepcopy(params)
                continue
            if action_kind == "send_plc":
                self._plc_action_payload = copy.deepcopy(params)
                continue
            self._orig_actions_cache.append(copy.deepcopy(action))
        tree_payload = _normalize_condition_tree(payload.get("condition_tree"))
        normalized_condition = _normalize_rule_condition(condition) if isinstance(condition, dict) else {}
        if tree_payload is None and normalized_condition:
            tree_payload = _wrap_condition_as_tree(normalized_condition)
        if tree_payload is None:
            tree_payload = _make_condition_group(self.var_root_operator.get() or "and")
        elif tree_payload.get("type") != "group":
            primary_condition = _extract_primary_condition(tree_payload) or normalized_condition
            tree_payload = _wrap_condition_as_tree(primary_condition) if primary_condition else _make_condition_group("and")
        operator = tree_payload.get("operator", "and") if tree_payload.get("type") == "group" else "and"
        if operator not in {"and", "or"}:
            operator = "and"
        self._condition_tree = tree_payload
        self.var_root_operator.set(operator)
        self.var_root_negated.set(bool(tree_payload.get("negated")))
        self._sync_root_operator_label()
        self._refresh_condition_list()
        self._update_condition_summary()
        if self._get_tree_children():
            # self._select_condition_index(0)  # Usuario prefiere sin selección inicial
            self._show_trigger_placeholder(True)
        else:
            self._show_trigger_placeholder(True)
        self._toggle_condition_frames()

    def _select_condition_index(self, index: int | None) -> None:
        if not hasattr(self, "condition_listbox"):
            return
        self.condition_listbox.selection_clear(0, tk.END)
        if index is None:
            self._editing_node_id = None
            return
        children = self._get_tree_children()
        if not (0 <= index < len(children)):
            self._editing_node_id = None
            return
        self.condition_listbox.selection_set(index)
        self.condition_listbox.see(index)
        self._on_condition_selected()
        # self._start_blinking_save()  # Eliminado parpadeo al seleccionar

    def _on_accept(self) -> None:
        tree = self._prepare_condition_tree_payload()
        if tree is None:
            return
        condition = _extract_primary_condition(tree)
        if not condition:
            messagebox.showerror("Condiciones", "Debes definir al menos una condición.")
            return

        actions: list[dict[str, object]] = [copy.deepcopy(item) for item in self._orig_actions_cache]
        if self.var_block.get():
            targets = self._get_selected_mute_triggers()
            if not targets:
                messagebox.showerror("Reacciones", "Selecciona al menos un trigger a deshabilitar.")
                return
            params: dict[str, object] = {"triggers": targets}
            duration = max(0, int(self.var_block_duration.get()))
            if duration > 0:
                params["duration_sec"] = duration
            actions.append({"kind": "mute_triggers", "params": params})

        if self._snapshot_dialog_payload:
            actions.append({"kind": "take_snapshot", "params": copy.deepcopy(self._snapshot_dialog_payload)})
        if self._message_dialog_payload:
            actions.append({"kind": "show_message", "params": copy.deepcopy(self._message_dialog_payload)})
        if self._plc_action_payload:
            actions.append({"kind": "send_plc", "params": copy.deepcopy(self._plc_action_payload)})

        rule_id = self._orig_payload.get("rule_id") if self._orig_payload else uuid.uuid4().hex
        self.result = {
            "rule_id": rule_id,
            "name": self.var_name.get().strip() or "Regla",
            "enabled": bool(self.var_enabled.get()),
            "priority": int(self.var_priority.get()),
            "description": self.txt_description.get("1.0", tk.END).strip(),
            "condition": condition,
            "condition_tree": tree,
            "actions": actions,
        }
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()

    def _get_available_rules_for_monitor(self) -> list[tuple[str, str]]:
        if not self._service:
            return []
        current_id = str(self._orig_payload.get("rule_id", "")) if self._orig_payload else ""
        options = []
        try:
            # Ordenar por nombre para facilitar búsqueda
            rules = sorted(
                self._service.profile_manager.get_active_rules(), 
                key=lambda r: str(r.get("name", "")).lower()
            )
            for r in rules:
                rid = str(r.get("rule_id", ""))
                # Evitar auto-referencia
                if rid == current_id:
                    continue
                name = str(r.get("name", "Regla"))
                label = f"{name} ({rid[:6]})"
                options.append((rid, label))
        except Exception:
            pass
        return options

    def _update_rule_target_options(self) -> None:
        choices = self._get_available_rules_for_monitor()
        self._rule_target_choices = choices
        values = [label for _, label in choices]
        self.combo_rule_target.configure(values=values)
        
        # Si hay valor seleccionado, verificar que siga existiendo
        current_label = self.combo_rule_target.get()
        if current_label and current_label not in values:
             pass 

    def _populate_rule_condition(self, condition: dict[str, object]) -> None:
        self._reset_condition_fields()
        self._update_rule_target_options()
        
        rid = str(condition.get("rule_id") or "").strip()
        self.var_rule_target.set(rid)
        
        # Seleccionar label en combobox
        found_label = ""
        for rule_id, label in self._rule_target_choices:
            if rule_id == rid:
                found_label = label
                break
        
        if found_label:
            self.combo_rule_target.set(found_label)
        elif rid:
            self.combo_rule_target.set(rid)
                
        self.var_rule_min_firings.set(int(condition.get("min_firings", 1)))
        
        max_f = condition.get("max_firings")
        if max_f is not None:
            self.var_rule_max_firings.set(str(max_f))
            
        self.var_rule_window_sec.set(int(condition.get("window_sec", 60)))
        self.var_rule_debounce_ms.set(int(condition.get("debounce_ms", 0)))
        self.var_rule_cooldown_sec.set(int(condition.get("cooldown_sec", 0)))
        self.var_rule_label.set(str(condition.get("label", "")))

    def _collect_condition_rule(self) -> dict[str, object] | None:
        label = self.combo_rule_target.get()
        rule_id = self.var_rule_target.get()
        
        # Resolver ID desde label
        if label:
            for rid, l in self._rule_target_choices:
                if l == label:
                    rule_id = rid
                    break
        
        if not rule_id:
            messagebox.showerror("Condición", "Debes seleccionar una regla a monitorear.")
            return None
            
        try:
            min_f = int(self.var_rule_min_firings.get())
            window = int(self.var_rule_window_sec.get())
        except ValueError:
             messagebox.showerror("Condición", "Valores numéricos inválidos.")
             return None
             
        max_f_raw = self.var_rule_max_firings.get().strip()
        max_f = int(max_f_raw) if max_f_raw else None
        
        debounce = int(self.var_rule_debounce_ms.get() or 0)
        cooldown = int(self.var_rule_cooldown_sec.get() or 0)
        
        return {
            "kind": CONDITION_KIND_RULE,
            "rule_id": rule_id,
            "min_firings": min_f,
            "max_firings": max_f,
            "window_sec": window,
            "debounce_ms": debounce,
            "cooldown_sec": cooldown,
            "label": self.var_rule_label.get().strip()
        }

    def _populate_vision_condition(self, condition: dict[str, object]) -> None:
        self._reset_condition_fields()
        classe = str(condition.get("class_name") or condition.get("class") or "").strip()
        if classe:
            self.var_class.set(classe)
            self.combo_class.set(classe if classe in self.combo_class.cget("values") else classe)

        try:
            min_count = int(condition.get("min_count", 1))
        except (TypeError, ValueError):
            min_count = 1
        self.var_min_count.set(max(0, min_count))

        max_count = condition.get("max_count")
        if isinstance(max_count, (int, float)):
            self.var_max_count.set(str(int(max_count)))

        for key, target in (("min_area", self.var_min_area), ("max_area", self.var_max_area)):
            value = condition.get(key)
            if value is not None:
                try:
                    target.set(self._format_float(value))
                except Exception:  # noqa: BLE001
                    target.set(str(value))

        area_unit = str(condition.get("area_unit", "px")).strip().lower()
        if area_unit:
            self.var_area_unit.set(area_unit)
            self.combo_plc_condition_area.set(area_unit.upper()) if False else None

        min_conf = condition.get("min_conf")
        if min_conf is None:
            min_conf = condition.get("min_confidence")
        if min_conf is not None:
            try:
                self.var_min_conf.set(self._format_float(min_conf))
            except Exception:  # noqa: BLE001
                self.var_min_conf.set(str(min_conf))

        window_sec = condition.get("window_sec")
        if window_sec is not None:
            try:
                self.var_window_sec.set(int(window_sec))
            except (TypeError, ValueError):
                self.var_window_sec.set(int(WINDOW_SHORT_SEC))

        # Modo simple: solo detectar presencia
        detection_only = bool(condition.get("detection_only"))
        self.var_detection_only.set(detection_only)
        self._on_detection_only_toggle()

        # NUEVO: Cargar sector(es)
        sector_val = condition.get("sector")
        if sector_val is None:
            self.var_sector.set("")
        elif isinstance(sector_val, list):
            self.var_sector.set(",".join(str(s) for s in sector_val))
        else:
            self.var_sector.set(str(sector_val))
        self._sync_sector_from_textfield()

        # NUEVO: Cargar modo de evaluación de sectores
        sector_mode = str(condition.get("sector_mode", "aggregate")).strip().lower()
        self.var_sector_eval_mode.set(sector_mode if sector_mode in ("aggregate", "any") else "aggregate")

    def _populate_plc_condition(self, condition: dict[str, object]) -> None:
        self._reset_condition_fields()

        plc_mode = str(condition.get("plc_mode", "bit")).strip().lower()
        if plc_mode == "numeric":
            self.var_plc_type.set("numeric")
            self.var_plc_numeric_address.set(str(condition.get("address", "")))
            data_type = str(condition.get("data_type", "WORD")).strip().upper()
            if data_type not in PLC_NUMERIC_TYPES:
                data_type = "WORD"
            self.var_plc_numeric_type.set(data_type)
            self.var_plc_operator.set(str(condition.get("operator", "=")))
            self.var_plc_value1.set(str(condition.get("value1", "")))
            self.var_plc_value2.set(str(condition.get("value2", "")))
            self._toggle_plc_type_ui()
            
            # Label also exists in numeric
            label = str(condition.get("label", "")).strip()
            self.var_plc_condition_label.set(label)
            return

        self.var_plc_type.set("bit")
        self._toggle_plc_type_ui()

        preset_id = str(condition.get("preset_id", "")).strip()
        preset_label = self._plc_preset_label_by_id.get(preset_id, "")
        if preset_label:
            self.var_plc_condition_preset.set(preset_label)
            self.combo_plc_condition_preset.set(preset_label)

        ip = str(condition.get("ip", "")).strip()
        self.var_plc_condition_ip.set(ip)

        area = str(condition.get("area", "M")).strip().upper()
        if area not in {"M", "DB", "Q", "I"}:
            area = "M"
        self.var_plc_condition_area.set(area)
        self.combo_plc_condition_area.set(area)

        try:
            rack = int(condition.get("rack", 0))
        except (TypeError, ValueError):
            rack = 0
        self.var_plc_condition_rack.set(max(0, rack))

        try:
            slot = int(condition.get("slot", 2))
        except (TypeError, ValueError):
            slot = 2
        self.var_plc_condition_slot.set(max(0, slot))

        db_number = condition.get("db_number")
        if db_number is not None:
            self.var_plc_condition_db.set(str(db_number))

        try:
            byte_index = int(condition.get("byte_index", 0))
        except (TypeError, ValueError):
            byte_index = 0
        self.var_plc_condition_byte.set(max(0, byte_index))

        try:
            bit_index = int(condition.get("bit_index", 0))
        except (TypeError, ValueError):
            bit_index = 0
        self.var_plc_condition_bit.set(max(0, min(7, bit_index)))

        val = condition.get("expected_value")
        if val is None:
            val = condition.get("expected", 1)
        if isinstance(val, bool):
            expected = "1" if val else "0"
        else:
            expected = str(val).strip()
            if expected not in {"0", "1"}:
                expected = "1"
        self.var_plc_condition_expected.set(expected)

        label = str(condition.get("label", "")).strip()
        self.var_plc_condition_label.set(label)

        advanced_present = any(
            bool(condition.get(key))
            for key in ("ip", "area", "rack", "slot", "db_number")
        )
        manual_required = not preset_label or advanced_present or not self._plc_preset_labels
        self.var_plc_manual_enabled.set(manual_required)
        self._toggle_plc_manual_section()

    def _collect_condition_vision(self) -> dict[str, object] | None:
        selected_class = self.var_class.get().strip()
        if not selected_class:
            messagebox.showerror("Condición", "Debes seleccionar una clase para el trigger de visión.")
            return None

        # Parsear sector(es) SIEMPRE (antes de detection_only check)
        sector_value = None
        sector_text = self.var_sector.get().strip()
        if sector_text:
            if "," in sector_text:
                try:
                    sector_value = [int(s.strip()) for s in sector_text.split(",") if s.strip()]
                except ValueError:
                    messagebox.showerror("Condición", "Los sectores deben ser números enteros separados por comas.")
                    return None
            else:
                try:
                    sector_value = int(sector_text)
                except ValueError:
                    messagebox.showerror("Condición", "El sector debe ser un número entero.")
                    return None

        # Modo de evaluación de sectores
        sector_mode = (self.var_sector_eval_mode.get() or "aggregate").strip().lower()
        if sector_mode not in ("aggregate", "any"):
            sector_mode = "aggregate"

        # Modo simple: solo detectar presencia
        detection_only = bool(self.var_detection_only.get())
        if detection_only:
            result = {
                "kind": CONDITION_KIND_VISION,
                "class_name": selected_class,
                "class": selected_class,
                "detection_only": True,
                "sector_mode": sector_mode,
            }
            if sector_value is not None:
                result["sector"] = sector_value
            return result

        # Modo avanzado: validar y recopilar todos los parámetros
        min_count = max(0, int(self.var_min_count.get()))
        max_count_text = (self.var_max_count.get() or "").strip()
        max_count: int | None = None
        if max_count_text:
            try:
                max_count = max(0, int(max_count_text))
            except (TypeError, ValueError):
                messagebox.showerror("Condición", "El máximo de apariciones debe ser un número entero.")
                return None
            if max_count and max_count < min_count:
                messagebox.showerror("Condición", "El máximo de apariciones debe ser mayor o igual al mínimo.")
                return None

        min_area = self._parse_float(self.var_min_area.get())
        max_area = self._parse_float(self.var_max_area.get())
        if min_area is not None and min_area < 0:
            messagebox.showerror("Condición", "El área mínima no puede ser negativa.")
            return None
        if max_area is not None and max_area < 0:
            messagebox.showerror("Condición", "El área máxima no puede ser negativa.")
            return None
        if min_area is not None and max_area is not None and max_area < min_area:
            messagebox.showerror("Condición", "El área máxima debe ser mayor o igual al área mínima.")
            return None

        min_conf = self._parse_float(self.var_min_conf.get())
        if min_conf is not None and not 0 <= min_conf <= 1:
            messagebox.showerror("Condición", "La confianza mínima debe estar entre 0 y 1.")
            return None

        window_sec = max(0, int(self.var_window_sec.get() or 0))

        condition: dict[str, object] = {
            "kind": CONDITION_KIND_VISION,
            "class_name": selected_class,
            "min_count": min_count,
        }
        # Compatibilidad con reglas antiguas que persistían la clave "class"
        condition["class"] = selected_class
        if max_count is not None:
            condition["max_count"] = max_count
        if min_area is not None:
            condition["min_area"] = min_area
        if max_area is not None:
            condition["max_area"] = max_area
        area_unit = (self.var_area_unit.get() or "px").strip().lower()
        if area_unit:
            condition["area_unit"] = area_unit
        if min_conf is not None:
            condition["min_conf"] = min_conf
        if window_sec:
            condition["window_sec"] = window_sec

        # Añadir sector y sector_mode (ya parseados arriba)
        if sector_value is not None:
            condition["sector"] = sector_value
        condition["sector_mode"] = sector_mode

        return condition

    def _toggle_plc_type_ui(self) -> None:
        """Muestra/Oculta secciones según Bit o Numérico"""
        mode = self.var_plc_type.get()
        # La seccion de conexion (Preset / Manual) debe mantenerse activa en ambos casos
        self.combo_plc_condition_preset.state(["readonly"] if self._plc_preset_labels else ["!disabled", "!readonly"])
        self.chk_plc_manual.state(["!disabled"])
        
        # Validar si mostramos manual
        if self.var_plc_manual_enabled.get():
            self.frame_plc_manual.grid()
        else:
            self.frame_plc_manual.grid_remove()

        if mode == "numeric":
            # Ocultar campos especificos de BIT (byte, bit, expected)
            # Esto asume que están en un frame separado o mezclados.
            # En _build_condition_frames, parecen estar en self.frame_condition_plc directamente.
            # No hay un sub-frame 'frame_plc_bit'. Deberíamos haber creado uno.
            # Como parche rápido, deshabilitaremos/ocultaremos los widgets de bit si es posible,
            # pero dado que están en el grid principal, es sucio.
            # MEJOR: Mostrar Numérico y Ocultar Bit.
            # Para esto, necesitamos saber qué widgets son de bit.
            # Widgets de bit: lbl_plc_byte, spin_plc_byte, lbl_plc_bit, spin_plc_bit, lbl_plc_expected, radio_expected_1/0
            
            # Sin embargo, en _build_condition_frames NO AGRUPAMOS los de bit. 
            # Deberíamos moverlos a un frame o gestionarlos individualmente.
            # Asumiremos que el usuario creó _build_condition_frames correctamente o lo editaremos.
            # Revisando _build_condition_frames (no lo he editado para agrupar).
            # Voy a ocultar los widgets de bit individualmente aquí.
            if hasattr(self, "_frame_plc_byte_label"):
                self._frame_plc_byte_label.grid_remove()
            else:
                self.lbl_plc_byte.grid_remove()
            self.spin_plc_byte.grid_remove()
            self.lbl_plc_bit.grid_remove()
            self.spin_plc_bit.grid_remove()
            self.lbl_plc_expected.grid_remove()
            self.frame_plc_expected.grid_remove()

            self.frame_plc_numeric.grid()
            self._on_plc_operator_change()
        else:
            # Mostrar Bit
            if hasattr(self, "_frame_plc_byte_label"):
                self._frame_plc_byte_label.grid()
            else:
                self.lbl_plc_byte.grid()
            self.spin_plc_byte.grid()
            self.lbl_plc_bit.grid()
            self.spin_plc_bit.grid()
            self.lbl_plc_expected.grid()
            self.frame_plc_expected.grid()
            
            self.frame_plc_numeric.grid_remove()

    def _on_plc_operator_change(self, *_: object) -> None:
        """Ajusta inputs de valores según operador (between requiere 2 valores)"""
        op = self.var_plc_operator.get()
        if op == "between":
            self.lbl_plc_and.pack(side="left", padx=4)
            self.entry_plc_value2.pack(side="left")
        else:
            self.lbl_plc_and.pack_forget()
            self.entry_plc_value2.pack_forget()

    def _collect_condition_plc(self) -> dict[str, object] | None:
        kind = CONDITION_KIND_PLC_BIT
        mode = self.var_plc_type.get()

        # Datos comunes de conexión
        preset_label = (self.var_plc_condition_preset.get() or "").strip()
        preset_payload = self._plc_preset_by_label.get(preset_label)
        preset_id = str(preset_payload.get("preset_id", "")).strip() if preset_payload else ""
        
        ip = self.var_plc_condition_ip.get().strip()
        area = self.var_plc_condition_area.get().strip()
        rack = self.var_plc_condition_rack.get()
        slot = self.var_plc_condition_slot.get()
        db_number_raw = self.var_plc_condition_db.get().strip()
        db_number = int(db_number_raw) if db_number_raw else None
        
        base_payload = {
            "kind": kind,
            "plc_mode": mode,
            "label": self.var_plc_condition_label.get().strip(),
            "preset_id": preset_id,
        }
        
        # Si no hay preset o estamos en manual, usar datos manuales
        if not preset_id or self.var_plc_manual_enabled.get():
            base_payload.update({
                "ip": ip,
                "area": area,
                "rack": rack,
                "slot": slot,
                "db_number": db_number,
            })
        
        if mode == "numeric":
            address = self.var_plc_numeric_address.get().strip()
            if not address:
                messagebox.showwarning("Condición PLC", "Debes indicar una dirección S7 (ej: DB6.DBW0).")
                return None
            
            # Extender payload base
            base_payload.update({
                "address": address,
                "data_type": self.var_plc_numeric_type.get(),
                "operator": self.var_plc_operator.get(),
                "value1": self.var_plc_value1.get().strip(),
                "value2": self.var_plc_value2.get().strip(),
            })
            return base_payload
        
        # Modo BIT original
        byte_index = self.var_plc_condition_byte.get()
        bit_index = self.var_plc_condition_bit.get()
        expected = self.var_plc_condition_expected.get()
        
        base_payload.update({
            "byte_index": byte_index,
            "bit_index": bit_index,
            "expected_value": expected == "1",
        })
        return base_payload

    def _on_detection_only_toggle(self) -> None:
        """Muestra u oculta los campos avanzados según el estado del checkbox detection_only."""
        if not hasattr(self, "frame_vision_advanced"):
            return
        if bool(self.var_detection_only.get()):
            self.frame_vision_advanced.grid_remove()
        else:
            self.frame_vision_advanced.grid(row=2, column=0, columnspan=2, sticky="nsew")

    def _on_plc_preset_change(self, *_: object) -> None:
        if not self._plc_preset_labels:
            return
        selection = (self.combo_plc_condition_preset.get() or "").strip()
        if selection:
            self.var_plc_manual_enabled.set(False)
        else:
            self.var_plc_manual_enabled.set(True)
        self._toggle_plc_manual_section()

    def _toggle_plc_manual_section(self) -> None:
        if not hasattr(self, "frame_plc_manual") or self._plc_manual_row is None:
            return
        manual_forced = not self._plc_preset_labels
        manual_active = manual_forced or bool(self.var_plc_manual_enabled.get())
        self.var_plc_manual_enabled.set(manual_active)
        if manual_active:
            self.frame_plc_manual.grid(
                row=self._plc_manual_row,
                column=0,
                columnspan=2,
                sticky="nsew",
                pady=(6, 0),
            )
        else:
            self.frame_plc_manual.grid_remove()

    @staticmethod
    def _parse_float(text: str) -> float | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _format_float(value: object) -> str:
        try:
            return "" if value is None else f"{float(value):.2f}"
        except (TypeError, ValueError):
            return ""

    def _on_rule_add_snapshot(self) -> None:
        initial = self._snapshot_dialog_payload or {"label": "", "annotate": False}
        dialog = _SnapshotDialog(self.window, initial)
        result = dialog.show()
        if result is not None:
            self._snapshot_dialog_payload = result
            self._update_rule_summary()

    def _on_rule_add_message(self) -> None:
        initial = self._message_dialog_payload or {
            "text": "",
            "color": "#ffbc00",
            "duration_ms": 4000,
            "opacity": 0.8,
        }
        dialog = _MessageDialog(self.window, initial)
        result = dialog.show()
        if result is not None:
            self._message_dialog_payload = result
            self._update_rule_summary()

    def _on_rule_add_plc(self) -> None:
        initial_payload = {"kind": "send_plc", "params": self._plc_action_payload or {}}
        dialog = _ActionEditorDialog(self.window, initial_payload, presets=self._plc_presets_cache)
        result = dialog.show()
        if result is not None and result.get("kind") == "send_plc":
            params = result.get("params")
            if isinstance(params, dict):
                self._plc_action_payload = copy.deepcopy(params)
                self._update_rule_summary()

    def _fetch_plc_presets(self) -> list[dict[str, object]]:
        if self._service is None:
            return []
        try:
            return self._service.list_plc_presets()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudieron obtener presets PLC: %s", exc)
            return []


class _ActionEditorDialog:
    ACTION_KINDS: tuple[tuple[str, str], ...] = (
        ("show_message", "Mostrar mensaje en visor"),
        ("take_snapshot", "Captura de pantalla"),
        ("mute_triggers", "Deshabilitar trigger(s)"),
        ("block_classes", "Bloquear clases (legado)"),
        ("force_manual_level", "Forzar nivel manual"),
        ("resume_level", "Reanudar nivel manual"),
        ("send_plc", "Enviar señal PLC"),
    )

    def __init__(self, master: tk.Misc, payload: dict[str, object] | None = None, *, presets: list[dict[str, object]] | None = None) -> None:
        self._orig_payload = copy.deepcopy(payload) if payload else None
        self.result: dict[str, object] | None = None
        self._presets = presets or []

        self.window = tk.Toplevel(master)
        self.window.title("Acción automática")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_ui()
        self._populate_from_payload(self._orig_payload)

    def show(self) -> dict[str, object] | None:
        self.window.wait_window()
        return self.result

    def _add_info_icon(
        self,
        parent: tk.Misc,
        row: int,
        column: int,
        key: str,
        *,
        pady: tuple[int, int] | int | None = None,
    ) -> None:
        if InfoIcon is None:
            return
        icon = InfoIcon(parent, key)
        kwargs: dict[str, object] = {"sticky": "e", "padx": (0, 2)}
        if pady is not None:
            kwargs["pady"] = pady
        icon.grid(row=row, column=column, **kwargs)

    def _label_with_info(
        self,
        parent: tk.Misc,
        row: int,
        column: int,
        text: str,
        key: str,
        *,
        sticky: str = "w",
        pady: tuple[int, int] | int | None = None,
        columnspan: int | None = None,
    ) -> ttk.Frame:
        frame = ttk.Frame(parent)
        kwargs: dict[str, object] = {"sticky": sticky}
        if pady is not None:
            kwargs["pady"] = pady
        if columnspan is not None:
            kwargs["columnspan"] = columnspan
        frame.grid(row=row, column=column, **kwargs)
        ttk.Label(frame, text=text).pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frame, key).pack(side="left", padx=(6, 0))
        return frame

    def _check_with_info(
        self,
        parent: tk.Misc,
        row: int,
        column: int,
        text: str,
        variable: tk.BooleanVar,
        key: str,
        *,
        sticky: str = "w",
        pady: tuple[int, int] | int | None = None,
        columnspan: int | None = None,
        command: Callable[[], None] | None = None,
    ) -> ttk.Checkbutton:
        frame = ttk.Frame(parent)
        kwargs: dict[str, object] = {"sticky": sticky}
        if pady is not None:
            kwargs["pady"] = pady
        if columnspan is not None:
            kwargs["columnspan"] = columnspan
        frame.grid(row=row, column=column, **kwargs)
        chk = ttk.Checkbutton(frame, text=text, variable=variable, command=command)
        chk.pack(side="left")
        if InfoIcon is not None:
            InfoIcon(frame, key).pack(side="left", padx=(6, 0))
        return chk

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        self._label_with_info(frame, 0, 0, "Tipo de acción:", "sendToPLC.action_editor.kind", sticky="w")
        self._kind_index = {code: idx for idx, (code, _) in enumerate(self.ACTION_KINDS)}
        self.var_kind = tk.StringVar(value=self.ACTION_KINDS[0][0])
        self.combo_kind = ttk.Combobox(frame, values=[label for _, label in self.ACTION_KINDS], state="readonly")
        self.combo_kind.grid(row=0, column=1, sticky="we", padx=(6, 0))
        self.combo_kind.current(0)
        self.combo_kind.bind("<<ComboboxSelected>>", self._on_kind_change)

        self.params_frame = ttk.Frame(frame)
        self.params_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        self.params_frame.columnconfigure(1, weight=1)

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(button_bar, text="❌ Cancelar", command=self._on_cancel).grid(row=0, column=0, padx=4)
        ttk.Button(button_bar, text="💾 Guardar", command=self._on_accept).grid(row=0, column=1, padx=4)

        self._init_state()
        self._render_fields()

    def _init_state(self) -> None:
        self.var_message_text = tk.StringVar()
        self.var_message_color = tk.StringVar(value="#ffbc00")
        self.var_message_duration = tk.IntVar(value=4000)
        self.var_message_opacity = tk.StringVar(value="0.8")

        self.var_snapshot_label = tk.StringVar()
        self.var_snapshot_annotate = tk.BooleanVar(value=False)

        self.var_block_classes = tk.StringVar()
        self.var_block_duration = tk.IntVar(value=0)

        self.var_manual_level = tk.StringVar(value=MANUAL_LEVELS[0])

        self.var_plc_preset = tk.StringVar()
        self.var_plc_tag = tk.StringVar()
        self.var_plc_adv = tk.BooleanVar(value=False)
        self.var_plc_ip = tk.StringVar()
        self.var_plc_rack = tk.IntVar(value=0)
        self.var_plc_slot = tk.IntVar(value=2)
        self.var_plc_area = tk.StringVar(value="M")
        self.var_plc_db = tk.StringVar()
        self.tree_targets: ttk.Treeview | None = None
        self._plc_targets: list[dict[str, object]] = []

    def _on_kind_change(self, *_: object) -> None:
        idx = self.combo_kind.current()
        if idx < 0 or idx >= len(self.ACTION_KINDS):
            idx = 0
        self.var_kind.set(self.ACTION_KINDS[idx][0])
        self._render_fields()

    def _render_fields(self) -> None:
        for child in self.params_frame.winfo_children():
            child.destroy()

        kind = self.var_kind.get()
        if kind != "send_plc":
            self.tree_targets = None

        if kind == "show_message":
            self._label_with_info(
                self.params_frame,
                0,
                0,
                "Texto:",
                "sendToPLC.action_editor.message_text",
                sticky="w",
            )
            ttk.Entry(self.params_frame, textvariable=self.var_message_text).grid(row=0, column=1, sticky="we", padx=(6, 0))

            self._label_with_info(
                self.params_frame,
                1,
                0,
                "Color (HEX):",
                "sendToPLC.action_editor.message_color",
                sticky="w",
                pady=(6, 0),
            )
            ttk.Entry(self.params_frame, textvariable=self.var_message_color).grid(row=1, column=1, sticky="we", padx=(6, 0), pady=(6, 0))

            ttk.Label(self.params_frame, text="Duración (ms):").grid(row=2, column=0, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 2, 0, "sendToPLC.action_editor.message_duration", pady=(6, 0))
            ttk.Spinbox(
                self.params_frame,
                from_=500,
                to=60000,
                increment=100,
                textvariable=self.var_message_duration,
                width=8,
            ).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

            ttk.Label(self.params_frame, text="Opacidad (0-1):").grid(row=3, column=0, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 3, 0, "sendToPLC.action_editor.message_opacity", pady=(6, 0))
            ttk.Entry(self.params_frame, textvariable=self.var_message_opacity, width=8).grid(
                row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
            )

        elif kind == "take_snapshot":
            ttk.Label(self.params_frame, text="Etiqueta:").grid(row=0, column=0, sticky="w")
            self._add_info_icon(self.params_frame, 0, 0, "sendToPLC.action_editor.snapshot_label")
            ttk.Entry(self.params_frame, textvariable=self.var_snapshot_label).grid(
                row=0, column=1, sticky="we", padx=(6, 0)
            )

            ttk.Checkbutton(
                self.params_frame,
                text="💾 Guardar anotaciones",
                variable=self.var_snapshot_annotate,
            ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 1, 0, "sendToPLC.action_editor.snapshot_annotate", pady=(6, 0))

        elif kind in {"block_classes", "mute_triggers", "block_triggers"}:
            ttk.Label(self.params_frame, text="Triggers (coma):").grid(row=0, column=0, sticky="w")
            self._add_info_icon(self.params_frame, 0, 0, "sendToPLC.action_editor.block_triggers")
            ttk.Entry(self.params_frame, textvariable=self.var_block_classes).grid(
                row=0, column=1, sticky="we", padx=(6, 0)
            )

            ttk.Label(self.params_frame, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 1, 0, "sendToPLC.action_editor.block_duration", pady=(6, 0))
            ttk.Spinbox(self.params_frame, from_=0, to=3600, textvariable=self.var_block_duration, width=8).grid(
                row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
            )

        elif kind in {"force_manual_level", "resume_level"}:
            ttk.Label(self.params_frame, text="Nivel:").grid(row=0, column=0, sticky="w")
            self._add_info_icon(self.params_frame, 0, 0, "sendToPLC.action_editor.manual_level")
            ttk.Combobox(
                self.params_frame,
                textvariable=self.var_manual_level,
                values=MANUAL_LEVELS,
                state="readonly",
                width=12,
            ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        elif kind == "send_plc":
            preset_names = [str(p.get("name", p.get("preset_id", ""))) for p in self._presets]
            
            ttk.Label(self.params_frame, text="Preset PLC:").grid(row=0, column=0, sticky="w")
            self._add_info_icon(self.params_frame, 0, 0, "sendToPLC.action_editor.plc_preset")
            ttk.Combobox(
                self.params_frame,
                textvariable=self.var_plc_preset,
                values=preset_names,
                state="readonly" if preset_names else "normal",
                width=20,
            ).grid(row=0, column=1, sticky="we", padx=(6, 0))

            ttk.Label(self.params_frame, text="Etiqueta:").grid(row=1, column=0, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 1, 0, "sendToPLC.action_editor.plc_tag", pady=(6, 0))
            ttk.Entry(self.params_frame, textvariable=self.var_plc_tag, width=20).grid(
                row=1, column=1, sticky="we", padx=(6, 0), pady=(6, 0)
            )

            ttk.Label(
                self.params_frame,
                text="Bits a enviar:",
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 2, 0, "sendToPLC.action_editor.plc_targets", pady=(6, 0))

            targets_frame = ttk.Frame(self.params_frame)
            targets_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
            targets_frame.columnconfigure(0, weight=1)
            targets_frame.rowconfigure(0, weight=1)

            self.tree_targets = ttk.Treeview(
                targets_frame,
                columns=("byte", "bit", "value"),
                show="headings",
                height=4,
            )
            self.tree_targets.heading("byte", text="Byte")
            self.tree_targets.heading("bit", text="Bit")
            self.tree_targets.heading("value", text="Valor")
            self.tree_targets.column("byte", width=60, anchor="center")
            self.tree_targets.column("bit", width=60, anchor="center")
            self.tree_targets.column("value", width=70, anchor="center")
            self.tree_targets.grid(row=0, column=0, sticky="nsew")

            scrollbar = ttk.Scrollbar(targets_frame, orient="vertical", command=self.tree_targets.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            self.tree_targets.configure(yscrollcommand=scrollbar.set)

            buttons_frame = ttk.Frame(targets_frame)
            buttons_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))
            buttons_frame.columnconfigure((0, 1, 2), weight=1)
            ttk.Button(buttons_frame, text="Añadir", command=self._on_plc_target_add).grid(row=0, column=0, padx=2)
            ttk.Button(buttons_frame, text="📝 Editar", command=self._on_plc_target_edit).grid(row=0, column=1, padx=2)
            ttk.Button(buttons_frame, text="🗑 Eliminar", command=self._on_plc_target_delete).grid(row=0, column=2, padx=2)

            self._refresh_targets_tree()

            ttk.Checkbutton(
                self.params_frame,
                text="Opciones avanzadas",
                variable=self.var_plc_adv,
                command=self._render_fields,
            ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
            self._add_info_icon(self.params_frame, 4, 0, "sendToPLC.action_editor.plc_advanced", pady=(6, 0))

            if bool(self.var_plc_adv.get()):
                ttk.Label(self.params_frame, text="IP:").grid(row=5, column=0, sticky="w", pady=(6, 0))
                self._add_info_icon(self.params_frame, 5, 0, "sendToPLC.action_editor.plc_ip", pady=(6, 0))
                ttk.Entry(self.params_frame, textvariable=self.var_plc_ip).grid(
                    row=5, column=1, sticky="we", padx=(6, 0), pady=(6, 0)
                )
                ttk.Label(self.params_frame, text="Rack:").grid(row=6, column=0, sticky="w", pady=(6, 0))
                self._add_info_icon(self.params_frame, 6, 0, "sendToPLC.action_editor.plc_rack", pady=(6, 0))
                ttk.Spinbox(self.params_frame, from_=0, to=10, textvariable=self.var_plc_rack, width=8).grid(
                    row=6, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
                )
                ttk.Label(self.params_frame, text="Slot:").grid(row=7, column=0, sticky="w", pady=(6, 0))
                self._add_info_icon(self.params_frame, 7, 0, "sendToPLC.action_editor.plc_slot", pady=(6, 0))
                ttk.Spinbox(self.params_frame, from_=0, to=10, textvariable=self.var_plc_slot, width=8).grid(
                    row=7, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
                )
                ttk.Label(self.params_frame, text="Área:").grid(row=8, column=0, sticky="w", pady=(6, 0))
                self._add_info_icon(self.params_frame, 8, 0, "sendToPLC.action_editor.plc_area", pady=(6, 0))
                ttk.Combobox(
                    self.params_frame,
                    textvariable=self.var_plc_area,
                    values=("M", "DB", "Q", "I"),
                    state="readonly",
                    width=6,
                ).grid(row=8, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
                ttk.Label(self.params_frame, text="DB (si área DB):").grid(row=9, column=0, sticky="w", pady=(6, 0))
                self._add_info_icon(self.params_frame, 9, 0, "sendToPLC.action_editor.plc_db", pady=(6, 0))
                ttk.Entry(self.params_frame, textvariable=self.var_plc_db, width=10).grid(
                    row=9, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
                )

    def _populate_from_payload(self, payload: dict[str, object] | None) -> None:
        if not payload:
            return

        kind = str(payload.get("kind", self.ACTION_KINDS[0][0]))
        if kind not in self._kind_index:
            self._add_custom_kind(kind)
        idx = self._kind_index[kind]
        self.combo_kind.current(idx)
        self.var_kind.set(kind)

        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        self._plc_targets = []
        if kind == "show_message":
            self.var_message_text.set(str(params.get("text", "")))
            self.var_message_color.set(str(params.get("color", "#ffbc00")))
            try:
                self.var_message_duration.set(int(params.get("duration_ms", 4000) or 4000))
            except (TypeError, ValueError):
                self.var_message_duration.set(4000)
            opacity = params.get("opacity")
            self.var_message_opacity.set(str(opacity if opacity is not None else "0.8"))
        elif kind == "take_snapshot":
            self.var_snapshot_label.set(str(params.get("label", "")))
            self.var_snapshot_annotate.set(bool(params.get("annotate", False)))
        elif kind in {"block_classes", "mute_triggers", "block_triggers"}:
            classes = params.get("classes", [])
            targets = params.get("triggers", [])
            if isinstance(targets, (list, tuple, set)) and targets:
                labels: list[str] = []
                for item in targets:
                    if isinstance(item, dict):
                        labels.append(str(item.get("rule_id", "")) or str(item.get("label", "")))
                    else:
                        labels.append(str(item))
                self.var_block_classes.set(", ".join(l for l in labels if l))
            elif isinstance(classes, (list, tuple, set)):
                self.var_block_classes.set(", ".join(str(c) for c in classes))
            try:
                self.var_block_duration.set(int(params.get("duration_sec", 0) or 0))
            except (TypeError, ValueError):
                self.var_block_duration.set(0)
        elif kind in {"force_manual_level", "resume_level"}:
            level = str(params.get("level", MANUAL_LEVELS[0]))
            self.var_manual_level.set(level if level in MANUAL_LEVELS else MANUAL_LEVELS[0])
        elif kind == "send_plc":
            preset_id = str(params.get("preset_id", ""))
            # Store original preset_id to restore if preset isn't found during save
            self._original_preset_id = preset_id
            if preset_id:
                for preset in self._presets:
                    if str(preset.get("preset_id")) == preset_id:
                        self.var_plc_preset.set(str(preset.get("name", preset_id)))
                        break
            self.var_plc_tag.set(str(params.get("tag", "")))
            targets_param = params.get("targets")
            if isinstance(targets_param, list):
                for item in targets_param:
                    if not isinstance(item, dict):
                        continue
                    try:
                        byte_int = int(item.get("byte_index"))
                        bit_int = int(item.get("bit_index"))
                    except (TypeError, ValueError):
                        continue
                    if bit_int < 0 or bit_int > 7:
                        continue
                    value_flag = bool(item.get("value", params.get("value", True)))
                    self._plc_targets.append({"byte_index": byte_int, "bit_index": bit_int, "value": value_flag})
            elif isinstance(targets_param, str):
                tokens = [token.strip() for token in targets_param.split(",") if token.strip()]
                for token in tokens:
                    value_flag = params.get("value", True)
                    coord = token
                    if "=" in token:
                        coord, val_part = token.split("=", 1)
                        value_flag = val_part.strip() in {"1", "true", "True", "on", "ON"}
                    if "." not in coord:
                        continue
                    byte_part, bit_part = coord.split(".", 1)
                    try:
                        byte_int = int(byte_part)
                        bit_int = int(bit_part)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= bit_int <= 7:
                        self._plc_targets.append({"byte_index": byte_int, "bit_index": bit_int, "value": bool(value_flag)})
            ip = params.get("ip")
            rack = params.get("rack")
            slot = params.get("slot")
            area = params.get("area")
            dbn = params.get("db_number")
            advanced_present = any(v not in (None, "") for v in (ip, rack, slot, area, dbn))
            self.var_plc_adv.set(bool(advanced_present))
            if ip is not None:
                self.var_plc_ip.set(str(ip))
            if rack is not None:
                try:
                    self.var_plc_rack.set(int(rack))
                except Exception:
                    self.var_plc_rack.set(0)
            if slot is not None:
                try:
                    self.var_plc_slot.set(int(slot))
                except Exception:
                    self.var_plc_slot.set(2)
            if area:
                self.var_plc_area.set(str(area).upper())
            if dbn not in (None, ""):
                self.var_plc_db.set(str(dbn))

        self._refresh_targets_tree()
        self._render_fields()

    def _add_custom_kind(self, code: str) -> None:
        label = AUTOMATION_ACTION_KINDS.get(code, code)
        self.ACTION_KINDS = (*self.ACTION_KINDS, (code, label))
        self._kind_index = {item_code: idx for idx, (item_code, _) in enumerate(self.ACTION_KINDS)}
        self.combo_kind.configure(values=[label for _, label in self.ACTION_KINDS])

    def _on_accept(self) -> None:
        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror("Acciones", str(exc))
            return

        self.result = {"kind": self.var_kind.get(), "params": params}
        self.window.destroy()

    def _refresh_targets_tree(self) -> None:
        if not self.tree_targets:
            return
        self.tree_targets.delete(*self.tree_targets.get_children())
        for idx, target in enumerate(self._plc_targets):
            byte_idx = target.get("byte_index", 0)
            bit_idx = target.get("bit_index", 0)
            value_flag = "1" if bool(target.get("value", True)) else "0"
            self.tree_targets.insert("", "end", iid=str(idx), values=(byte_idx, bit_idx, value_flag))

    def _on_plc_target_add(self) -> None:
        dialog = _PLCTargetDialog(self.window)
        result = dialog.show()
        if result is None:
            return
        self._plc_targets.append(result)
        self._refresh_targets_tree()

    def _on_plc_target_edit(self) -> None:
        if not self.tree_targets:
            return
        selection = self.tree_targets.selection()
        if not selection:
            messagebox.showinfo("Bits adicionales", "Selecciona un objetivo para editar.")
            return
        idx = int(selection[0])
        if idx < 0 or idx >= len(self._plc_targets):
            return
        current = self._plc_targets[idx]
        dialog = _PLCTargetDialog(self.window, current)
        result = dialog.show()
        if result is None:
            return
        self._plc_targets[idx] = result
        self._refresh_targets_tree()

    def _on_plc_target_delete(self) -> None:
        if not self.tree_targets:
            return
        selection = self.tree_targets.selection()
        if not selection:
            messagebox.showinfo("Bits adicionales", "Selecciona un objetivo para eliminar.")
            return
        idx = int(selection[0])
        if idx < 0 or idx >= len(self._plc_targets):
            return
        self._plc_targets.pop(idx)
        self._refresh_targets_tree()

    def _collect_params(self) -> dict[str, object]:
        kind = self.var_kind.get()
        if kind == "show_message":
            text = self.var_message_text.get().strip()
            if not text:
                raise ValueError("El mensaje no puede estar vacío.")
            color = self.var_message_color.get().strip()
            if not color:
                color = "#ffbc00"
            try:
                duration = max(500, int(self.var_message_duration.get()))
            except (TypeError, ValueError):
                raise ValueError("Duración inválida para el mensaje.") from None
            opacity_raw = self.var_message_opacity.get().strip()
            opacity: float | None = None
            if opacity_raw:
                try:
                    opacity_val = float(opacity_raw)
                    if not 0.0 <= opacity_val <= 1.0:
                        raise ValueError
                    opacity = opacity_val
                except ValueError as exc:
                    raise ValueError("La opacidad debe estar entre 0 y 1.") from exc
            payload: dict[str, object] = {
                "text": text,
                "color": color,
                "duration_ms": duration,
            }
            if opacity is not None:
                payload["opacity"] = opacity
            return payload

        if kind == "take_snapshot":
            return {
                "label": self.var_snapshot_label.get().strip(),
                "annotate": bool(self.var_snapshot_annotate.get()),
            }

        if kind in {"block_classes", "mute_triggers", "block_triggers"}:
            classes_raw = self.var_block_classes.get().strip()
            classes = [item.strip() for item in classes_raw.split(",") if item.strip()]
            try:
                duration = max(0, int(self.var_block_duration.get()))
            except (TypeError, ValueError):
                raise ValueError("Duraci??n inv??lida para deshabilitar triggers.") from None
            payload: dict[str, object] = {"triggers": [{"rule_id": item, "label": item} for item in classes], "classes": classes}
            if duration > 0:
                payload["duration_sec"] = duration
            return payload

        if kind in {"force_manual_level", "resume_level"}:
            level = self.var_manual_level.get()
            if level not in MANUAL_LEVELS:
                raise ValueError("Selecciona un nivel manual válido.")
            return {"level": level}

        if kind == "send_plc":
            preset_name = self.var_plc_preset.get().strip()
            preset_id = ""
            for preset in self._presets:
                if str(preset.get("name")) == preset_name:
                    preset_id = str(preset.get("preset_id", ""))
                    break
            # If no preset found by name but we have an original preset_id, preserve it
            if not preset_id and hasattr(self, "_original_preset_id") and self._original_preset_id:
                preset_id = self._original_preset_id
            if not self._plc_targets:
                raise ValueError("Debes añadir al menos un bit en la tabla de 'Bits a enviar'.")
            primary = self._plc_targets[0]
            byte_index = int(primary.get("byte_index", 0))
            bit_index = int(primary.get("bit_index", 0))
            if bit_index < 0 or bit_index > 7:
                raise ValueError("El primer bit debe estar entre 0 y 7.")
            primary_value = bool(primary.get("value", True))
            payload: dict[str, object] = {
                "preset_id": preset_id,
                "byte_index": byte_index,
                "bit_index": bit_index,
                "value": primary_value,
                "tag": self.var_plc_tag.get().strip(),
            }
            use_adv = bool(self.var_plc_adv.get()) or (not self._presets)
            if use_adv:
                ip = self.var_plc_ip.get().strip()
                if not preset_id and not ip:
                    raise ValueError("Si no hay preset seleccionado, debes indicar una IP.")
                if ip:
                    payload["ip"] = ip
                payload["rack"] = int(self.var_plc_rack.get() or 0)
                payload["slot"] = int(self.var_plc_slot.get() or 2)
                area = (self.var_plc_area.get() or "M").upper()
                if area not in {"M", "DB", "Q", "I"}:
                    raise ValueError("Área inválida. Usa M, DB, Q o I.")
                payload["area"] = area
                db_text = (self.var_plc_db.get() or "").strip()
                if db_text:
                    try:
                        payload["db_number"] = int(db_text)
                    except (TypeError, ValueError):
                        raise ValueError("El DB debe ser un entero.") from None
            if self._plc_targets:
                targets_payload: list[dict[str, object]] = []
                for idx, item in enumerate(self._plc_targets):
                    byte_idx = int(item.get("byte_index", 0))
                    bit_idx = int(item.get("bit_index", 0))
                    if bit_idx < 0 or bit_idx > 7:
                        raise ValueError("Cada bit debe estar entre 0 y 7.")
                    value_flag = bool(item.get("value", primary_value))
                    targets_payload.append(
                        {
                            "byte_index": byte_idx,
                            "bit_index": bit_idx,
                            "value": value_flag,
                        }
                    )
                    if idx == 0:
                        payload["byte_index"] = byte_idx
                        payload["bit_index"] = bit_idx
                        payload["value"] = value_flag
                payload["targets"] = targets_payload
            return payload

        return {}

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()


class _PlcPresetsDialog:
    _AREA_OPTIONS = ("M", "DB", "Q", "I")

    def __init__(self, master: tk.Misc, service: "SendToPLCService") -> None:
        self.service = service
        self.window = tk.Toplevel(master)
        self.window.title("Creador de perfiles PLC")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(True, True)
        self.window.minsize(720, 420)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._presets_cache: dict[str, dict[str, object]] = {}
        self._current_id: str | None = None

        # Form variables
        self.var_preset_id = tk.StringVar()
        self.var_name = tk.StringVar()
        self.var_description = tk.StringVar()
        self.var_ip = tk.StringVar()
        self.var_rack = tk.IntVar(value=0)
        self.var_slot = tk.IntVar(value=2)
        self.var_area = tk.StringVar(value="M")
        self.var_db_number = tk.StringVar()
        self.var_default_byte = tk.IntVar(value=0)
        self.var_default_bit = tk.IntVar(value=0)

        self._build_ui()
        self._refresh_presets()

    def show(self) -> None:
        self.window.wait_window()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = ttk.Frame(self.window, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        container.columnconfigure(0, weight=2)
        container.columnconfigure(1, weight=3)
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=0)

        # Preset list ----------------------------------------------------
        list_frame = ttk.Labelframe(container, text="Presets")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.tree_presets = ttk.Treeview(
            list_frame,
            columns=("name", "ip"),
            show="headings",
            selectmode="browse",
            height=8,
        )
        self.tree_presets.heading("name", text="Nombre")
        self.tree_presets.heading("ip", text="IP")
        self.tree_presets.column("name", width=160, anchor="w")
        self.tree_presets.column("ip", width=120, anchor="center")
        self.tree_presets.grid(row=0, column=0, sticky="nsew")
        self.tree_presets.bind("<<TreeviewSelect>>", lambda *_: self._on_select())

        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree_presets.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree_presets.configure(yscrollcommand=scroll.set)

        buttons_frame = ttk.Frame(list_frame)
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(8, 0))
        buttons_frame.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(buttons_frame, text="Nuevo", command=self._on_new).grid(row=0, column=0, padx=4)
        ttk.Button(buttons_frame, text="Duplicar", command=self._on_duplicate).grid(row=0, column=1, padx=4)
        ttk.Button(buttons_frame, text="🗑 Eliminar", command=self._on_delete).grid(row=0, column=2, padx=4)

        # Form -----------------------------------------------------------
        form = ttk.Labelframe(container, text="Detalle")
        form.grid(row=0, column=1, sticky="nsew")
        for idx in range(6):
            form.rowconfigure(idx, weight=0)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Nombre PLC:").grid(row=0, column=0, sticky="w", pady=(0, 4))
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.name").grid(row=0, column=0, sticky="e", padx=(0, 2), pady=(0, 4))
        ttk.Entry(form, textvariable=self.var_name).grid(row=0, column=1, sticky="we", pady=(0, 4))

        ttk.Label(form, text="Descripción:").grid(row=1, column=0, sticky="w")
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.description").grid(row=1, column=0, sticky="e", padx=(0, 2))
        ttk.Entry(form, textvariable=self.var_description).grid(row=1, column=1, sticky="we")

        ttk.Label(form, text="IP:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.ip").grid(row=2, column=0, sticky="e", padx=(0, 2), pady=(4, 0))
        ttk.Entry(form, textvariable=self.var_ip).grid(row=2, column=1, sticky="we", pady=(4, 0))

        rack_slot = ttk.Frame(form)
        rack_slot.grid(row=3, column=1, sticky="we", pady=(4, 0))
        rack_slot.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Label(form, text="Rack / Slot:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.rack_slot").grid(row=3, column=0, sticky="e", padx=(0, 2), pady=(4, 0))
        ttk.Spinbox(rack_slot, from_=0, to=10, width=6, textvariable=self.var_rack).grid(row=0, column=0, sticky="w")
        ttk.Label(rack_slot, text="/").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Spinbox(rack_slot, from_=0, to=10, width=6, textvariable=self.var_slot).grid(row=0, column=2, sticky="w")

        ttk.Label(form, text="Área:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.area").grid(row=4, column=0, sticky="e", padx=(0, 2), pady=(4, 0))
        area_combo = ttk.Combobox(form, values=self._AREA_OPTIONS, state="readonly", textvariable=self.var_area, width=5)
        area_combo.grid(row=4, column=1, sticky="w", pady=(4, 0))

        ttk.Label(form, text="DB (opcional):").grid(row=5, column=0, sticky="w", pady=(4, 0))
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.db_number").grid(row=5, column=0, sticky="e", padx=(0, 2), pady=(4, 0))
        ttk.Entry(form, textvariable=self.var_db_number, width=10).grid(row=5, column=1, sticky="w", pady=(4, 0))

        defaults_frame = ttk.Frame(form)
        defaults_frame.grid(row=6, column=1, sticky="we", pady=(4, 0))
        defaults_frame.columnconfigure((0, 2), weight=1)
        ttk.Label(form, text="Byte / Bit por defecto:").grid(row=6, column=0, sticky="w", pady=(4, 0))
        if InfoIcon is not None:
            InfoIcon(form, "sendToPLC.plc_presets.default_byte_bit").grid(row=6, column=0, sticky="e", padx=(0, 2), pady=(4, 0))
        ttk.Spinbox(defaults_frame, from_=0, to=65535, width=8, textvariable=self.var_default_byte).grid(row=0, column=0, sticky="w")
        ttk.Label(defaults_frame, text="/").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Spinbox(defaults_frame, from_=0, to=7, width=5, textvariable=self.var_default_bit).grid(row=0, column=2, sticky="w")

        # Action buttons -------------------------------------------------
        action_bar = ttk.Frame(container)
        action_bar.grid(row=1, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(action_bar, text="💾 Guardar", command=self._on_save).grid(row=0, column=0, padx=4)
        ttk.Button(action_bar, text="Cerrar", command=self._on_close).grid(row=0, column=1, padx=4)

    # ------------------------------------------------------------------
    def _refresh_presets(self) -> None:
        try:
            presets = self.service.list_plc_presets()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Presets PLC", f"No se pudieron cargar los presets.\nDetalle: {exc}")
            presets = []
        self._presets_cache = {str(item.get("preset_id")): item for item in presets if isinstance(item, dict)}
        for row in self.tree_presets.get_children(""):
            self.tree_presets.delete(row)
        for preset in sorted(self._presets_cache.values(), key=lambda item: str(item.get("name", ""))):
            preset_id = str(preset.get("preset_id"))
            name = str(preset.get("name", preset_id))
            ip = str(preset.get("ip", ""))
            self.tree_presets.insert("", "end", iid=preset_id, values=(name, ip))
        if self._current_id and self._current_id in self._presets_cache:
            self.tree_presets.selection_set(self._current_id)
            self.tree_presets.focus(self._current_id)
        elif self._presets_cache:
            first = next(iter(self._presets_cache.keys()))
            self.tree_presets.selection_set(first)
            self.tree_presets.focus(first)
            self._load_into_form(self._presets_cache[first])
        else:
            self._on_new()

    def _on_select(self) -> None:
        selection = self.tree_presets.selection()
        if not selection:
            return
        preset_id = selection[0]
        payload = self._presets_cache.get(preset_id)
        if payload:
            self._load_into_form(payload)

    def _load_into_form(self, payload: dict[str, object]) -> None:
        self._current_id = str(payload.get("preset_id"))
        self.var_preset_id.set(self._current_id or "")
        self.var_name.set(str(payload.get("name", "")))
        self.var_description.set(str(payload.get("description", "")))
        self.var_ip.set(str(payload.get("ip", "")))
        self.var_rack.set(int(payload.get("rack", 0) or 0))
        self.var_slot.set(int(payload.get("slot", 0) or 0))
        area = str(payload.get("area", "M") or "M").upper()
        if area not in self._AREA_OPTIONS:
            area = "M"
        self.var_area.set(area)
        db_number = payload.get("db_number")
        self.var_db_number.set("" if db_number in {None, ""} else str(db_number))
        self.var_default_byte.set(int(payload.get("default_byte", 0) or 0))
        self.var_default_bit.set(int(payload.get("default_bit", 0) or 0))

    def _on_new(self) -> None:
        self._current_id = None
        self.var_preset_id.set("")
        self.var_name.set("")
        self.var_description.set("")
        self.var_ip.set("")
        self.var_rack.set(0)
        self.var_slot.set(2)
        self.var_area.set("M")
        self.var_db_number.set("")
        self.var_default_byte.set(0)
        self.var_default_bit.set(0)
        self.tree_presets.selection_remove(self.tree_presets.selection())

    def _on_duplicate(self) -> None:
        selection = self.tree_presets.selection()
        if not selection:
            messagebox.showinfo("Presets PLC", "Selecciona un preset para duplicar.")
            return
        preset = self._presets_cache.get(selection[0])
        if not preset:
            return
        clone = dict(preset)
        clone["preset_id"] = ""
        clone["name"] = f"{preset.get('name', 'PLC')} (copia)"
        self._load_into_form(clone)

    def _on_delete(self) -> None:
        selection = self.tree_presets.selection()
        if not selection:
            messagebox.showinfo("Presets PLC", "Selecciona un preset para eliminar.")
            return
        preset_id = selection[0]
        if not messagebox or not messagebox.askyesno(
            "🗑 Eliminar preset",
            "¿🗑 Eliminar el preset seleccionado?",
            parent=self.window,
        ):
            return
        try:
            self.service.delete_plc_preset(preset_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Presets PLC", f"No se pudo eliminar el preset.\nDetalle: {exc}")
            return
        self._current_id = None
        self._refresh_presets()

    def _collect_payload(self) -> dict[str, object]:
        name = self.var_name.get().strip()
        if not name:
            raise ValueError("El campo 'Nombre PLC' es obligatorio.")
        ip = self.var_ip.get().strip()
        if not ip:
            raise ValueError("Debes indicar la IP del PLC.")
        area = self.var_area.get().strip().upper()
        if area not in self._AREA_OPTIONS:
            area = "M"
        try:
            rack = int(self.var_rack.get())
            slot = int(self.var_slot.get())
            default_byte = max(0, int(self.var_default_byte.get()))
            default_bit = max(0, min(7, int(self.var_default_bit.get())))
        except (TypeError, ValueError) as exc:
            raise ValueError("Rack, slot y byte/bit deben ser números válidos.") from exc

        db_text = self.var_db_number.get().strip()
        db_number: int | None
        if not db_text:
            db_number = None
        else:
            try:
                db_number = int(db_text)
            except (TypeError, ValueError) as exc:
                raise ValueError("El DB debe ser un entero o dejarse vacío.") from exc

        payload: dict[str, object] = {
            "preset_id": self.var_preset_id.get().strip() or None,
            "name": name,
            "description": self.var_description.get().strip(),
            "ip": ip,
            "rack": max(0, rack),
            "slot": max(0, slot),
            "area": area,
            "db_number": db_number,
            "default_byte": default_byte,
            "default_bit": default_bit,
        }
        return payload

    def _on_save(self) -> None:
        try:
            payload = self._collect_payload()
        except ValueError as exc:
            messagebox.showerror("Presets PLC", str(exc))
            return
        try:
            saved = self.service.upsert_plc_preset(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Presets PLC", f"No se pudo guardar el preset.\nDetalle: {exc}")
            return
        self._current_id = str(saved.get("preset_id"))
        self.var_preset_id.set(self._current_id)
        self._refresh_presets()
        messagebox.showinfo("Presets PLC", "Preset guardado correctamente.")

    def _on_close(self) -> None:
        self.window.destroy()


class _PLCTargetDialog:
    def __init__(self, master: tk.Misc, payload: dict[str, object] | None = None) -> None:
        self.result: dict[str, object] | None = None
        self.window = tk.Toplevel(master)
        self.window.title("Bit adicional del PLC")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.var_byte = tk.IntVar(value=int(payload.get("byte_index", 0)) if payload else 0)
        self.var_bit = tk.IntVar(value=int(payload.get("bit_index", 0)) if payload else 0)
        self.var_value = tk.BooleanVar(value=bool(payload.get("value", True)) if payload else True)

        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Byte:").grid(row=0, column=0, sticky="w")
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.plc_target.byte").grid(row=0, column=0, sticky="e", padx=(0, 2))
        ttk.Spinbox(frame, from_=0, to=65535, textvariable=self.var_byte, width=8).grid(
            row=0, column=1, sticky="w", padx=(6, 0)
        )

        ttk.Label(frame, text="Bit (0-7):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.plc_target.bit").grid(row=1, column=0, sticky="e", padx=(0, 2), pady=(6, 0))
        ttk.Spinbox(frame, from_=0, to=7, textvariable=self.var_bit, width=8).grid(
            row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Checkbutton(frame, text="Forzar a 1 (desmarcar = 0)", variable=self.var_value).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.plc_target.value").grid(row=2, column=0, sticky="e", padx=(0, 2), pady=(8, 0))

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(button_bar, text="❌ Cancelar", command=self._on_cancel).grid(row=0, column=0, padx=4)
        ttk.Button(button_bar, text="Aceptar", command=self._on_accept).grid(row=0, column=1, padx=4)

    def show(self) -> dict[str, object] | None:
        self.window.wait_window()
        return self.result

    def _on_accept(self) -> None:
        try:
            byte_idx = int(self.var_byte.get())
        except (TypeError, ValueError):
            if messagebox:
                messagebox.showerror("Bit adicional", "El byte debe ser un entero.")
            return
        try:
            bit_idx = int(self.var_bit.get())
        except (TypeError, ValueError):
            if messagebox:
                messagebox.showerror("Bit adicional", "El bit debe ser un entero.")
            return
        if bit_idx < 0 or bit_idx > 7:
            if messagebox:
                messagebox.showerror("Bit adicional", "El bit debe estar entre 0 y 7.")
            return
        self.result = {
            "byte_index": byte_idx,
            "bit_index": bit_idx,
            "value": bool(self.var_value.get()),
        }
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()


class _SnapshotDialog:
    def __init__(self, master: tk.Misc, payload: dict[str, object] | None = None) -> None:
        self.result: dict[str, object] | None = None
        self.window = tk.Toplevel(master)
        self.window.title("Captura de pantalla")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Etiqueta (opcional):").grid(row=0, column=0, sticky="w")
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.snapshot.label").grid(row=0, column=0, sticky="e", padx=(0, 2))
        self.var_label = tk.StringVar(value=str(payload.get("label", "")) if payload else "")
        ttk.Entry(frame, textvariable=self.var_label).grid(row=0, column=1, sticky="we", padx=(6, 0))

        self.var_annotate = tk.BooleanVar(value=bool(payload.get("annotate", False)) if payload else False)
        ttk.Checkbutton(frame, text="Incluir anotaciones", variable=self.var_annotate).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.snapshot.annotate").grid(row=1, column=0, sticky="e", padx=(0, 2), pady=(8, 0))

        self.var_enabled = tk.BooleanVar(value=_ensure_bool(payload.get("enabled", True), default=True) if payload else True)
        ttk.Checkbutton(frame, text="Habilitar captura", variable=self.var_enabled).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.snapshot.enabled").grid(row=2, column=0, sticky="e", padx=(0, 2), pady=(6, 0))

        self.var_require_trigger = tk.BooleanVar(
            value=_ensure_bool(payload.get("require_trigger", True), default=True) if payload else True
        )
        ttk.Checkbutton(frame, text="Solo cuando la regla se active", variable=self.var_require_trigger).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.snapshot.require_trigger").grid(row=3, column=0, sticky="e", padx=(0, 2), pady=(6, 0))

        ttk.Label(frame, text="Cooldown (s):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.snapshot.cooldown").grid(row=4, column=0, sticky="e", padx=(0, 2), pady=(6, 0))
        try:
            cooldown_value = float(payload.get("cooldown_sec", 0.0)) if payload else 0.0
        except (TypeError, ValueError):
            cooldown_value = 0.0
        self.var_cooldown = tk.DoubleVar(value=max(0.0, cooldown_value))
        ttk.Spinbox(frame, from_=0, to=3600, increment=0.5, textvariable=self.var_cooldown, width=8).grid(
            row=4, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(button_bar, text="❌ Cancelar", command=self._on_cancel).grid(row=0, column=0, padx=4)
        ttk.Button(button_bar, text="Aceptar", command=self._on_accept).grid(row=0, column=1, padx=4)

    def _on_accept(self) -> None:
        cooldown = max(0.0, float(self.var_cooldown.get() or 0.0))
        self.result = {
            "label": self.var_label.get().strip(),
            "annotate": bool(self.var_annotate.get()),
            "enabled": bool(self.var_enabled.get()),
            "require_trigger": bool(self.var_require_trigger.get()),
            "cooldown_sec": cooldown,
        }
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()

    def show(self) -> dict[str, object] | None:
        self.window.wait_window()
        return self.result


class _MessageDialog:
    def __init__(self, master: tk.Misc, payload: dict[str, object]) -> None:
        self.result: dict[str, object] | None = None
        self.window = tk.Toplevel(master)
        self.window.title("Mensaje en visor")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Texto:").grid(row=0, column=0, sticky="w")
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.action_editor.message_text").grid(row=0, column=0, sticky="e", padx=(0, 2))
        self.var_text = tk.StringVar(value=str(payload.get("text", "")))
        ttk.Entry(frame, textvariable=self.var_text).grid(row=0, column=1, sticky="we", padx=(6, 0))

        ttk.Label(frame, text="Color (HEX):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.action_editor.message_color").grid(row=1, column=0, sticky="e", padx=(0, 2), pady=(6, 0))
        self.var_color = tk.StringVar(value=str(payload.get("color", "#ffbc00")))
        ttk.Entry(frame, textvariable=self.var_color).grid(row=1, column=1, sticky="we", padx=(6, 0), pady=(6, 0))

        ttk.Label(frame, text="Duración (ms):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.action_editor.message_duration").grid(row=2, column=0, sticky="e", padx=(0, 2), pady=(6, 0))
        self.var_duration = tk.IntVar(value=int(payload.get("duration_ms", 4000) or 4000))
        ttk.Spinbox(frame, from_=500, to=60000, increment=100, textvariable=self.var_duration, width=8).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(frame, text="Opacidad (0-1):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        if InfoIcon is not None:
            InfoIcon(frame, "sendToPLC.action_editor.message_opacity").grid(row=3, column=0, sticky="e", padx=(0, 2), pady=(6, 0))
        self.var_opacity = tk.StringVar(value=str(payload.get("opacity", 0.8)))
        ttk.Entry(frame, textvariable=self.var_opacity, width=8).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(button_bar, text="❌ Cancelar", command=self._on_cancel).grid(row=0, column=0, padx=4)
        ttk.Button(button_bar, text="Aceptar", command=self._on_accept).grid(row=0, column=1, padx=4)

    def _on_accept(self) -> None:
        text = self.var_text.get().strip()
        if not text:
            messagebox.showerror("Mensaje", "El texto no puede estar vacío.")
            return
        color = self.var_color.get().strip() or "#ffbc00"
        try:
            duration = max(500, int(self.var_duration.get()))
        except (TypeError, ValueError):
            messagebox.showerror("Mensaje", "Duración inválida.")
            return
        opacity_raw = self.var_opacity.get().strip()
        opacity: float | None = None
        if opacity_raw:
            try:
                opacity_val = float(opacity_raw)
                if not 0.0 <= opacity_val <= 1.0:
                    raise ValueError
                opacity = opacity_val
            except ValueError:
                messagebox.showerror("Mensaje", "La opacidad debe estar entre 0 y 1.")
                return
        result: dict[str, object] = {"text": text, "color": color, "duration_ms": duration}
        if opacity is not None:
            result["opacity"] = opacity
        self.result = result
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()

    def show(self) -> dict[str, object] | None:
        self.window.wait_window()
        return self.result

class _ConditionEditorDialog:
    def __init__(
        self,
        master: tk.Misc,
        available_classes: Sequence[str],
        payload: dict[str, object] | None = None,
    ) -> None:
        self.available_classes = tuple(available_classes)
        self._orig_payload = copy.deepcopy(payload) if payload else None
        self.result: dict[str, object] | None = None

        self.window = tk.Toplevel(master)
        self.window.title("Condición avanzada")
        self.window.transient(master)
        self.window.grab_set()
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_ui()
        self._populate_from_payload(self._orig_payload)

    def show(self) -> dict[str, object] | None:
        self.window.wait_window()
        return self.result

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.window, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Clase:").grid(row=0, column=0, sticky="w")
        self.var_class = tk.StringVar()
        self.combo_class = ttk.Combobox(
            frame,
            textvariable=self.var_class,
            values=sorted(self.available_classes),
            state="readonly",
        )
        self.combo_class.grid(row=0, column=1, sticky="we", padx=(6, 0))

        ttk.Label(frame, text="Ventana (s):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.var_window = tk.IntVar(value=int(WINDOW_SHORT_SEC))
        ttk.Spinbox(frame, from_=1, to=600, textvariable=self.var_window, width=8).grid(
            row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(frame, text="Mín. apariciones:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.var_min_count = tk.IntVar(value=1)
        ttk.Spinbox(frame, from_=0, to=1000, textvariable=self.var_min_count, width=6).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(frame, text="Máx. apariciones (vacío = sin límite):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.var_max_count = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.var_max_count, width=8).grid(
            row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(frame, text="Área mínima:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        area_frame = ttk.Frame(frame)
        area_frame.grid(row=4, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        area_frame.columnconfigure(0, weight=1)
        self.var_min_area = tk.StringVar()
        ttk.Entry(area_frame, textvariable=self.var_min_area, width=10).grid(row=0, column=0, sticky="we")

        ttk.Label(frame, text="Área máxima:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        area_frame_max = ttk.Frame(frame)
        area_frame_max.grid(row=5, column=1, sticky="we", padx=(6, 0), pady=(6, 0))
        area_frame_max.columnconfigure(0, weight=1)
        self.var_max_area = tk.StringVar()
        ttk.Entry(area_frame_max, textvariable=self.var_max_area, width=10).grid(row=0, column=0, sticky="we")

        ttk.Label(frame, text="Unidad área:").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.var_area_unit = tk.StringVar(value="px")
        ttk.Combobox(
            frame,
            textvariable=self.var_area_unit,
            values=("px", "cm"),
            state="readonly",
            width=8,
        ).grid(row=6, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Label(frame, text="Conf. mínima:").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.var_min_conf = tk.StringVar()
        ttk.Entry(frame, textvariable=self.var_min_conf, width=10).grid(
            row=7, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        # NUEVO: Campo de sector(es) - acepta número único o lista separada por comas
        ttk.Label(frame, text="Sector(es):").grid(row=8, column=0, sticky="w", pady=(6, 0))
        self.var_sector = tk.StringVar()
        sector_entry = ttk.Entry(frame, textvariable=self.var_sector, width=15)
        sector_entry.grid(row=8, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(frame, text="(vacío=todos, ej: 3 o 1,2,3)", foreground="gray").grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        button_bar = ttk.Frame(frame)
        button_bar.grid(row=10, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(button_bar, text="❌ Cancelar", command=self._on_cancel).grid(row=0, column=0, padx=4)
        ttk.Button(button_bar, text="💾 Guardar", command=self._on_accept).grid(row=0, column=1, padx=4)

    def _populate_from_payload(self, payload: dict[str, object] | None) -> None:
        if not payload:
            return

        class_name = str(payload.get("class_name", ""))
        if class_name and class_name not in self.available_classes:
            self.combo_class.configure(values=(*self.combo_class.cget("values"), class_name))
        self.var_class.set(class_name)
        self.var_window.set(int(payload.get("window_sec", WINDOW_SHORT_SEC) or WINDOW_SHORT_SEC))
        self.var_min_count.set(int(payload.get("min_count", 1) or 0))
        max_count = payload.get("max_count")
        if max_count in {None, ""}:
            self.var_max_count.set("")
        else:
            try:
                self.var_max_count.set(str(int(max_count)))
            except (TypeError, ValueError):
                self.var_max_count.set("")
        self.var_min_area.set(self._format_float(payload.get("min_area")))
        self.var_max_area.set(self._format_float(payload.get("max_area")))
        self.var_area_unit.set(str(payload.get("area_unit", "px")).lower() if payload.get("area_unit") else "px")
        self.var_min_conf.set(self._format_float(payload.get("min_conf")))
        # NUEVO: Cargar sector(es)
        sector_val = payload.get("sector")
        if sector_val is None:
            self.var_sector.set("")
        elif isinstance(sector_val, list):
            self.var_sector.set(",".join(str(s) for s in sector_val))
        else:
            self.var_sector.set(str(sector_val))

    def _on_accept(self) -> None:
        class_name = self.var_class.get().strip()
        if not class_name:
            messagebox.showerror("Condiciones", "Selecciona una clase.")
            return
        try:
            window_sec = max(1, int(self.var_window.get()))
        except (TypeError, ValueError):
            messagebox.showerror("Condiciones", "La ventana debe ser un entero positivo.")
            return
        try:
            min_count = max(0, int(self.var_min_count.get()))
        except (TypeError, ValueError):
            messagebox.showerror("Condiciones", "El número mínimo debe ser un entero válido.")
            return

        max_text = self.var_max_count.get().strip()
        if not max_text:
            max_count = None
        else:
            try:
                max_count = max(0, int(max_text))
            except (TypeError, ValueError):
                messagebox.showerror("Condiciones", "El número máximo debe ser un entero no negativo o quedar vacío.")
                return

        # NUEVO: Parsear sector(es)
        sector_text = self.var_sector.get().strip()
        if not sector_text:
            sector_value = None
        elif "," in sector_text:
            # Lista de sectores
            try:
                sector_value = [int(s.strip()) for s in sector_text.split(",") if s.strip()]
            except ValueError:
                messagebox.showerror("Condiciones", "Los sectores deben ser números enteros separados por comas.")
                return
        else:
            # Sector único
            try:
                sector_value = int(sector_text)
            except ValueError:
                messagebox.showerror("Condiciones", "El sector debe ser un número entero.")
                return

        self.result = {
            "class_name": class_name,
            "window_sec": window_sec,
            "min_count": min_count,
            "max_count": max_count,
            "min_area": self._parse_float(self.var_min_area.get()),
            "max_area": self._parse_float(self.var_max_area.get()),
            "area_unit": self.var_area_unit.get(),
            "min_conf": self._parse_float(self.var_min_conf.get()),
            "sector": sector_value,  # NUEVO: incluir sector en el resultado
        }
        self.window.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()

    @staticmethod
    def _parse_float(text: str) -> float | None:
        value = (text or "").strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _format_float(value: object) -> str:
        try:
            return "" if value is None else f"{float(value):.2f}"
        except (TypeError, ValueError):
            return ""


if tk is None:  # Entorno sin Tk
    SendToPLCWindow = None  # type: ignore


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------


def _normalize_rule_condition(condition: object) -> dict[str, object]:
    """Master dispatcher for normalizing rule conditions based on kind."""
    payload = condition if isinstance(condition, dict) else {}
    kind_raw = payload.get("kind") if isinstance(payload, dict) else None
    kind = str(kind_raw or CONDITION_KIND_VISION).strip().lower()
    if kind == CONDITION_KIND_PLC_BIT:
        return _normalize_plc_condition(payload)
    if kind == CONDITION_KIND_RULE:
        return _normalize_meta_rule_condition(payload)
    return _normalize_vision_condition(payload)


def _normalize_vision_condition(payload: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"kind": CONDITION_KIND_VISION}

    class_name = str(payload.get("class_name") or payload.get("class") or "").strip()
    result["class_name"] = class_name

    detection_only = bool(payload.get("detection_only"))
    result["detection_only"] = detection_only

    window_raw = payload.get("window_sec", WINDOW_SHORT_SEC)
    try:
        window_sec = int(window_raw or WINDOW_SHORT_SEC)
    except (TypeError, ValueError):
        window_sec = int(WINDOW_SHORT_SEC)
    result["window_sec"] = max(1, window_sec)

    min_count_raw = payload.get("min_count", 1)
    try:
        min_count = int(min_count_raw or 0)
    except (TypeError, ValueError):
        min_count = 0 if min_count_raw in {0, "0"} else 1
    result["min_count"] = max(0, min_count)

    max_count_raw = payload.get("max_count")
    if max_count_raw in {None, ""}:
        max_count: int | None = None
    else:
        try:
            max_count = int(max_count_raw)
        except (TypeError, ValueError):
            max_count = None
        if isinstance(max_count, int) and max_count < 0:
            max_count = None
    result["max_count"] = max_count

    min_area = _ensure_float(payload.get("min_area"))
    if isinstance(min_area, float) and min_area < 0:
        min_area = None
    result["min_area"] = min_area

    max_area = _ensure_float(payload.get("max_area"))
    if isinstance(max_area, float) and max_area < 0:
        max_area = None
    result["max_area"] = max_area

    area_unit = str(payload.get("area_unit", "px")).strip().lower() or "px"
    result["area_unit"] = area_unit

    min_conf = _ensure_float(payload.get("min_conf") if "min_conf" in payload else payload.get("min_confidence"))
    if isinstance(min_conf, float) and min_conf < 0:
        min_conf = None
    result["min_conf"] = min_conf

    sector_mode = str(payload.get("sector_mode", "aggregate")).strip().lower()
    if sector_mode not in {"aggregate", "any"}:
        sector_mode = "aggregate"
    result["sector_mode"] = sector_mode

    sector_raw = payload.get("sector")
    sector_value: int | list[int] | None = None
    if sector_raw is None or sector_raw == "":
        sector_value = None
    elif isinstance(sector_raw, list):
        parsed: list[int] = []
        for item in sector_raw:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        if parsed:
            sector_value = parsed
    elif isinstance(sector_raw, str):
        text = sector_raw.strip()
        if text:
            if "," in text:
                parsed = []
                for part in text.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        parsed.append(int(part))
                    except (TypeError, ValueError):
                        continue
                if parsed:
                    sector_value = parsed
            else:
                try:
                    sector_value = int(text)
                except (TypeError, ValueError):
                    sector_value = None
    else:
        try:
            sector_value = int(sector_raw)
        except (TypeError, ValueError):
            sector_value = None

    if sector_value is not None:
        # UI suele trabajar 1-based; convertir a 0-based para el snapshot si aplica.
        if isinstance(sector_value, list):
            if sector_value and all(val >= 1 for val in sector_value):
                sector_value = [val - 1 for val in sector_value]
        elif sector_value >= 1:
            sector_value -= 1
        result["sector"] = sector_value

    return result


def _extract_area_px(snapshot: SnapshotState, class_name: str, window: str) -> Optional[float]:
    stats = snapshot.get_class_stats(class_name, window)
    if not isinstance(stats, dict):
        return None
    try:
        return float(stats.get("area_avg")) if stats.get("area_avg") is not None else None
    except (TypeError, ValueError):
        return None


def _extract_area_cm(snapshot: SnapshotState, class_name: str, window: str) -> Optional[float]:
    stats = snapshot.get_class_stats(class_name, window)
    if not isinstance(stats, dict):
        return None
    # Nuevos snapshots guardan métricas específicas en area_avg_cm2/area_max_cm2
    cm_key = "area_avg_cm2"
    if cm_key in stats and stats[cm_key] is not None:
        try:
            return float(stats[cm_key])
        except (TypeError, ValueError):
            return None
    # Compatibilidad con datos antiguos: buscar nested en raw
    raw = stats.get("raw") if isinstance(stats.get("raw"), dict) else None
    if raw and isinstance(raw.get("area_cm2"), (int, float)):
        try:
            return float(raw["area_cm2"])
        except (TypeError, ValueError):
            return None
    return None


def _normalize_meta_rule_condition(payload: dict[str, object]) -> dict[str, object]:
    """Normalize a meta-rule condition (kind=rule) - used for rules that trigger based on other rules."""
    result: dict[str, object] = {"kind": CONDITION_KIND_RULE}

    # Helper interno
    def _coerce_int(key: str, default: int) -> int:
        try:
            val = int(payload.get(key) or default)
            return val
        except (TypeError, ValueError):
            return default

    # ID de regla
    rule_id = str(payload.get("rule_id") or "").strip()
    if rule_id:
        result["rule_id"] = rule_id

    # Parámetros numéricos
    result["min_firings"] = max(1, _coerce_int("min_firings", 1))
    
    max_f_raw = payload.get("max_firings")
    if max_f_raw not in (None, ""):
        try:
            max_f = int(max_f_raw)
            if max_f >= result["min_firings"]:
                result["max_firings"] = max_f
        except (TypeError, ValueError):
            pass

    result["window_sec"] = max(1, _coerce_int("window_sec", 60))
    result["debounce_ms"] = max(0, _coerce_int("debounce_ms", 0))
    result["cooldown_sec"] = max(0, _coerce_int("cooldown_sec", 0))
    
    # Etiqueta
    label = str(payload.get("label") or "").strip()
    if label:
        result["label"] = label
        
    return result


def _normalize_plc_condition(payload: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"kind": CONDITION_KIND_PLC_BIT}

    def _coerce_int(key: str, *, default: int = 0, aliases: tuple[str, ...] = (), minimum: int | None = None,
                    maximum: int | None = None) -> int:
        keys = (key, *aliases)
        raw = None
        for entry in keys:
            if entry in payload:
                raw = payload.get(entry)
                break
        if raw is None:
            value = default
        else:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = default
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    preset_id = str(payload.get("preset_id", "")).strip()
    if preset_id:
        result["preset_id"] = preset_id

    ip = str(payload.get("ip", "")).strip()
    if ip:
        result["ip"] = ip

    area = str(payload.get("area", "M")).strip().upper() or "M"
    if area not in {"M", "DB", "Q", "I"}:
        area = "M"
    result["area"] = area

    rack = _coerce_int("rack", default=0, minimum=0)
    slot = _coerce_int("slot", default=2, minimum=0)
    result["rack"] = rack
    result["slot"] = slot

    db_number_raw = payload.get("db_number")
    db_number: int | None
    if db_number_raw in {None, ""}:
        db_number = None
    else:
        try:
            db_number = int(db_number_raw)
        except (TypeError, ValueError):
            db_number = None
    result["db_number"] = db_number

    plc_mode = str(payload.get("plc_mode", "bit")).strip().lower()
    result["plc_mode"] = plc_mode

    if plc_mode == "numeric":
        result["address"] = str(payload.get("address", "")).strip()
        result["data_type"] = str(payload.get("data_type", "WORD")).strip()
        result["operator"] = str(payload.get("operator", "=")).strip()
        result["value1"] = str(payload.get("value1", "0")).strip()
        result["value2"] = str(payload.get("value2", "0")).strip()
    else:
        byte_index = _coerce_int("byte_index", aliases=("byte",), default=0, minimum=0)
        bit_index = _coerce_int("bit_index", aliases=("bit",), default=0, minimum=0, maximum=7)
        result["byte_index"] = byte_index
        result["bit_index"] = bit_index

        expected_raw = payload.get("expected_value")
        if expected_raw is None and "expected" in payload:
            expected_raw = payload.get("expected")
        if expected_raw is None:
            expected_raw = payload.get("value", True)
        expected = _ensure_bool(expected_raw, default=True)
        result["expected_value"] = expected

    tag = str(payload.get("tag", "")).strip()
    if tag:
        result["tag"] = tag

    label = str(payload.get("label", "")).strip()
    if label:
        result["label"] = label

    return result


def _ensure_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on", "si", "s", "sí"}:
            return True
        if normalized in {"false", "0", "no", "off", "n"}:
            return False
    return default


def _ensure_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def _cast_action(value: str) -> ActionChoice:
    return value if value in {"ignore", "freeze_level"} else "ignore"


def _cast_resume(value: str) -> ResumeMode:
    return value if value in {"instant", "delayed"} else "instant"


def _coerce_plc_action_params(params: dict[str, object]) -> dict[str, object]:
    if not isinstance(params, dict):
        raise ValueError("Parámetros PLC inválidos (no es diccionario)")

    def _coerce_int(key: str, *, default: int | None = None, required: bool = False) -> int | None:
        raw = params.get(key, default)
        if raw is None:
            if required:
                raise ValueError(f"Campo requerido '{key}' ausente")
            return None
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Campo '{key}' debe ser entero") from exc

    def _coerce_str(key: str, *, default: str = "", allowed: set[str] | None = None) -> str:
        raw = params.get(key, default)
        value = "" if raw is None else str(raw).strip()
        if allowed is not None and value and value.upper() not in allowed:
            raise ValueError(f"Campo '{key}' debe ser uno de: {', '.join(sorted(allowed))}")
        return value

    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError("Valor booleano no reconocido")

    preset_id = _coerce_str("preset_id")
    ip = _coerce_str("ip")
    area = _coerce_str("area", default="M", allowed={"M", "DB", "Q", "I"}).upper() or "M"
    rack = _coerce_int("rack", default=0) or 0
    slot = _coerce_int("slot", default=2) or 2
    db_number = _coerce_int("db_number")

    byte_idx = params.get("byte_index", params.get("byte"))
    bit_idx = params.get("bit_index", params.get("bit"))
    if byte_idx is None:
        raise ValueError("Campo 'byte' requerido")
    if bit_idx is None:
        raise ValueError("Campo 'bit' requerido")
    try:
        byte_index = int(byte_idx)
    except (TypeError, ValueError) as exc:
        raise ValueError("Campo 'byte' debe ser entero") from exc
    try:
        bit_index = int(bit_idx)
    except (TypeError, ValueError) as exc:
        raise ValueError("Campo 'bit' debe ser entero") from exc
    if bit_index < 0 or bit_index > 7:
        raise ValueError("Campo 'bit' debe estar entre 0 y 7")

    value_raw = params.get("value", True)
    if isinstance(value_raw, str):
        value_clean = value_raw.strip().lower()
        if value_clean in {"1", "true", "yes", "on"}:
            value = True
        elif value_clean in {"0", "false", "no", "off"}:
            value = False
        else:
            raise ValueError("Campo 'value' no reconocido")
    else:
        value = bool(value_raw)

    tag = _coerce_str("tag")

    targets_raw = params.get("targets", [])
    targets_list: list[dict[str, object]] = []
    if isinstance(targets_raw, str):
        tokens = [token.strip() for token in targets_raw.split(",") if token.strip()]
        for token in tokens:
            if "=" in token:
                coord, value_part = token.split("=", 1)
                target_value = _coerce_bool(value_part)
            else:
                coord = token
                target_value = value
            if "." not in coord:
                raise ValueError("Formato de objetivo adicional inválido (usar byte.bit)")
            byte_part, bit_part = coord.split(".", 1)
            try:
                byte_idx_extra = int(byte_part)
                bit_idx_extra = int(bit_part)
            except (TypeError, ValueError) as exc:
                raise ValueError("Objetivo adicional debe usar enteros para byte/bit") from exc
            if bit_idx_extra < 0 or bit_idx_extra > 7:
                raise ValueError("Bit en objetivo adicional debe estar entre 0 y 7")
            targets_list.append({
                "byte_index": byte_idx_extra,
                "bit_index": bit_idx_extra,
                "value": target_value,
            })
    elif isinstance(targets_raw, list):
        for item in targets_raw:
            if not isinstance(item, dict):
                continue
            try:
                byte_idx_extra = int(item.get("byte_index"))
                bit_idx_extra = int(item.get("bit_index"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Objetivo adicional debe incluir byte_index y bit_index enteros") from exc
            if bit_idx_extra < 0 or bit_idx_extra > 7:
                raise ValueError("Bit en objetivo adicional debe estar entre 0 y 7")
            try:
                value_extra = _coerce_bool(item.get("value", value))
            except ValueError as exc:
                raise ValueError("Valor de objetivo adicional inválido") from exc
            targets_list.append({
                "byte_index": byte_idx_extra,
                "bit_index": bit_idx_extra,
                "value": value_extra,
            })

    return {
        "preset_id": preset_id,
        "ip": ip,
        "rack": rack,
        "slot": slot,
        "area": area,
        "db_number": db_number,
        "byte_index": byte_index,
        "bit_index": bit_index,
        "value": value,
        "tag": tag,
        "targets": targets_list,
    }


def main(show_ui: bool = True) -> None:
    """Punto de entrada para ejecutar el servicio con o sin interfaz."""

    service = SendToPLCService()
    service.start()

    if show_ui and tk is not None and ttk is not None:
        root = tk.Tk()
        ttk.Style().configure("TCombobox", padding=4)
        ttk.Style().configure("TCheckbutton", padding=4)
        ttk.Style().configure("TRadiobutton", padding=2)
        SendToPLCWindow(root, service)
        try:
            root.mainloop()
        finally:
            service.stop()
    else:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            service.stop()


if __name__ == "__main__":
    main(show_ui=True)




