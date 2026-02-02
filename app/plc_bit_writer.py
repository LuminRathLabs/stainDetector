# plc_bit_writer.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
import socket
import platform
import subprocess
import threading
import re

# ---- snap7 ----
try:
    import snap7
    from snap7.types import Areas
    import snap7.util as s7u
except Exception as e:
    raise RuntimeError(
        "Falta python-snap7. Instálalo con: pip install python-snap7"
    ) from e

# ---- logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("S7BitWriter")

# ---- helpers ----
def ping_tcp_102(ip: str, timeout=2.0) -> bool:
    try:
        socket.create_connection((ip, 102), timeout=timeout).close()
        return True
    except Exception:
        # fallback ping una vez
        try:
            param = "-n" if platform.system().lower() == "windows" else "-c"
            return subprocess.call(["ping", param, "1", ip],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL) == 0
        except Exception:
            return False

def parse_s7_address(addr: str):
    if not isinstance(addr, str):
        raise ValueError("Direccion invalida")
    a = addr.strip().upper()
    if not a:
        raise ValueError("Direccion vacia")
    a = re.sub(r"\s+", "", a)
    a = a.replace(",", ".")

    m = re.match(r"^DB(\d+)(?:\.)?DB([XBWD])(.+)$", a)
    if not m:
        raise ValueError("Formato esperado: DB<num>.DBW<offset> (ej: DB6.DBW0)")

    try:
        db_number = int(m.group(1))
    except ValueError as e:
        raise ValueError("DB numero invalido") from e

    access_char = m.group(2)
    rest = m.group(3).replace("O", "0")

    bit_offset = None
    if access_char == "X":
        if "." not in rest:
            raise ValueError("DBX requiere bit: DB6.DBX0.0")
        byte_str, bit_str = rest.split(".", 1)
        if not byte_str or not bit_str:
            raise ValueError("DBX requiere byte y bit: DB6.DBX0.0")
        try:
            byte_offset = int(byte_str)
            bit_offset = int(bit_str)
        except ValueError as e:
            raise ValueError("Offset o bit invalido") from e
        if bit_offset < 0 or bit_offset > 7:
            raise ValueError("Bit debe estar entre 0..7")
        access = "DBX"
        size_bytes = 1
    else:
        if "." in rest:
            raise ValueError("Formato invalido para DBB/DBW/DBD")
        try:
            byte_offset = int(rest)
        except ValueError as e:
            raise ValueError("Offset invalido") from e
        access = "DB" + access_char
        if access_char == "B":
            size_bytes = 1
        elif access_char == "W":
            size_bytes = 2
        else:
            size_bytes = 4

    return {
        "area": Areas.DB,
        "db_number": db_number,
        "byte_offset": byte_offset,
        "bit_offset": bit_offset,
        "access": access,
        "size_bytes": size_bytes,
    }

def parse_bool(text: str) -> bool:
    if isinstance(text, bool):
        return text
    val = str(text).strip().lower()
    if val in ("1", "true", "t", "yes", "y", "on"):
        return True
    if val in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError("Valor BOOL invalido (use 0/1 o true/false)")

class PLC:
    def __init__(self):
        self.client = snap7.client.Client()
        self.lock = threading.Lock()

    def connect(self, ip: str, rack: int, slot: int) -> bool:
        log.info(f"Connecting to {ip}:102 rack {rack} slot {slot}")
        if not ping_tcp_102(ip):
            log.error(f"IP {ip} not reachable")
            return False
        try:
            with self.lock:
                # desconecta si ya estaba
                try:
                    if self.client.get_connected():
                        self.client.disconnect()
                except Exception:
                    pass
                self.client.connect(ip, rack, slot)
                ok = self.client.get_connected()
            if ok:
                log.info(f"Connected to PLC at {ip}")
            else:
                log.error(f"Could not connect to {ip}")
            return ok
        except Exception as e:
            log.error(f"Connection error: {e}")
            return False

    def disconnect(self):
        with self.lock:
            try:
                if self.client.get_connected():
                    self.client.disconnect()
            except Exception:
                pass
        log.info("Disconnected from PLC")

    def is_connected(self) -> bool:
        try:
            return self.client.get_connected()
        except Exception:
            return False

    # --- Merker (M) read/write byte.bit ---
    def read_m_bit(self, byte_idx: int, bit_idx: int):
        if bit_idx < 0 or bit_idx > 7:
            raise ValueError("Bit debe estar entre 0..7")
        with self.lock:
            raw = self.client.read_area(Areas.MK, 0, byte_idx, 1)
        return bool(s7u.get_bool(raw, 0, bit_idx))

    def write_m_bit(self, byte_idx: int, bit_idx: int, value: bool):
        if bit_idx < 0 or bit_idx > 7:
            raise ValueError("Bit debe estar entre 0..7")
        with self.lock:
            # leer 1 byte, modificar bit y devolver
            try:
                raw = self.client.read_area(Areas.MK, 0, byte_idx, 1)
            except Exception as e:
                # típicamente: b'CPU : Address out of range'
                raise RuntimeError(
                    f"No se pudo leer M{byte_idx}.{bit_idx}: {e}"
                ) from e
            s7u.set_bool(raw, 0, bit_idx, bool(value))
            try:
                self.client.write_area(Areas.MK, 0, byte_idx, raw)
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo escribir M{byte_idx}.{bit_idx}: {e}"
                ) from e
        log.info(f"Wrote {int(bool(value))} to M{byte_idx}.{bit_idx}")
        return True

    # --- DB read/write helpers ---
    def read_db_bytes(self, db_number: int, start: int, size: int) -> bytes:
        with self.lock:
            return self.client.read_area(Areas.DB, db_number, start, size)

    def write_db_bytes(self, db_number: int, start: int, data: bytes):
        with self.lock:
            self.client.write_area(Areas.DB, db_number, start, data)

    def read_db_bit(self, db_number: int, byte_offset: int, bit_idx: int) -> bool:
        if bit_idx < 0 or bit_idx > 7:
            raise ValueError("Bit debe estar entre 0..7")
        raw = self.read_db_bytes(db_number, byte_offset, 1)
        return bool(s7u.get_bool(raw, 0, bit_idx))

    def write_db_bit(self, db_number: int, byte_offset: int, bit_idx: int, value: bool):
        if bit_idx < 0 or bit_idx > 7:
            raise ValueError("Bit debe estar entre 0..7")
        raw = self.read_db_bytes(db_number, byte_offset, 1)
        s7u.set_bool(raw, 0, bit_idx, bool(value))
        self.write_db_bytes(db_number, byte_offset, raw)

    def read_db_byte(self, db_number: int, byte_offset: int) -> int:
        raw = self.read_db_bytes(db_number, byte_offset, 1)
        return int(raw[0])

    def write_db_byte(self, db_number: int, byte_offset: int, value: int):
        data = bytearray(1)
        data[0] = int(value) & 0xFF
        self.write_db_bytes(db_number, byte_offset, data)

    def read_db_word(self, db_number: int, byte_offset: int, signed: bool = False) -> int:
        raw = self.read_db_bytes(db_number, byte_offset, 2)
        return s7u.get_int(raw, 0) if signed else s7u.get_uint(raw, 0)

    def write_db_word(self, db_number: int, byte_offset: int, value: int, signed: bool = False):
        data = bytearray(2)
        if signed:
            s7u.set_int(data, 0, int(value))
        else:
            s7u.set_uint(data, 0, int(value))
        self.write_db_bytes(db_number, byte_offset, data)

    def read_db_real(self, db_number: int, byte_offset: int) -> float:
        raw = self.read_db_bytes(db_number, byte_offset, 4)
        return float(s7u.get_real(raw, 0))

    def write_db_real(self, db_number: int, byte_offset: int, value: float):
        data = bytearray(4)
        s7u.set_real(data, 0, float(value))
        self.write_db_bytes(db_number, byte_offset, data)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("S7 PLC Reader/Writer (M/DB)")
        self.geometry("720x480")
        self.resizable(False, False)

        self.plc = PLC()
        self._build_ui()

        # valores por defecto pedidos
        self.ip_entry.insert(0, "17.2.193.160")
        self.rack_entry.insert(0, "0")
        self.slot_entry.insert(0, "2")
        self.byte_entry.insert(0, "500")
        self.bit_box.set("0")
        self.addr_entry.insert(0, "DB6.DBW0")
        self.type_box.set("UINT")
        self.db_value_entry.insert(0, "0")

        self._set_controls(False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- UI ----
    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Connection
        con = ttk.LabelFrame(main, text="Connection")
        con.pack(fill=tk.X, pady=6)

        ttk.Label(con, text="IP:").grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.ip_entry = ttk.Entry(con, width=16)
        self.ip_entry.grid(row=0, column=1, padx=4)

        ttk.Label(con, text="Rack:").grid(row=0, column=2, padx=4)
        self.rack_entry = ttk.Entry(con, width=4)
        self.rack_entry.grid(row=0, column=3, padx=4)

        ttk.Label(con, text="Slot:").grid(row=0, column=4, padx=4)
        self.slot_entry = ttk.Entry(con, width=4)
        self.slot_entry.grid(row=0, column=5, padx=4)

        self.conn_btn = ttk.Button(con, text="Connect", command=self.on_connect_toggle)
        self.conn_btn.grid(row=0, column=6, padx=10)

        # Operations
        ops = ttk.LabelFrame(main, text="Read/Write")
        ops.pack(fill=tk.X, pady=6)

        mode_frame = ttk.Frame(ops)
        mode_frame.pack(fill=tk.X, pady=4)
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, padx=4, sticky="w")
        self.mode_var = tk.StringVar(value="DB")
        ttk.Radiobutton(mode_frame, text="M bit", variable=self.mode_var, value="M",
                        command=self._update_mode_ui).grid(row=0, column=1, padx=6, sticky="w")
        ttk.Radiobutton(mode_frame, text="DB address", variable=self.mode_var, value="DB",
                        command=self._update_mode_ui).grid(row=0, column=2, padx=6, sticky="w")

        self.m_frame = ttk.Frame(ops)
        self.m_frame.pack(fill=tk.X, pady=4)
        ttk.Label(self.m_frame, text="Byte:").grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.byte_entry = ttk.Entry(self.m_frame, width=8)
        self.byte_entry.grid(row=0, column=1, padx=4)

        ttk.Label(self.m_frame, text="Bit:").grid(row=0, column=2, padx=4, sticky="w")
        self.bit_box = ttk.Combobox(self.m_frame, width=4,
                                    values=[str(i) for i in range(8)], state="readonly")
        self.bit_box.grid(row=0, column=3, padx=4)

        self.value_var = tk.BooleanVar(value=False)
        self.value_chk = ttk.Checkbutton(self.m_frame, text="Value (1/0)",
                                         variable=self.value_var)
        self.value_chk.grid(row=0, column=4, padx=12)

        self.db_frame = ttk.Frame(ops)
        self.db_frame.pack(fill=tk.X, pady=4)
        ttk.Label(self.db_frame, text="Address:").grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.addr_entry = ttk.Entry(self.db_frame, width=18)
        self.addr_entry.grid(row=0, column=1, padx=4)

        ttk.Label(self.db_frame, text="Type:").grid(row=0, column=2, padx=4, sticky="w")
        self.type_box = ttk.Combobox(self.db_frame, width=8, state="readonly",
                                     values=["UINT", "INT", "REAL", "BYTE", "BOOL"])
        self.type_box.grid(row=0, column=3, padx=4)

        ttk.Label(self.db_frame, text="Value:").grid(row=0, column=4, padx=4, sticky="w")
        self.db_value_entry = ttk.Entry(self.db_frame, width=12)
        self.db_value_entry.grid(row=0, column=5, padx=4)

        btn_frame = ttk.Frame(ops)
        btn_frame.pack(fill=tk.X, pady=4)
        self.write_btn = ttk.Button(btn_frame, text="Write", command=self.on_write)
        self.write_btn.grid(row=0, column=0, padx=4)
        self.read_btn = ttk.Button(btn_frame, text="Read", command=self.on_read)
        self.read_btn.grid(row=0, column=1, padx=4)

        # Data & Log
        data_frame = ttk.LabelFrame(main, text="Data")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.data_text = scrolledtext.ScrolledText(data_frame, height=6)
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        log_frame = ttk.LabelFrame(main, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log_text.config(state=tk.DISABLED)
        self._update_mode_ui()

    def _log(self, level: str, msg: str):
        getattr(log, level.lower())(msg)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{level.upper()}: {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _set_controls(self, connected: bool):
        state = "normal" if connected else "disabled"
        self.write_btn["state"] = state
        self.read_btn["state"] = state
        self.value_chk["state"] = state
        self.db_value_entry["state"] = state

    def _update_mode_ui(self):
        if self.mode_var.get() == "M":
            self.db_frame.pack_forget()
            self.m_frame.pack(fill=tk.X, pady=4)
        else:
            self.m_frame.pack_forget()
            self.db_frame.pack(fill=tk.X, pady=4)

    def _validate_db_type(self, dtype: str, parsed: dict):
        access = parsed["access"]
        if dtype in ("UINT", "INT") and access != "DBW":
            raise ValueError("Tipo requiere DBW (ej: DB6.DBW0)")
        if dtype == "REAL" and access != "DBD":
            raise ValueError("Tipo REAL requiere DBD (ej: DB6.DBD0)")
        if dtype == "BYTE" and access != "DBB":
            raise ValueError("Tipo BYTE requiere DBB (ej: DB6.DBB0)")
        if dtype == "BOOL" and access != "DBX":
            raise ValueError("Tipo BOOL requiere DBXb.bit (ej: DB6.DBX0.0)")

    def _parse_db_value(self, dtype: str, text: str):
        if dtype == "REAL":
            return float(text)
        if dtype == "BOOL":
            return parse_bool(text)
        value = int(text)
        if dtype == "UINT" and (value < 0 or value > 65535):
            raise ValueError("UINT fuera de rango 0..65535")
        if dtype == "INT" and (value < -32768 or value > 32767):
            raise ValueError("INT fuera de rango -32768..32767")
        if dtype == "BYTE" and (value < 0 or value > 255):
            raise ValueError("BYTE fuera de rango 0..255")
        return value

    def _read_db_value(self, parsed: dict, dtype: str):
        dbn = parsed["db_number"]
        offset = parsed["byte_offset"]
        if dtype == "UINT":
            return self.plc.read_db_word(dbn, offset, signed=False)
        if dtype == "INT":
            return self.plc.read_db_word(dbn, offset, signed=True)
        if dtype == "REAL":
            return self.plc.read_db_real(dbn, offset)
        if dtype == "BYTE":
            return self.plc.read_db_byte(dbn, offset)
        if dtype == "BOOL":
            bit = parsed["bit_offset"]
            return int(self.plc.read_db_bit(dbn, offset, bit))
        raise ValueError("Tipo no soportado")

    def _write_db_value(self, parsed: dict, dtype: str, value):
        dbn = parsed["db_number"]
        offset = parsed["byte_offset"]
        if dtype == "UINT":
            return self.plc.write_db_word(dbn, offset, int(value), signed=False)
        if dtype == "INT":
            return self.plc.write_db_word(dbn, offset, int(value), signed=True)
        if dtype == "REAL":
            return self.plc.write_db_real(dbn, offset, float(value))
        if dtype == "BYTE":
            return self.plc.write_db_byte(dbn, offset, int(value))
        if dtype == "BOOL":
            bit = parsed["bit_offset"]
            return self.plc.write_db_bit(dbn, offset, bit, bool(value))
        raise ValueError("Tipo no soportado")

    # ---- handlers ----
    def on_connect_toggle(self):
        if self.plc.is_connected():
            self.plc.disconnect()
            self.conn_btn.config(text="Connect")
            self._set_controls(False)
            self._log("info", "Disconnected")
            return

        ip = self.ip_entry.get().strip()
        try:
            rack = int(self.rack_entry.get())
            slot = int(self.slot_entry.get())
        except ValueError:
            self._log("error", "Rack y Slot deben ser enteros")
            return

        if self.plc.connect(ip, rack, slot):
            self.conn_btn.config(text="Disconnect")
            self._set_controls(True)
            self._log("info", f"Connected to {ip} (rack {rack}, slot {slot})")
        else:
            self._set_controls(False)
            messagebox.showerror("Connection", f"No se pudo conectar a {ip} (rack {rack}, slot {slot})")

    def on_write(self):
        if not self.plc.is_connected():
            self._log("error", "Not connected")
            return
        if self.mode_var.get() == "M":
            try:
                byte_idx = int(self.byte_entry.get())
                bit_idx = int(self.bit_box.get())
            except ValueError:
                self._log("error", "Byte y Bit deben ser enteros")
                return

            try:
                ok = self.plc.write_m_bit(byte_idx, bit_idx, self.value_var.get())
                if ok:
                    val = self.plc.read_m_bit(byte_idx, bit_idx)
                    self.data_text.delete(1.0, tk.END)
                    self.data_text.insert(tk.END, f"M{byte_idx}.{bit_idx} = {int(val)}\n")
                    self._log("info", f"Write OK M{byte_idx}.{bit_idx} -> {int(self.value_var.get())}")
            except Exception as e:
                self._log("error", str(e))
                messagebox.showerror("Write error",
                                     f"No se pudo escribir M{byte_idx}.{bit_idx}\n\n{e}")
            return

        addr = self.addr_entry.get().strip()
        dtype = self.type_box.get().strip().upper()
        try:
            if not addr:
                raise ValueError("Direccion requerida")
            if not dtype:
                raise ValueError("Tipo requerido")
            parsed = parse_s7_address(addr)
            self._validate_db_type(dtype, parsed)
            value_str = self.db_value_entry.get().strip()
            if value_str == "":
                raise ValueError("Value requerido")
            value = self._parse_db_value(dtype, value_str)
            self._write_db_value(parsed, dtype, value)
            read_back = self._read_db_value(parsed, dtype)
            display_val = int(read_back) if dtype == "BOOL" else read_back
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, f"{addr} ({dtype}) = {display_val}\n")
            self._log("info", f"Write OK {addr} ({dtype}) -> {value}")
        except Exception as e:
            self._log("error", str(e))
            messagebox.showerror("Write error",
                                 f"No se pudo escribir {addr}\n\n{e}")

    def on_read(self):
        if not self.plc.is_connected():
            self._log("error", "Not connected")
            return
        if self.mode_var.get() == "M":
            try:
                byte_idx = int(self.byte_entry.get())
                bit_idx = int(self.bit_box.get())
                val = self.plc.read_m_bit(byte_idx, bit_idx)
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, f"M{byte_idx}.{bit_idx} = {int(val)}\n")
                self._log("info", f"Read M{byte_idx}.{bit_idx} -> {int(val)}")
            except Exception as e:
                self._log("error", str(e))
                messagebox.showerror("Read error",
                                     f"No se pudo leer M{byte_idx}.{bit_idx}\n\n{e}")
            return

        addr = self.addr_entry.get().strip()
        dtype = self.type_box.get().strip().upper()
        try:
            if not addr:
                raise ValueError("Direccion requerida")
            if not dtype:
                raise ValueError("Tipo requerido")
            parsed = parse_s7_address(addr)
            self._validate_db_type(dtype, parsed)
            val = self._read_db_value(parsed, dtype)
            display_val = int(val) if dtype == "BOOL" else val
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, f"{addr} ({dtype}) = {display_val}\n")
            self._log("info", f"Read {addr} ({dtype}) -> {display_val}")
        except Exception as e:
            self._log("error", str(e))
            messagebox.showerror("Read error",
                                 f"No se pudo leer {addr}\n\n{e}")

    def on_close(self):
        try:
            if self.plc.is_connected():
                self.plc.disconnect()
        finally:
            self.destroy()


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        input("Presione Enter para salir...")
        sys.exit(1)
