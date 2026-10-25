"""The engine's GPUs by PCI address, for the /vram routes (serve/vram_lend.py).  With --vram-lend the engine prints one
line per GPU at load, before READY:

    GPU-PCI dev=<n> pci=<address>

`dev` is the engine's own device number (the one LEND and LENDABLE use), `pci` the address CUDA gives that device.
The routes name a GPU by its address, so a borrower never sees device numbers, whatever CUDA_VISIBLE_DEVICES and the
config's gpu list are.  Without those lines (an engine started without --vram-lend, or one not running) the server
reads the addresses itself: nvidia_smi."""
from __future__ import annotations

import subprocess
import threading

LINE = "GPU-PCI "
SMI = ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"]

_smi: dict[int, str] | None = None     # nvidia_smi's answer, read once
_smi_lock = threading.Lock()


def parse(line: str) -> dict | None:
    """`GPU-PCI dev=0 pci=0000:01:00.0` -> {"dev": 0, "pci": "0000:01:00", "addr": "0000:01:00.0"} (`addr`: pci_addr,
    the GPU the /vram routes name); None for a line that is not one."""
    if not line.startswith(LINE):
        return None
    out: dict = {}
    for kv in line.split()[1:]:
        k, _, v = kv.partition("=")
        if k == "pci":
            out["pci"] = pci_key(v)
            out["addr"] = pci_addr(v)
        elif k == "dev" and v.isdigit():
            out["dev"] = int(v)
    return out if "dev" in out and out.get("addr") else None


def pci_key(s: str | None) -> str | None:
    """CUDA's "0000:01:00.0" and NVML's "00000000:01:00.0" (or the telemetry's "0000:01:00") as "0000:01:00"."""
    if not s or s == "-":
        return None
    f = s.strip().lower().split(":")
    if len(f) < 3:
        return None
    try:
        return f"{int(f[-3], 16):04x}:{int(f[-2], 16):02x}:{int(f[-1].split('.')[0], 16):02x}"
    except ValueError:
        return None


def pci_addr(s) -> str | None:
    """A GPU's full PCI address as one spelling, "0000:0a:00.0" (lower case, a 4-digit domain, the function): from
    CUDA's "0000:0A:00.0", NVML's "00000000:0A:00.0" or lspci's "0a:00.0" (domain 0, and function 0 when it has none),
    whatever their case; None for anything else."""
    if not isinstance(s, str):
        return None
    f = s.strip().lower().split(":")
    if not 2 <= len(f) <= 3 or len(f[0]) > 8:
        return None
    dev, _, fn = f[-1].partition(".")
    parts = ([f[0]] if len(f) == 3 else ["0"]) + [f[-2], dev, fn or "0"]
    if not all(p and len(p) <= 8 and all(c in "0123456789abcdef" for c in p) for p in parts):
        return None
    d, b, v, n = (int(p, 16) for p in parts)
    if d > 0xFFFF or b > 0xFF or v > 0x1F or n > 7:
        return None
    return f"{d:04x}:{b:02x}:{v:02x}.{n:x}"


def nvidia_smi() -> dict[int, str]:
    """{nvidia-smi index: PCI address} of every NVIDIA GPU, read once for the server's life (nvidia-smi numbers the
    cards by PCI bus, as the config's gpu list and CUDA_DEVICE_ORDER=PCI_BUS_ID do).  {} without nvidia-smi or when
    it fails; a timeout is not kept, the next call asks again."""
    global _smi
    with _smi_lock:
        if _smi is None:
            try:
                r = subprocess.run(SMI, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
                got = parse_smi(r.stdout) if r.returncode == 0 else {}
            except subprocess.TimeoutExpired:
                return {}
            except (OSError, ValueError):
                got = {}
            _smi = got
        return dict(_smi)


def parse_smi(text: str) -> dict[int, str]:
    """nvidia-smi's "0, 00000000:01:00.0" lines -> {0: "0000:01:00.0"}; lines that are not one are skipped."""
    out: dict[int, str] = {}
    for line in (text or "").splitlines():
        index, _, bus = line.partition(",")
        addr = pci_addr(bus)
        if index.strip().isdigit() and addr:
            out[int(index)] = addr
    return out
