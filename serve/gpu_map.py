"""The engine's GPUs by PCI address, for the /vram routes (serve/vram_lend.py).  With --vram-lend the engine prints one
line per GPU at load, before READY:

    GPU-PCI dev=<n> pci=<address>

`dev` is the engine's own device number (the one LEND and LENDABLE use), `pci` the address CUDA gives that device.
The routes name a GPU by its address, so a borrower never sees device numbers, whatever CUDA_VISIBLE_DEVICES and the
config's gpu list are."""
from __future__ import annotations

LINE = "GPU-PCI "


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
