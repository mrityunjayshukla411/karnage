"""
_coverage_script.py --- GDB Python script for deterministic function-level coverage.

Loaded by the coverage runner via:
    gdb --batch -q -x _coverage_script.py --args python _wrapper.py <script.py>

Purpose
-------
Set ONE silent software breakpoint at the entry VMA of each function in
flip_sites.json.  When the function is called, record its name and immediately
continue — the inferior never visibly stops.  At process exit write all hit
function names to KARNAGE_COVERAGE_OUTPUT as JSON.

Why function-level, not site-level
-----------------------------------
Setting a breakpoint per flip site would require ~tens of thousands of ptrace()
calls during library load, which (a) is extremely slow and (b) conflicts with
CUDA's internal use of SIGTRAP for GPU synchronisation.  One breakpoint per
function (~hundreds to low thousands) avoids both issues while still providing
the key filter: any function whose entry is never reached during a real
compilation has zero live flip sites.

Environment variables consumed
------------------------------
  KARNAGE_SITES            Path to flip_sites.json
  KARNAGE_COVERAGE_OUTPUT  Path to write {"hit_functions": [...]} JSON
  KARNAGE_TARGET_SO        Substring matched against objfile names
                           (default: "libtriton.so")
"""

import json
import os

import gdb

gdb.execute("set pagination off")
gdb.execute("set debuginfod enabled off", to_string=True)
gdb.execute("set print thread-events off")
gdb.execute("set print inferior-events off")

_TARGET_SO = os.environ.get("KARNAGE_TARGET_SO", "libtriton.so")
_SITES_PATH = os.environ.get("KARNAGE_SITES", "")
_COVERAGE_OUTPUT = os.environ.get("KARNAGE_COVERAGE_OUTPUT", "")

_armed = False
_hit_functions: set[str] = set()
_total_set = 0


def _get_load_base(pid: int, soname: str) -> int:
    """Return the ASLR load base of *soname* from /proc/{pid}/maps."""
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if soname not in line:
                continue
            parts = line.split()
            if len(parts) >= 3 and int(parts[2], 16) == 0:
                return int(parts[0].split("-")[0], 16)
    raise RuntimeError(f"{soname!r} not found in /proc/{pid}/maps")


class CoverageBreakpoint(gdb.Breakpoint):
    """Silent breakpoint at a function entry that records the hit and continues."""

    def __init__(self, runtime_addr: int, func_name: str) -> None:
        super().__init__(f"*{runtime_addr:#x}")
        self.silent = True
        self._func_name = func_name

    def stop(self) -> bool:
        _hit_functions.add(self._func_name)
        self.delete()  # one-shot: remove after first call to avoid loop overhead
        return False   # continue without stopping the inferior


def on_new_objfile(event) -> None:
    global _armed, _total_set
    if _armed:
        return
    objfile = event.new_objfile
    if objfile is None or _TARGET_SO not in (objfile.filename or ""):
        return

    _armed = True

    if not _SITES_PATH:
        gdb.write("[coverage] KARNAGE_SITES not set — skipping\n", gdb.STDERR)
        return

    try:
        with open(_SITES_PATH) as f:
            sites = json.load(f)
    except Exception as exc:
        gdb.write(f"[coverage] failed to load sites: {exc}\n", gdb.STDERR)
        return

    inf = gdb.selected_inferior()
    try:
        load_base = _get_load_base(inf.pid, _TARGET_SO)
    except Exception as exc:
        gdb.write(f"[coverage] load_base error: {exc}\n", gdb.STDERR)
        return

    count = 0
    skipped = 0
    for func_name, fd in sites.get("functions", {}).items():
        vma_str = fd.get("vma", "0x0")
        vma = int(vma_str, 16)
        if vma == 0:
            skipped += 1
            continue
        addr = load_base + vma
        try:
            CoverageBreakpoint(addr, func_name)
            count += 1
        except Exception as exc:
            gdb.write(f"[coverage] bp failed 0x{addr:x}: {exc}\n", gdb.STDERR)
            skipped += 1

    _total_set = count
    gdb.write(
        f"[coverage] load_base=0x{load_base:016x}  "
        f"{count} function breakpoints set ({skipped} skipped)\n",
        gdb.STDERR,
    )


def on_exit(event) -> None:
    if not _COVERAGE_OUTPUT:
        gdb.write("[coverage] KARNAGE_COVERAGE_OUTPUT not set — results discarded\n", gdb.STDERR)
        return
    payload = {
        "total_set": _total_set,
        "total_hit": len(_hit_functions),
        "hit_functions": sorted(_hit_functions),
    }
    try:
        with open(_COVERAGE_OUTPUT, "w") as f:
            json.dump(payload, f, indent=2)
        gdb.write(
            f"[coverage] {len(_hit_functions)}/{_total_set} functions hit "
            f"→ {_COVERAGE_OUTPUT}\n",
            gdb.STDERR,
        )
    except Exception as exc:
        gdb.write(f"[coverage] failed to write output: {exc}\n", gdb.STDERR)


gdb.events.new_objfile.connect(on_new_objfile)
gdb.events.exited.connect(on_exit)
gdb.execute("run")
