"""
_gdb_script.py — GDB Python script for MatcherTable bit-flip injection.

Loaded by the orchestrator via:
    gdb --batch -q -x _gdb_script.py --args python _wrapper.py <triton_script.py>

Why new_objfile instead of stop-on-solib-events:
  stop-on-solib-events causes gdb.execute("run") to RETURN the moment the
  inferior first stops (first solib event).  In --batch mode the script then
  ends and GDB exits, so any gdb.post_event(continue) is discarded and the
  inferior never actually runs.

  gdb.events.new_objfile fires while the inferior is paused at the dynamic
  linker probe point (_dl_debug_state).  Memory is fully mapped and writable
  via ptrace at that point.  After our handler returns, the inferior resumes
  automatically — no explicit continue required.  gdb.execute("run") then
  blocks until the inferior exits normally, which is exactly what we need in
  batch mode.

Environment variables consumed:
  KARNAGE_PATCH_SPEC   — path to {"patch_vmas": [int, ...], "mask": int}
  KARNAGE_TARGET_SO    — substring matched against objfile filenames
                         (default: "libtriton.so")

Docker note: requires --cap-add SYS_PTRACE (or --security-opt seccomp=unconfined).
"""
import gdb
import json
import os

gdb.execute("set pagination off")
gdb.execute("set debuginfod enabled off", to_string=True)

_TARGET_SO = os.environ.get("KARNAGE_TARGET_SO", "libtriton.so")
_patched = False


def _get_load_base(pid: int, soname: str) -> int:
    """
    Return the ASLR load base of *soname* from /proc/{pid}/maps.
    The load base is the start address of the mapping whose file offset is 0.
    """
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if soname not in line:
                continue
            parts = line.split()
            if len(parts) >= 3 and int(parts[2], 16) == 0:
                return int(parts[0].split('-')[0], 16)
    raise RuntimeError(f"{soname!r} not found in /proc/{pid}/maps")


def _apply_patches(patch_vmas: list, mask: int) -> None:
    inf = gdb.selected_inferior()
    load_base = _get_load_base(inf.pid, _TARGET_SO)
    gdb.write(
        f"[karnage] load_base=0x{load_base:016x}  "
        f"mask=0x{mask:02x}  targets={len(patch_vmas)}\n"
    )
    for vma in patch_vmas:
        addr = load_base + vma
        try:
            buf     = bytes(inf.read_memory(addr, 1))
            patched = buf[0] ^ mask
            inf.write_memory(addr, bytes([patched]))
            gdb.write(f"[karnage]   0x{addr:016x}: 0x{buf[0]:02x} -> 0x{patched:02x}\n")
        except gdb.MemoryError as exc:
            gdb.write(f"[karnage]   write failed at 0x{addr:016x}: {exc}\n")


def on_new_objfile(event):
    global _patched
    if _patched:
        return
    objfile = event.new_objfile
    if objfile is None or _TARGET_SO not in (objfile.filename or ""):
        return

    _patched = True
    spec_path = os.environ.get("KARNAGE_PATCH_SPEC")
    if not spec_path:
        return
    try:
        with open(spec_path) as f:
            spec = json.load(f)
        _apply_patches(spec["patch_vmas"], spec["mask"])
    except Exception as exc:
        gdb.write(f"[karnage] patch error: {exc}\n")


gdb.events.new_objfile.connect(on_new_objfile)
gdb.execute("run")
