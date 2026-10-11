#!/usr/bin/env python3
"""Rewrite Verilog parameter defaults into a per-run copy of the RTL.

The Genus flow here has no parameter passing, and hand-editing defaults 17
times is exactly where a sweep goes wrong. This copies each source file and
substitutes the requested defaults, so every run is reproducible from the
driver script alone and the checked-in RTL is never modified.

  set_params.py OUTDIR file.v [file.v ...] -- W=14 MODE=1

Fails loudly if a requested parameter is not found in any file, rather than
silently synthesising the default.
"""
import os, re, shutil, sys


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: set_params.py OUTDIR file.v [...] -- NAME=VAL [...]")
    cut = argv.index("--")
    outdir, files, assigns = argv[0], argv[1:cut], argv[cut + 1:]
    os.makedirs(outdir, exist_ok=True)

    want = {}
    for a in assigns:
        k, _, v = a.partition("=")
        want[k.strip()] = v.strip()

    hits = {k: 0 for k in want}
    for src in files:
        text = open(src).read()
        for name, val in want.items():
            pat = re.compile(r"(^\s*parameter\s+(?:signed\s+)?" + re.escape(name)
                             + r"\s*=\s*)(-?\d+)", re.M)
            text, n = pat.subn(lambda m: m.group(1) + val, text)
            hits[name] += n
        open(os.path.join(outdir, os.path.basename(src)), "w").write(text)

    missing = [k for k, v in hits.items() if v == 0]
    if missing:
        sys.exit(f"ERROR: parameter(s) {missing} not found in {files} — "
                 "the run would have used the defaults. Aborting.")
    print("  params: " + ", ".join(f"{k}={v} ({hits[k]}x)"
                                   for k, v in want.items()))


if __name__ == "__main__":
    main()
