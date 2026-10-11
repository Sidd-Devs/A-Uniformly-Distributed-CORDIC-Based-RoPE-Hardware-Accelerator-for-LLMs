#!/usr/bin/env python3
"""Pull area / slack / power out of every completed run into one CSV.

  python3 collect_results.py build            # table to stdout
  python3 collect_results.py build > r.csv    # CSV to paste back

Reads the same fields the earlier campaign used, so new numbers line up with
the ones already in the plan: 'Total Cell Area' and 'Slack (ps)' from
final.rpt, and the Subtotal row of power.rpt (converted to mW).
"""
import os, re, sys


def scan(run_dir):
    rep = os.path.join(run_dir, "reports")
    if not os.path.isdir(rep):
        return None
    for design in sorted(os.listdir(rep)):
        opt = os.path.join(rep, design, "opt")
        fin = os.path.join(opt, "final.rpt")
        pwr = os.path.join(opt, "power.rpt")
        if not os.path.exists(fin):
            continue
        area = slack = power = ""
        for line in open(fin, errors="ignore"):
            if line.startswith("Total Cell Area:"):
                area = line.replace(",", "").split()[-1]
            elif line.startswith("Slack (ps):"):
                slack = line.replace(",", "").split()[-1]
        if os.path.exists(pwr):
            for line in open(pwr, errors="ignore"):
                if "Subtotal" in line:
                    f = line.split()
                    if len(f) >= 5:
                        try:
                            power = f"{float(f[4]) * 1e3:.4f}"
                        except ValueError:
                            pass
        return design, area, slack, power
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "build"
    rows, missing = [], []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        r = scan(p)
        if r:
            rows.append((d,) + r)
        else:
            missing.append(d)

    print("run,design,area_um2,slack_ps,power_mW")
    for run, design, a, s, w in rows:
        print(f"{run},{design},{a},{s},{w}")

    if missing:
        print("\n# INCOMPLETE (no reports found):", file=sys.stderr)
        for m in missing:
            print(f"#   {m}", file=sys.stderr)
    print(f"\n# {len(rows)} runs collected, {len(missing)} incomplete",
          file=sys.stderr)


if __name__ == "__main__":
    main()
