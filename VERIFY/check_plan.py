#!/usr/bin/env python3
"""Keep the plan's scoreboard honest: recount the ledger and flag drift.

The scoreboard went stale twice by being maintained by hand. Run this after
editing §0.6.
"""
import collections, re, sys, os

P = os.path.join(os.path.dirname(__file__), "..", "ISCAS_REVISION_PLAN.md")
s = open(P).read()
led = s[s.index("### Reviewer issues"):s.index("**Scoreboard")]
c = collections.Counter()
for line in led.splitlines():
    m = re.match(r"\|\s*\*{0,2}([A-Z][0-9-]+[^|]*?)\*{0,2}\s*\|"
                 r"\s*(OPEN|DATA|PART|DONE)\s*\|", line)
    if m:
        c[m.group(2)] += 1

stated = re.search(r"\*\*Scoreboard: (\d+) DONE · (\d+) DATA · (\d+) PART · (\d+) OPEN",
                   s)
want = (c["DONE"], c["DATA"], c["PART"], c["OPEN"])
print(f"  counted: {want[0]} DONE, {want[1]} DATA, {want[2]} PART, {want[3]} OPEN"
      f"  ({sum(c.values())} issues)")
if not stated:
    sys.exit("  ERROR: no scoreboard line found")
got = tuple(int(g) for g in stated.groups())
if got != want:
    sys.exit(f"  DRIFT: scoreboard says {got}, ledger says {want}")
print("  scoreboard matches the ledger")
