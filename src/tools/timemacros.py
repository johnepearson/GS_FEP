"""Write build/results/times.tex: run-time macros for the reproducibility table of the supplement.

Usage:  python3 src/tools/timemacros.py BUILD_FOLDER MAPPING_FILE      (the Makefile does this)

Input   BUILD/logs/times.tsv     one line per stage run: stage, start, end, seconds, exit status (written by trace_run.py)
        MAPPING_FILE             one line per macro:   TimeName: stage stage ...     (# starts a comment)
Output  BUILD/results/times.tex  \\newcommand{\\TimeName}{...}  holding the summed time of the listed stages, rounded:
            under 3 minutes     $<$3 min
            3 to 60 minutes     whole minutes            (12 min)
            over 60 minutes     hours to one decimal     (2.9 h)
        plus \\TimeOther (stages that are in no table row), \\TimeTotal (sum of all stages) and \\TimeWall (first start to last finish).
A stage that ran more than once (stopped and resumed, or failed and rerun) counts ALL its runs: the time actually spent.
Stages listed in the mapping but missing from times.tsv give the value nan, so  make check  reports them.
"""
import os
import re
import sys
from datetime import datetime


def fmt(sec):
    """Round a time in seconds the way the supplement quotes it."""
    if sec < 180:
        return "$<$3~min"
    if sec < 3570:                       # up to 59.5 minutes
        return f"{round(sec / 60)}~min"
    return f"{sec / 3600:.1f}~h"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build, mapfile = sys.argv[1], sys.argv[2]
    path = os.path.join(build, "logs", "times.tsv")
    if not os.path.exists(path):
        sys.exit(f"{path} does not exist: no stage has run in {build}/")
    secs, runs, starts, ends = {}, {}, [], []
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) != 5:
            continue
        secs[p[0]] = secs.get(p[0], 0) + int(p[3])
        runs[p[0]] = runs.get(p[0], 0) + 1
        starts.append(p[1])
        ends.append(p[2])

    out = ["% Run times of the stages behind each row of the reproducibility table.  Written by src/tools/timemacros.py; do not edit."]
    seen, mapped = set(), set()
    for n, line in enumerate(open(mapfile), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"(Time[A-Za-z]+)\s*:\s*(.+)", line)
        if not m:
            sys.exit(f"{mapfile}:{n}: expected  TimeName: stage stage ...   (letters only in the name), got: {line}")
        name, stages = m.group(1), m.group(2).split()
        if name in seen:
            sys.exit(f"{mapfile}:{n}: {name} is defined twice")
        seen.add(name)
        mapped.update(stages)
        missing = [s for s in stages if s not in secs]
        if missing:
            print(f"timemacros: {name}: no run time recorded for {' '.join(missing)} (value nan)", file=sys.stderr)
            val = "nan"
        else:
            val = fmt(sum(secs[s] for s in stages))
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    multi = sorted(s for s, k in runs.items() if k > 1)
    if multi:
        print("timemacros: stages run more than once (all runs counted): " + " ".join(multi), file=sys.stderr)

    rest = sorted(set(secs) - mapped)
    if rest:
        print("timemacros: stages in no table row (counted in \\TimeOther): " + " ".join(rest), file=sys.stderr)
    big = [s for s in rest if secs[s] >= 180]
    if big:
        print("timemacros: WARNING: these stages take 3 minutes or more but are in no table row, so they would be hidden in \\TimeOther: "
              + " ".join(f"{s} ({fmt(secs[s]).replace('~', ' ')})" for s in big), file=sys.stderr)
    out.append(f"\\newcommand{{\\TimeOther}}{{{fmt(sum(secs[s] for s in rest))}}}")
    t0 = datetime.strptime(min(starts), "%Y-%m-%d %H:%M:%S")
    t1 = datetime.strptime(max(ends), "%Y-%m-%d %H:%M:%S")
    out.append(f"\\newcommand{{\\TimeTotal}}{{{fmt(sum(secs.values()))}}}")
    out.append(f"\\newcommand{{\\TimeWall}}{{{fmt((t1 - t0).total_seconds())}}}")
    os.makedirs(os.path.join(build, "results"), exist_ok=True)
    dest = os.path.join(build, "results", "times.tex")
    open(dest, "w").write("\n".join(out) + "\n")
    print(f"wrote {dest} ({len(out) - 1} macros)")


main()
