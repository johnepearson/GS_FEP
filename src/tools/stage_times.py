"""Print the run time of every stage, longest first, from BUILD/logs/times.tsv (written by trace_run.py).
Usage: python3 src/tools/stage_times.py build      (make times, or make QUICK=1 times)
If a stage was run more than once, the latest run is shown.  Stages that failed or were stopped are marked."""
import os
import sys

def main():
    b = sys.argv[1] if len(sys.argv) > 1 else "build"
    path = os.path.join(b, "logs", "times.tsv")
    if not os.path.exists(path):
        print(f"{path} does not exist yet: no stage has run in {b}/")
        sys.exit(0)
    last = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 5:
            last[p[0]] = (p[1], p[2], int(p[3]), int(p[4]))
    rows = sorted(last.items(), key=lambda kv: -kv[1][2])


    def hms(s):
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


    print(f"{'stage':<18}{'started':<21}{'finished':<21}{'h:mm:ss':>9}")
    for name, (a, z, sec, st) in rows:
        print(f"{name:<18}{a:<21}{z:<21}{hms(sec):>9}" + ("" if st == 0 else f"   (exit {st})"))
    tot = sum(v[2] for v in last.values())
    print(f"{'sum of stages':<60}{hms(tot):>9}")
    first = min(v[0] for v in last.values())
    end = max(v[1] for v in last.values())
    print(f"first start {first}, last finish {end}  (stages can overlap, so the sum can exceed the elapsed time)")


try:
    main()
except BrokenPipeError:   # output cut short, as in  make times | head: stop quietly
    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(0)
