"""Run one analysis script and record what it depended on, as a Makefile fragment (like gcc -MMD does for C).

Arguments of the form @Name become the value of that parameter (and a dependency on it).
Usage (from build/, with TOP set to the project folder; the Makefile does this):
    python3 ../src/tools/trace_run.py deps/NAME.d results/NAME.tex ../src/analysis/script.py [args ...]

While the script runs this records
  - every project file it imports (the script itself, texout.py, partial_corr_utils.py, ...),
  - every file it opens for reading (the data files; symbolic links in build/ count; initialize.tex does not,
    because its numbers are tracked one by one through the stamp files below),
  - every parameter it reads through params.load() (see params.py),
and writes them to deps/NAME.d as   results/NAME.tex: <files> <build/params/Name.stamp ...>
It also records the files the run created or changed (the simulation data), as   <file>: <target>   lines, so that an
analysis that reads those files is rebuilt after the simulation that makes them.  A target ending in .done is a
sentinel written when the run succeeds (simulations use it, since they make many files).
The Makefile includes the .d files, so the next "make" reruns this script only when one of those changes.
The .d file is written even if the script fails.  Dependencies are recorded relative to the project folder.
"""
import builtins
import io
import os
import runpy
import sys
import time

TOP = os.path.abspath(os.environ["TOP"])
depfile, target, script = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
args = sys.argv[4:]
# An argument may name a parameter as @Name; it is replaced by that parameter's value from initialize.tex, and the run
# then depends on that parameter (the Makefile never holds the numbers).
import re
import params as _params
_P = _params.load("initialize.tex")
args = [re.sub(r"@([A-Za-z]+)", lambda m: str(_P[m.group(1)]), a) for a in args]

opened = set()
_orig_open = builtins.open


def _tracking_open(file, mode="r", *a, **k):
    if isinstance(file, (str, os.PathLike)) and not any(c in str(mode) for c in "wax+"):
        opened.add(os.path.abspath(os.fspath(file)))
    return _orig_open(file, mode, *a, **k)


def _snapshot(root):
    snap = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not (dp == root and d in ("results", "logs", "deps", "params", "sim", "__pycache__"))]
        for f in fn:
            fp = os.path.join(dp, f)
            try:
                st = os.stat(fp)
                snap[fp] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return snap


_before = _snapshot(os.getcwd())
builtins.open = _tracking_open
io.open = _tracking_open

sys.path.insert(0, os.path.dirname(script))
sys.argv = [script] + args
status = 0
import signal
_t0 = time.time()   # for the run-time record in logs/times.tsv


interrupted = False   # set by any stop signal, so that a script cannot turn a stop into a clean exit


def _terminated(signum, frame):   # being stopped is a failure: no completion mark is written
    global interrupted
    interrupted = True
    raise SystemExit(128 + signum)


for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(_sig, _terminated)
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if isinstance(e.code, str):   # a script that stops with raise SystemExit("message") must still show the message
        print(e.code, file=sys.stderr, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    status = 1
finally:
    import params
    deps = {script}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).startswith(TOP + os.sep) and f.endswith(".py") \
                and os.path.abspath(f) != os.path.abspath(__file__):   # this runner is not an input of the result
            deps.add(os.path.abspath(f))
    here = os.getcwd()
    skip = [os.path.join(here, d) + os.sep for d in ("results", "logs", "deps")]
    for f in opened:
        if f.startswith(TOP + os.sep) and not any(f.startswith(s) for s in skip) \
                and "__pycache__" not in f and os.path.basename(f) != "initialize.tex" and os.path.exists(f):
            deps.add(f)
    outs = sorted(p for p, v in _snapshot(os.getcwd()).items() if _before.get(p) != v)
    outs = [os.path.relpath(p, TOP) for p in outs if not os.path.islink(p)]
    rel = sorted(r for r in (os.path.relpath(d, TOP) for d in deps) if r not in outs)   # a run's own outputs are not its inputs
    stamps = sorted(os.path.join(os.path.relpath(here, TOP), "params", n + ".stamp") for n in params.ACCESSED)
    allreq = rel + stamps
    try:   # one line per run: stage, start, end, seconds, exit status (read by "make times")
        _t1 = time.time()
        _stage = os.path.splitext(os.path.basename(target))[0]
        _fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
        os.makedirs("logs", exist_ok=True)
        with _orig_open("logs/times.tsv", "a") as _f:
            _f.write(f"{_stage}\t{_fmt(_t0)}\t{_fmt(_t1)}\t{_t1 - _t0:.0f}\t{130 if interrupted and status == 0 else status}\n")
    except OSError:
        pass
    if status == 0 and not interrupted and target.endswith(".done"):
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w") as f:
            f.write("done\n")
    os.makedirs(os.path.dirname(depfile) or ".", exist_ok=True)
    with open(depfile, "w") as f:
        f.write(f"{os.path.relpath(os.path.abspath(target), TOP)}: " + " ".join(allreq) + "\n")
        for o in outs:              # the files this run made depend on the run
            f.write(f"{o}: {os.path.relpath(os.path.abspath(target), TOP)}\n")
        for d in allreq:            # empty rules: a file that has gone away does not stop make
            f.write(f"{d}:\n")
sys.exit(130 if interrupted and status == 0 else status)   # a stop is never reported as success
