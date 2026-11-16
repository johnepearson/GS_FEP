"""One small file per parameter: build/params/<Name>.stamp holds that parameter's value. A file is rewritten only when
its value changes, so its timestamp changes only then.  The Makefile lists these stamps as the prerequisites of each
result (see trace_run.py), which is how "change a number, run make" reruns only the results that use that number.

Usage:  python3 src/tools/paramstamps.py build/initialize.tex build/params      (the Makefile does this every time it starts)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import params  # noqa: E402

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
for name, value in dict.items(params.load(src)):
    path = os.path.join(outdir, name + ".stamp")
    text = repr(value) + "\n"
    if not os.path.exists(path) or open(path).read() != text:
        with open(path, "w") as f:
            f.write(text)
