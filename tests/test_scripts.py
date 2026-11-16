"""Quick checks on ALL the scripts, with no simulation and no data (about 15 seconds).  They catch the cheap failures
that would otherwise show up only ten minutes into make QUICK=1:

  1. every script named in Makefile or stages.mk exists;
  2. every script compiles (no syntax errors);
  3. every script that is safe to import imports (no missing package, no bad import);
  4. every script that uses argparse answers --help (the argument parser builds);
  5. no two scripts write the same LaTeX macro name (LaTeX stops on a duplicate \\newcommand).

A script that needs data at import, or that runs code when imported, is skipped in check 3 and listed in the message.
Check 4 runs each script in a temporary folder that holds a copy of src/TeX/initialize_quick.tex as initialize.tex,
which is what a stage finds in its working folder.

    make test        (or, from the project root:  python -m pytest tests/test_scripts.py)
"""
import ast
import collections
import glob
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "src", "*", "*.py")))
rel = lambda p: os.path.relpath(p, ROOT)


def parse(path):
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def has_main_guard(tree):
    for n in tree.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and getattr(n.test.left, "id", "") == "__name__":
            return True
    return False


def top_level_runs_code(tree):
    """True if importing the module would run something other than definitions: a bare call, a loop, a with block."""
    for i, n in enumerate(tree.body):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue  # docstring
        if isinstance(n, (ast.Expr, ast.For, ast.While, ast.With)):
            return True
    return False


# 1. the scripts named in the makefiles exist -------------------------------------------------------------------

def test_every_script_named_in_the_makefiles_exists():
    text = ""
    for name in ("Makefile", "stages.mk"):
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            text += open(path).read()
    assert text, "no Makefile found next to tests/"
    named = re.findall(r"^(?:SIMSCRIPT|SCRIPT)\.[A-Za-z0-9_]+\s*:?=\s*(\S+)", text, re.M)
    named += re.findall(r"src/[a-z]+/([A-Za-z0-9_]+\.py)", text)
    on_disk = {os.path.basename(p) for p in SCRIPTS}
    missing = sorted({n for n in named if n.endswith(".py") and n not in on_disk})
    assert named, "found no script names in the makefiles (has their format changed?)"
    assert not missing, f"named in Makefile or stages.mk but not in src/: {missing}"


# 2. every script compiles -------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", SCRIPTS, ids=rel)
def test_script_compiles(path):
    parse(path)


# 3. the scripts that are safe to import import --------------------------------------------------------------

IMPORTABLE = [p for p in SCRIPTS if not top_level_runs_code(parse(p))]


@pytest.mark.parametrize("path", IMPORTABLE, ids=rel)
def test_script_imports(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        importlib.import_module(name)
    except (FileNotFoundError, SystemExit) as err:
        pytest.skip(f"needs data or arguments at import: {err!r}")


# 4. the argument parsers build -------------------------------------------------------------------------------

def test_scripts_with_argparse_answer_help():
    tmp = tempfile.mkdtemp()
    try:
        init = os.path.join(ROOT, "src", "TeX", "initialize_quick.tex")
        if not os.path.exists(init):
            pytest.skip("src/TeX/initialize_quick.tex not found")
        shutil.copy(init, os.path.join(tmp, "initialize.tex"))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([ROOT] + sorted(glob.glob(os.path.join(ROOT, "src", "*"))))
        env["MPLBACKEND"] = "Agg"
        env["TOP"] = ROOT
        todo = []
        for p in SCRIPTS:
            text = open(p).read()
            if "argparse" in text and has_main_guard(ast.parse(text)):
                todo.append(p)

        def run(p):
            r = subprocess.run([sys.executable, "-B", p, "--help"], capture_output=True, text=True, timeout=180,
                               env=env, cwd=tmp)
            out = r.stdout + r.stderr
            ok = r.returncode == 0 and "usage" in out.lower()
            return p, ok, (out.strip().splitlines() or [""])[-1][:150]

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
            results = list(pool.map(run, todo))
        bad = [f"{rel(p)}: {msg}" for p, ok, msg in results if not ok]
        assert todo, "found no scripts with argparse"
        assert not bad, "these scripts do not answer --help:\n  " + "\n  ".join(bad)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# 5. no two scripts write the same macro name -----------------------------------------------------------------

NAME_OK = re.compile(r"^[A-Za-z]+$")


def literal_pairs(node):
    return [n.elts[0].value for n in ast.walk(node)
            if isinstance(n, ast.Tuple) and len(n.elts) == 2
            and isinstance(n.elts[0], ast.Constant) and isinstance(n.elts[0].value, str)]


def macro_names(path):
    """The literal macro names a script passes to write_macros(path, script, items): items written out in the call, or
    a list variable that is assigned or appended to.  Names built with f-strings are not seen."""
    tree = parse(path)
    names, variables = [], set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            fn = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if fn == "write_macros" and len(n.args) >= 3:
                arg = n.args[2]
                if isinstance(arg, ast.Name):
                    variables.add(arg.id)
                else:
                    names += literal_pairs(arg)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in variables for t in n.targets):
            names += literal_pairs(n.value)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("append", "extend")
                and isinstance(n.func.value, ast.Name) and n.func.value.id in variables):
            for a in n.args:
                names += literal_pairs(a)
    return {x for x in names if NAME_OK.match(x)}


def test_no_two_scripts_write_the_same_macro_name():
    owners = collections.defaultdict(set)
    for p in SCRIPTS:
        for name in macro_names(p):
            owners[name].add(rel(p))
    assert owners, "found no macro names (has write_macros changed?)"
    dup = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
    assert not dup, f"macro names written by more than one script: {dup}"
