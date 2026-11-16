"""Read parameter macros from a TeX file (default build/initialize.tex) into a Python dict.

Understands   \\def\\Fval{0.018}   and   \\newcommand{\\Fval}{0.018}   (also \\renewcommand, \\providecommand).
Values that look like integers or floats become int or float; anything else stays a string.
Lines starting with % are comments; a % after a value also starts a comment.

    from params import load
    P = load()
    F, k = P["Fval"], P["kvak"]
"""
import re
import sys

_DEF = re.compile(r"\\def\s*\\([A-Za-z]+)\s*\{([^}]*)\}")
_NEW = re.compile(r"\\(?:re|provide)?newcommand\s*\{\\([A-Za-z]+)\}\s*\{([^}]*)\}")


def _convert(s):
    s = s.strip()
    for typ in (int, float):
        try:
            return typ(s)
        except ValueError:
            pass
    return s


# Names of the parameters a run has read. trace_run.py saves this list, so the Makefile knows which numbers each
# result depends on, and a change to a number reruns only the results that read it.
ACCESSED = set()


class Params(dict):
    """A dict that remembers which keys were read."""
    def __getitem__(self, key):
        ACCESSED.add(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in self:
            ACCESSED.add(key)
        return dict.get(self, key, default)


def load(path="build/initialize.tex"):
    out = Params()
    with open(path) as f:
        for line in f:
            line = line.split("%", 1)[0]
            for rx in (_DEF, _NEW):
                for name, val in rx.findall(line):
                    out[name] = _convert(val)
    return out


if __name__ == "__main__":
    for k, v in load(*sys.argv[1:2]).items():
        print(f"{k} = {v!r}")
