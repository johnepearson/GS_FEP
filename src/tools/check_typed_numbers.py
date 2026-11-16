"""
List the numbers still typed by hand in the paper and supplement, and flag the ones that look like stale results.

For every number in src/TeX/paper.tex and supplement.tex that is NOT produced by a macro (\\rr, \\nn, \\num) the script
decides which kind it is, and prints a report:

  PARAM   equals a parameter in initialize.tex (e.g. a typed 7000 that should be \\BurninOne)
  MACRO   equals a generated value, to the number of digits typed (probably a result that should be a macro)
  NEAR    within 15% of a generated value of the same sign (a possibly stale result; check)
  OTHER   anything else with at least two significant digits (method constants, one-off results, timings, ...)

Skipped as structure: grid sizes (1024x1024, 256^2), exponents of ten, LaTeX lengths, labels/refs/citations, file names,
\\texttt, section numbers, years, confidence levels (95%), and the digits 0-9 on their own.

Usage (from ANDY_STYLE):   python3 src/tools/check_typed_numbers.py build [build_medium]
  first folder: initialize.tex (parameters) and where typed_numbers.txt is written; optional second folder: the
  results.tex to compare against (default: the first folder).  Run it on build/ after the full run.
"""
import re
import sys
import os

B = sys.argv[1] if len(sys.argv) > 1 else "build"
R = sys.argv[2] if len(sys.argv) > 2 else B      # folder whose results.tex supplies the generated values
TEX = ["src/TeX/paper.tex", "src/TeX/supplement.tex"]
SKIP_VALUES = {"95", "100", "10", "2024", "2026", "1993", "2021", "2022", "2023", "2018", "2019", "2020"}


def numval(s):
    s = s.replace("{,}", "").replace(",", "").replace("\u2212", "-")
    try:
        return float(s)
    except ValueError:
        return None


def digits(s):
    s = s.replace("{,}", "").replace(",", "").lstrip("-")
    return len(s.split(".")[1]) if "." in s else 0


def read_macros(path):
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", open(path).read()):
        v = numval(m.group(2))
        if v is not None:
            out[m.group(1)] = v
    return out


params = read_macros(os.path.join(B, "initialize.tex"))
results = read_macros(os.path.join(R, "results.tex")) if os.path.exists(os.path.join(R, "results.tex")) else {}

STRIP = [
    (r"(?m)(?<!\\)%.*$", " "),
    (r"\\begin\{verbatim\}.*?\\end\{verbatim\}", " "),
    (r"\\(rr)\{[^}]*\}\{\\[A-Za-z]+\}", " @ "),
    (r"\\(nn|num)\{\\[A-Za-z]+\}", " @ "),
    (r"\\(label|ref|eqref|cite[a-z]*|citet|citeauthor|citeyear|includegraphics|input|url|texttt|bibliography[a-z]*|"
     r"bibliographystyle|S|footnote|usepackage|documentclass|setlength|hspace|vspace|begin|end)(\[[^\]]*\])?\{[^{}]*\}", " "),
    (r"\\the\\numexpr[^\\]*\\relax", " @ "),
    (r"\d+\s*\\times\s*\d+", " "),
    (r"\d+\^\{?\d+\}?", " "),
    (r"\\times\s*10\^\{[^}]*\}", " "),
    (r"10\^\{[^}]*\}", " "),
    (r"[0-9.]+\s*(cm|mm|pt|em|ex)\b", " "),
    (r"[0-9.]+\\(textwidth|linewidth|columnwidth)", " "),
    (r"p\{[^}]*\}", " "),
    (r"\\[A-Za-z]+\s*", " "),
]


def clean(t):
    t = t.split("\\begin{document}")[-1]
    for pat, rep in STRIP:
        t = re.sub(pat, rep, t, flags=re.S)
    return t


NUM = re.compile(r"(?<![A-Za-z\\_@])[-\u2212]?\d[\d,]*(?:\{,\}\d{3})*(?:\.\d+)?")
rows = []
for f in TEX:
    raw = open(f).read()
    body = raw.split("\\begin{document}")[-1]
    start_line = raw[:raw.index("\\begin{document}")].count("\n") if "\\begin{document}" in raw else 0
    lines = body.split("\n")
    for k, line in enumerate(lines):
        c = clean(line)
        for m in NUM.finditer(c):
            s = m.group(0)
            v = numval(s)
            if v is None or s.lstrip("-\u2212") in SKIP_VALUES:
                continue
            sig = len(re.sub(r"\D", "", s).lstrip("0"))
            if sig < 2 and "." not in s:
                continue
            ctx = c[max(0, m.start() - 45):m.end() + 30].strip()
            kind, hit = "OTHER", ""
            for n, pv in params.items():
                if pv == v and abs(v) >= 100:
                    kind, hit = "PARAM", "\\" + n
                    break
            if kind == "OTHER" and abs(v) > 0 and sig >= 3:
                d = digits(s)
                for n, rv in results.items():
                    if abs(round(rv, d) - round(v, d)) < 1e-12 and abs(v) >= 0.01:
                        kind, hit = "MACRO", "\\" + n
                        break
                if kind == "OTHER":
                    for n, rv in results.items():
                        if rv != 0 and abs(v) >= 0.01 and rv * v > 0 and abs(rv - v) / abs(v) < 0.15 and digits(s) >= 2:
                            kind, hit = "NEAR", f"\\{n}={rv:g}"
                            break
            rows.append((kind, f, start_line + k + 1, s, hit, ctx))

order = {"PARAM": 0, "MACRO": 1, "NEAR": 2, "OTHER": 3}
rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))
out = os.path.join(B, "typed_numbers.txt")
with open(out, "w") as fh:
    for kind, f, ln, s, hit, ctx in rows:
        fh.write(f"{kind:5s} {os.path.basename(f)}:{ln:<5d} {s:>10s}  {hit:28s} | {ctx}\n")
cnt = {k: sum(1 for r in rows if r[0] == k) for k in order}
print(cnt, "total", len(rows), "->", out)
