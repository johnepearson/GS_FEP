"""Is the burn-in long enough?  Run from build/.  Writes results/check_burnin.tex.

The burn-in has to last until two noisy copies of the same settled field have separated, so that realisations of an
ensemble no longer depend on the field they started from.  run_divergence_rate.py measures the step at which two such
copies first reach 90% of the separation of two independent draws (t90).  The rule here is
        burn-in >= 2 x (mean t90 over the reference states).
Macros:  DivSeparation  mean t90 (steps);  BurninMinimum  twice that.
Prints a WARNING for every burn-in parameter below the minimum; it does not stop the build."""
import os
import numpy as np
from params import load
from texout import num, write_macros

P = load("initialize.tex")
names = ("BurninOne", "BurninTwo", "BurninThree", "ThreeVarBurnin")
if not os.path.exists("divergence_rate_results.npz"):
    raise SystemExit("divergence_rate_results.npz not found: run the divergence stage first")
D = np.load("divergence_rate_results.npz")
t90 = D["t90"]
ok = t90[t90 > 0]
rates, bench = D["rates"], D["bench"]
WORDS = ("One", "Two", "Three")
DIV = []
for j, w in enumerate(WORDS):
    DIV += [(f"DivRate{w}", num(rates[j], 5) if j < len(rates) else "nan"),
            (f"DivBench{w}", num(bench[j], 4) if j < len(bench) else "nan"),
            (f"DivTNinety{w}", num(t90[j], 0) if j < len(t90) and t90[j] > 0 else "nan")]
DIV += [("DivRateMin", num(rates.min(), 5)), ("DivRateMax", num(rates.max(), 5)),
        ("DivRateSDPercent", num(100 * rates.std() / rates.mean(), 0)),
        ("DivRateRangePercent", num(100 * (rates.max() - rates.min()) / rates.mean(), 0))]
if len(ok):
    DIV += [("DivTNinetyMin", num(ok.min(), 0)), ("DivTNinetyMax", num(ok.max(), 0)),
            ("DivTNinetyRangePercent", num(100 * (ok.max() - ok.min()) / ok.mean(), 0))]
else:
    DIV += [("DivTNinetyMin", "nan"), ("DivTNinetyMax", "nan"), ("DivTNinetyRangePercent", "nan")]
DIV += [("DivSpan", num(float(D["times"].max() - D["times"].min()), 0)), ("DivBenchMin", num(bench.min(), 4)), ("DivBenchMax", num(bench.max(), 4))]
if len(ok) == 0:
    print("separation time not reached within the branch length: nothing to compare (test run?)")
    write_macros("results/check_burnin.tex", "check_burnin.py", DIV + [("DivSeparation", "nan"), ("BurninMinimum", "nan")])
else:
    sep = float(ok.mean())
    need = 2.0 * sep
    print(f"copies reach 90% of the independent-draw separation at {', '.join(str(int(t)) for t in ok)} steps "
          f"(mean {sep:.0f});  minimum burn-in 2 x mean = {need:.0f}")
    for n in names:
        status = "ok" if P[n] >= need else "WARNING: below the minimum"
        print(f"  {n} = {P[n]}  {status}")
    write_macros("results/check_burnin.tex", "check_burnin.py",
                 DIV + [("DivSeparation", num(sep, 0)), ("BurninMinimum", num(need, 0))])
