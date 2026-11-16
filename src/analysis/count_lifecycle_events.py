"""
Count completed lifecycles, divisions and deaths in a lifecycle run.

Paper (main text, life-cycle comparison; ESM, lineage tracking): merged organisms give 695 confirmed divisions and 771 deaths
(1466 completed lifecycles) over 10,000 time units; tracking raw watershed regions gave 1000 divisions and 1956 lifecycles.

Usage:  python3 count_lifecycle_events.py                               # lifecycle_progress.pkl (merged organisms)
        python3 count_lifecycle_events.py lifecycle_raw_progress.pkl    # from  python3 run_lifecycle.py --n-steps 10000 --fast --raw
"""
import sys
from run_lifecycle import LifecycleTracker

path = sys.argv[1] if len(sys.argv) > 1 else "lifecycle_progress.pkl"
tr = LifecycleTracker.load(path)
causes = [e["cause"] for e in tr.examples]
print(f"{path}: t = {tr.t}; completed lifecycles {len(causes)}; divisions {causes.count('split')}; deaths {causes.count('death')}")
