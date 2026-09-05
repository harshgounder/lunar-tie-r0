# BRIEF-R0-AUDIT-20260905.md

## GOAL
Read-only hostile audit of the lunar-tie-r0 pipeline code. You write NOTHING.
You produce ONE report file at the end: AUDIT-MUSE-20260905.md in this repo root.

## SCOPE (read these fully)
- src/lunar_tie/: pipeline.py, consensus.py, subpixel.py, masking.py, detect.py, photometric.py, coverage.py, conformal.py, tiling.py, pds3label.py
- tests/ (skim: what the tests actually assert vs what the modules claim)

## HUNT LIST (priority order)
1. REALITY-COUPLING HOLES: every comparison in the live path, ask "compared against WHAT, is that satisfiable by construction?" (known class: a drift gate comparing fixed-qty vs derived-target = unsatisfiable; a log comparing a fill to itself = vacuous).
2. SILENT-JUNK PATHS: any place a numerical routine (LAPACK/numpy) can return garbage with only a C-level warning, or an exception is caught and a fallback silently substitutes junk. Past catches here: NaN 2-pt hypothesis returned junk matrix behind a green suite; Otsu returning None on flat images.
3. DOC-CLAIM DRIFT: README.md + ARCHITECTURE.md + MILESTONE-1-CLOSEOUT.md + PROGRESS-LOG.md + the unit ticket .md files in the root. Cross-check every capability claim ("handles X", "proven Y", "supports Z") against the actual code. Specifically reconcile: README "honesty layer" says MINI-GATE-2 rotate+scale is NOT yet green, but PROGRESS-LOG 2026-08-30 rows say MINI-GATE-2 PASSED with s=1.2001 theta=0.2000. Determine which is true from tests/test_minigate2.py (or its equivalent) and report exactly what the fixture tests.
4. CONTRACT LIES: docstrings or comments promising behavior the code does not enforce (losslessness, invariance bounds, loud-vs-silent error policy, empty-input safety).
5. BOUNDARY BUGS: off-by-one in tiling grids, mask morphology edges, quadrant assignment at cell boundaries, percentile/order-statistic indexing in conformal (q_hat = ceil((n+1)(1-alpha))/n), phase-correlation parabola peak near array edges.

## RULES
- READ-ONLY except: create AUDIT-MUSE-20260905.md at repo root at the END.
- Do NOT git commit, do NOT modify any existing file, do NOT run network calls.
- You MAY run the test suite (.venv/bin/pytest tests/ -q) and small python snippets to verify a suspicion. Never modify files to make a test pass.
- No em dashes in your report. No AI-tell words (delve, leverage, robust, furthermore, moreover, notably, significantly, essentially, comprehensive, seamless). Plain factual sentences.
- Every finding: file, line number, what the code does, what it should do, severity (P0 blocker / P1 real bug / P2 smell / P3 doc), and a 3-line repro or evidence snippet you actually ran or read.
- If a suspected finding turns out fine when you verify it, do not list it. Verified findings only.
- End the report with a VERIFIED-OK section: things you attacked that held up (so the absence of findings is evidence of checking, not of skipping).
