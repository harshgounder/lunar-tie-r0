# UNIT-9 TICKET - conformal ABSTAIN + metrics panel (D10 12.x / P8.m3 + P9.m4)

Assignment for opencode. Final M1 module. Implement `src/lunar_tie/conformal.py`
exactly to this spec.

## context
The pipeline must say "I do not know" loudly when it cannot prove a tie. R0 unit:
split-conformal calibration on per-tie residuals -> per-tie ACCEPT/REJECT/ABSTAIN.
Plus the metrics panel exporter that stamps every number with a truth-tier label
(the honesty contract IS the product).

## requirements (exact, no scope creep)
1. function conformal_calibrate(residuals_cal, alpha=0.05) -> q_hat float:
   split-conformal quantile = ceil((n+1)(1-alpha))/n empirical quantile of the
   calibration residuals (finite-sample exact); residuals_cal are error px of
   held-out ties; raise ValueError if empty or non-finite
2. function conformal_gate(scores_test, q_hat, sigma_thr=None) -> labels ndarray
   of 'ACCEPT'/'REJECT': ACCEPT iff score <= q_hat (and <= sigma_thr when
   provided); else REJECT
3. function three_way_gate(scores_test, scores_cal outliers_flag_cal, alpha=0.05,
   abstain_penalty=1.5) -> labels where CANDIDATE-REJECTED pairs whose residual
   exceeded q_hat * abstain_penalty are REJECT, and the band (q_hat, q_hat *
   abstain_penalty] is ABSTAIN (neither trusted nor discarded: flagged for
   review). Implement exactly this ladder; document it in the docstring.
4. function metrics_panel(residuals, inliers, q_hat, tier_label, extra=None)
   -> JSON-serializable dict:
   {'n_pairs', 'n_valid', 'inlier_ratio', 'rms_px', 'rms_subpx',
    'q_hat', 'ece_proxy', 'coverage_gates': {...}, 'tier_label',
    'gate_name', 'provenance': {'date', 'git_sha', 'seed'}}
   - ece_proxy: |mean(predicted_confusion)|implified: |inlier_ratio -
     (1 - alpha)| honest proxy, DO NOT overseel: docstring labels it a proxy
   - tier_label from {1,2,3,4,5} per brief; NO tier = raise ValueError
5. function export_panel(panel, path) writes JSON with indent=2; function
   load_panel(path) -> dict; round-trip exact
6. unit gate (tests/test_conformal.py): 3 in-test fixtures:
   (a) residuals_cal ~ N(0.1, 0.05)^400: q_hat statistically near the 95th
       percentile (|q_hat - q95_empirical| <= 0.02); coverage on fresh scores:
       >= 93% of 2000 fresh draws <= q_hat (validity >= 0.93)
   (b) three-way: scores deliberately [0.01, 0.05, 0.09, 0.11, 0.5] with sane
       params produce exactly labels [ABSTAIN? verify] - assert exact strings
       for each band edge case (write the expected list in the test)
   (c) panel export/load round-trip preserves every field including nested
       provenance dict; missing tier_label raises ValueError
7. expose summarize_zoo_candidate(panel) -> dict: the exact subset of fields a
   monster-zoo ledger row needs (stratum_key placeholder + config placeholder +
   gate_evidence from panel; honest PLACEHOLDERS, no invention).

## out of scope
- full ECE via binning (needs predicted probability model; proxy only in R0)
- online/adaptive conformal (B4 wave leads; later)
- per-bin calibration across strata (that is ZOO router milestone 3+)

## constraints
- numpy + stdlib only
- do NOT modify existing modules (8 exist)
- < 250 lines, docstring unit id + gate
- no em-dash, no AI-tell vocabulary
- ambiguity: scores_test empty -> return empty labels array, no crash

## definition of done
- tests/test_conformal.py: >= 12 asserts, 3 fixtures, pass
- gate name: 'conformal gate on 3 synthetic fixtures'
- one PROGRESS-LOG row
- no git commit (Hermes audits + commits)