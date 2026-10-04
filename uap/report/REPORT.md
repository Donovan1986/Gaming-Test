# UAP empirical-signature investigation: interim status (2026-10-04)

**Status: DISCOVERY PHASE COMPLETE; VALIDATION NOT YET RUN UNDER FINAL CODE.**
Nothing below has been tested on held-out data under the final (merged, inference v4) code.
Every discovery-phase number is provisional and comes from the superseded v3 run
(`results/superseded_v3_pre_merge/`). No confidence grade above "unvalidated" is assigned yet.
The archived handoff report (`archive/REPORT_blocked_rerun_handoff.md`) described a different
environment whose raw inputs were missing. In this environment the input preflight passes and
the pipeline runs.

## Bottom line so far

**No pattern has been found that "refuses to disappear."** Every strong discovery-phase signal
has a conventional explanation:

- **Observing conditions:** clear sky, no rain, moonlight.
- **Known objects:** launches, bolides, ISS, Venus, holidays with fireworks and lanterns.
- **Geography of who reports and where:** population and urbanisation gradients, plus NUFORC's
  home region.

These effects are just as strong, or stronger, in reports the pipeline classifies as
*explained* as in *unexplained* ones. That is the signature of a reporting and visibility
mechanism, not of a distinct phenomenon. The space-weather, seismic and nuclear-facility
hypotheses produced no signal that survives the robust designs. This is provisional until the
locked holdouts are evaluated.

## 1. Does the pipeline find things that are really there? (positive controls)

There were 38 positive-control tests (`results/positive_controls.csv`):

| Outcome | Count |
|---|---|
| Passed | 28 |
| Inconclusive | 4 |
| Not testable | 4 |
| Failed | 1 |
| Informational | 1 |

The pipeline recovers known relationships with large, correctly signed effects. Odds ratios
are against matched controls: CS1 is the same place, same month, weekday and local time; CS2 is
the same place ±1 or ±2 years.

| Known stimulus | Report subset | OR (CS1 / CS2) |
|---|---|---|
| CNEOS bolide within 30 min and 1000 km | all | 28.7 / 67.3 |
| Meteor outburst ±1 d (dark sky) | meteor-like | 3.8 / 2.8 |
| Rocket launch 0–3 h before, ≤1500 km | all | 1.68 / 1.68 |
| Launch during observer twilight | all | 3.2 / 3.2 |
| July 4th | orange/fireball | 7.8 |
| New Year | orange/fireball | 5.3 |
| Venus conspicuous | planet-like | 1.6 / 2.8 |
| ISS visible pass ±10 min | satellite-like | 1.56 / 1.79 |
| Moon above horizon | moon-word | 2.5 / 3.2 |
| Clear sky (≤2 oktas) | all | 1.68 / 1.65 |

- **Failed:** "aircraft-like reports more common within 25 km of a large airport." The spatial
  OR was 0.72, meaning fewer reports near large airports. Non-aircraft reports showed the same
  pattern (OR 0.61), and the ratio between the two was 1.18 (NS). The lesson: spatial
  (place-vs-place) designs are dominated by urban-core vs suburban/rural sky-view and reporting
  gradients, so every spatial result needs the urbanisation adjudication (A1).
- **Not testable:** Starlink, because there is no Starlink-era data in the discovery split.
  This is re-tested after the freeze.
- **Calibration:** placebo catalogues shifted by whole years, plus Gaussian-noise exposures,
  give false-positive rates of 0–7.7% at α=0.05, with Clopper–Pearson CIs covering 5%
  (`results/placebo_calibration_summary.csv`). Two-way cluster-robust SEs were needed for this:
  naive SEs gave 15–19%.

## 2. Discovery-phase patterns (provisional, v3, NUFORC discovery split 1995–2015)

| Theme | What was seen | Leading explanation |
|---|---|---|
| Weather (largest, most consistent) | Clear sky OR ≈1.7. Overcast ≈0.55. Any precipitation ≈0.3–0.5. Calm clear night (inversion proxy) ≈1.4. Per okta of cloud 0.917. Clearing within 6 h ≈1.25. Fronts/pressure falls ≈0.7. The same in ALL / HQ / UNEXPLAINED / EXPLAINED / multi-sensor subsets. | Observation opportunity: you can only report what you can see. Same-mechanism in explained reports. |
| Known objects | Launches, bolides, ISS, Venus and holidays, as above. A global launch within ±30 min and 500 km gave OR 7.3. | Conventional stimuli: WELL KNOWN. |
| Spatial clustering | County rates are strongly autocorrelated: Moran's I 0.36 on population only, 0.08 after density and latitude adjustment. The top Kulldorff cluster is western Washington and Oregon, RR 2.7, still significant under overdispersed Monte Carlo (p=0.008). | NUFORC is headquartered in Washington State: a home-region reporting artifact. Most other clusters lose significance under the overdispersed null. |
| Space–time clustering | Knox ratios: 1.48 at 10 km / 1 day, 1.19 at 25 km / 7 d, 1.08 at 50 km / 30 d. Larger in the HQ subset (3.7 at 10 km / 1 d). | Multiple witnesses of one object, local flaps and media waves. Expected. |
| Military, airport and airspace proximity | Many CS4 (place-vs-place) associations of mixed sign. Closer to DoD sites and large airports gives fewer reports; within 500 km of military operations areas gives more. Equally present in EXPLAINED reports. | Urbanisation and land-use gradient; same-mechanism. To be adjudicated (A1, A2). |
| Space weather | 228 tests. 7 reach q<0.05, all F10.7 solar flux under CS2 only, OR 1.03–1.07 per 10 sfu. Nothing under CS1. Kp, Dst, Bz and high-speed streams are null in discovery. | CS2 compares years ~1 solar-cycle phase apart, so it is confounded with NUFORC's multi-year growth. It was not promoted. A cross-system detrended test is scheduled. |
| Seismic | 543 tests, 2 at q<0.05: an M4+ quake within 250 km the prior day, OR 1.29 (ALL, CS1) and 1.97 (EXPLAINED, CS2). | Weak, inconsistent across designs, stronger in *explained* reports. |
| Nuclear / ICBM / DOE | Within 500 km of a nuclear test site, OR ≈0.52 (fewer reports): effectively "is in the Great Basin". Natural experiments (DiD): ICBM field deactivation IRR 1.66 (95% CI 1.00–2.77; bootstrap 0.85–3.55; 106 events); nuclear-plant shutdown IRR 0.79–0.90 (NS). | No robust signal. The ICBM result is borderline and rests on 106 events. It is a frozen secondary hypothesis for replication. |

## 3. What the frozen validation plan contains

The (superseded) v3 freeze held:

- 130 primary candidates:
  - 124 rule-promoted;
  - 5 ML interactions;
  - 1 ordered sequence (substorm → convective storm).
- 13 secondary, literature-driven hypotheses: moonlight; Kp ≥ 5; Dst ≤ −50; southward Bz;
  high-speed streams; F10.7 top decile; M4 quakes in the prior 7 d; nuclear plants; ICBM fields;
  DOE weapons sites.
- Pre-declared adjudication tests: urbanisation adjustment, same-mechanism ratio of ORs,
  solar cycle across 5 independent reporting systems, Knox and scan replication on holdouts,
  and DiD replication.

The final freeze is regenerated from the v4 discovery run.

Each candidate is then evaluated on:

- validation (E1), the locked random holdout (E2) and the 2016+ temporal holdout (E3);
- France GEIPAN (E4) and non-North-American NUFORC (E5);
- Hatch (E6), Blue Book unknowns (E7) and NICAP (E8);
- leave-one-source-out;
- a sensitivity grid;
- structure-preserving permutations;
- Holm correction across all validation tests;
- a rolling-origin prospective backtest on 2016–2023, with coefficients learned only from
  earlier years.

## 4. Changes in this pass

- **Runner bug fixed:** after the merge, the freeze stage had stopped calling
  `select_candidates.main()` before `build_frozen.main()`. The full run would have crashed hours
  in, at the freeze, so that run was stopped 3 minutes in. Fixed in `src/run_investigation.py`.
- **Backtest rewritten and added as a journaled stage after validation** (`src/backtest.py`):
  - test years are restricted to 2016–2023, the only selection-clean years;
  - it reuses the frozen subset and candidate materialisation;
  - it applies the frozen scale;
  - it computes a bootstrap CI over test years;
  - it registers its results.
- The 99 unit tests pass.

## 5. Next passes (in order)

1. Run `python3 src/run_investigation.py --python "$(which python3)" --feature-chunk 150000 --permutations 30`.
   It takes hours. The journal is under `logs/` and status is in `results/investigation_status.json`.
2. Before that, finish the per-stage pre-flight code review (interrupted in this pass): check
   every declared stage output is written and the post-freeze code paths don't crash.
3. After validation:
   - generate grades A–F mechanically (METHODS §11);
   - build `results/patterns_table.csv` with the required columns, plus the top-10 cards and
     the TOP RESULT section;
   - run an adversarial "try to explain it away" review of the top candidates;
   - attach literature novelty labels from `results/literature_review.json`.

The `results/patterns_table.csv` currently in the repo is a placeholder from the handoff and has
no validated numbers.

## Data and reproducibility

- Sources and retrieval: `report/DATA_SOURCES.md`, `data/manifest.jsonl` (URL, timestamp, SHA-256).
- Methods: `report/METHODS.md`.
- Every test ever run: `results/hypothesis_registry.csv`.
- Master seed: 20261004.
- NUFORC raw text and per-record data are not redistributed, per NUFORC terms.
