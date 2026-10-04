# Methods

This document describes exactly what was computed. Code paths are relative to `uap/src/`. Every
statistical test, including nulls, failures and superseded runs, is in `results/hypothesis_registry.csv`
(column `inference_version` marks superseded rows; only `v3_*` rows are used for conclusions).

## 1. Data and unified schema

Sources, URLs, retrieval dates and SHA-256 hashes: `report/DATA_SOURCES.md`, `data/manifest.jsonl`.
An independent check (`independent_check/`) re-hashed all 9,907 retrieved files (9,907 matches).

`build_reports.py` maps every source to one REPORT-level table with the requested fields (EVENT_ID,
SOURCE, SOURCE_RECORD_ID, ORIGINAL_TEXT, DATE, LOCAL_TIME, UTC_TIME, TIMEZONE, TIME_UNCERTAINTY,
LATITUDE/LONGITUDE, LOCATION_UNCERTAINTY_RADIUS, ALTITUDE(+status), OBSERVATION_DURATION, NUMBER_OF_WITNESSES
(+basis), WITNESS_TYPE, OBSERVATION_TYPE, SHAPE, COLOR, MOTION_DESCRIPTION, ANGULAR_SIZE/VELOCITY (+status),
DIRECTION_OF_TRAVEL, ELEVATION_AZIMUTH (+status), RADAR_INVOLVEMENT, PHOTOGRAPHIC/VIDEO/MULTI-SENSOR
EVIDENCE, REPORTED EM / PHYSIOLOGICAL EFFECTS, CASE_QUALITY_SCORE, EXPLANATION_STATUS, KNOWN_OBJECT_MATCHES,
SOURCE_URL, RETRIEVAL_DATE, RAW_RECORD_HASH). Environmental variables are kept in separate tables keyed by
event (report_context, points).

Uncertainty conventions (never false precision):
* Time: 5/10/15/30 min by clock-time rounding; NUFORC local 00:00 = 720 min (scraper default for blank
  times); vague words ("evening", "daytime") stay as TIME_TEXT and are never converted to a clock time;
  no time -> TIME_PRECISION="none" and the event is excluded from time-of-day analyses.
* Date precision: day / month / year (+approx). Two impossible calendar dates were downgraded to month.
* Location: gazetteer matching (`geo.py`) returns a radius >= 5 km (town) growing with place size; "N mi
  DIR of X" is offset with radius max(10 km, 0.3*offset); military-base names match DoD MIRTA before town
  names; state/country-only locations are NOT geocoded (precision="region", excluded from spatial work).
  Every decision is in `results/geocoding_decisions.csv`.
* Missing data codes: YES_STRUCTURED / YES_TEXT / NOT_REPORTED / DATA_UNAVAILABLE / NOT_ESTIMABLE. Absence of
  a mention is NOT_REPORTED, never "no". Absent covariate data (no weather station within 50 km, outside a
  catalog's coverage) is NaN, never zero.

Text-derived fields carry the suffix `_txt`; English and French regexes include negation handling
("not a plane", "faster than any jet", "ce n'était pas un avion").

## 2. Deduplication (`build_events.py::cluster`)

Union-find over report pairs. LIKELY_SAME_EVENT_CLUSTER: both UTC times known and |dt| <= max(60 min, sum
of time uncertainties; cap 3 h) and distance <= max(50 km, r1+r2); or, for date-only records, same local
date and distance <= max(100 km, r1+r2). REGIONAL_CLUSTER (widely visible stimuli): |dt| <= 2 h and <= 400
km. Report- and event-level tables are both kept (REPORT_COUNT, INDEPENDENT_SOURCE_COUNT). 123,686 reports
-> 115,184 events.

## 3. Case quality score (`results/quality_score_formula.md`)

Transparent additive score, clipped to [0, 10]; narrative strangeness is never used. INFORMATION_QUALITY
(precision of time/date/location, duration, witnesses, trained observers, sensors, contemporaneity, source
type, hoax flags) and CASE_QUALITY_SCORE = INFORMATION_QUALITY - 4 x P_EXPLAINED. EVENT_QUALITY = max over the
cluster + 1 if geographically separated simultaneous witnesses. HIGH_QUALITY: EVENT_QUALITY >= 7
(sensitivity 6 and 8). Every major analysis is run on ALL events as well (score-free).

## 4. Known-object matching (`build_events.py::explanation_scores`, validated in `validate_matcher.py`)

Context components (time/place only): CNEOS bolides, meteor-shower activity and outbursts, rocket launches
(GCAT, incl. suborbital) 0-3 h before within 1,000-2,500 km and observer twilight, Starlink launch age,
ISS visibility from historical TLEs (SGP4; 1998-2004, 2012-13 only), Venus/Jupiter visibility, storm
reports, satellite re-entries, airport proximity, holidays. Text components: meteor, satellite/ISS,
planet/star, aircraft, drone, balloon, lantern, fireworks, lightning, searchlight, flare, cloud, plus NUFORC
analyst notes. Combined P = 1 - prod(1 - p_k); status LIKELY_EXPLAINED >= 0.6, AMBIGUOUS 0.3-0.6,
UNEXPLAINED_BY_MATCHER < 0.3. GEIPAN investigator class (A/B/C/D) supersedes the matcher for GEIPAN.

Validation against NUFORC analyst notes (English; text with the note removed): launch context sensitivity
0.83 / false-positive rate 0.007; planet context 0.41 / 0.015; Starlink 0.23-0.32 / 0.01-0.02; meteor text
0.51 / 0.035; satellites and aircraft poorly detected (<= 0.06). Against GEIPAN labels the French text
components look excellent but are contaminated by investigator conclusions in the narrative, and the
French aircraft rule has a 54% false-positive rate. Consequence: "unexplained by matcher" contains many
unrecognised conventional objects; this is a limitation of every 'unexplained' subset below.

## 5. Matched controls (`analysis_data.py`)

For each case (event with UTC time and coordinates), three control strategies, all measured with
identical code (`features.py`, `weather.py`, `spatial.py`, `iss.py`):
* CS1 time-stratified case-crossover (Janes, Sheppard & Lumley 2005): same place, same local wall-clock
  time, same weekday, all other such days in the same calendar month. Matches location, population,
  urbanicity, airports, bases, season, weekday, local time, year, reporting-system availability, internet
  adoption.
* CS2 same place and local wall-clock time, +-364 and +-728 days (same weekday). Informative for slowly
  varying exposures but confounded by multi-year reporting trends.
* CS4 same UTC instant, 4 populated places (GeoNames) in the same country, same population-size decile
  (+-1), same US Census region, >= 200 km away, sampled proportional to population.
Weather matching: CS1 with sky-cover-matched referents is reported as a sensitivity analysis.

Daylight-saving correction (external audit, item 1): referent calendar offsets are applied to the NAIVE
local time and localised with an explicit policy (nonexistent spring-forward times dropped; ambiguous
fall-back times take the first instance). 24,373 of 805,079 referents changed (16,284 CS1, 8,089 CS2), 49
added, 99 removed; all 805,029 referents now verified to share the case's wall-clock time and weekday
(`results/referent_dst_fix.json`). Case, CS4 rows and event splits were not changed.

## 6. Inference (`stats.py`, `cc.py`)

* Conditional logistic regression on matched sets (custom Newton-Raphson with step-halving; verified to 6
  digits against statsmodels ConditionalLogit). The Hessian is evaluated at the final estimate.
* Standard errors: two-way cluster-robust sandwich (Cameron, Gelbach & Miller 2011), clusters = case night
  (UTC date of t - 12 h) x 0.5-degree case-location cell. Rationale: all cases on the same night share
  global/regional exposure windows; repeat reports from one place share local ones.
* Predeclared rule (audit items 2 and 6): Wald inference only if the fit converged without separation.
  Binary exposures use EXACT conditional inference (Poisson-binomial distribution of exposed cases given
  the exposed count in each matched set; median-unbiased OR; tail-inversion 95% CI; two-sided p by
  doubling) when the fit separated / did not converge or there are < 10 exposed cases or < 10 exposed
  controls. Exposed-case / exposed-control counts are integers computed on the rows the model retains.
* Calibration (placebo exposures): real catalogs shifted by each distinct k x 364 days (k = +-3..+-15,
  each once; coverage window shifted with the catalog; missing stays missing) and Gaussian noise
  exposures. Results with Clopper-Pearson intervals: `results/placebo_calibration_summary.csv`.
* Multiple testing: Benjamini-Hochberg across ALL discovery tests and Holm (family-wise) reported;
  effect size, replication and robustness are prioritised over p.

## 7. Splits, freezing and holdouts

Event-level hash split (seed 20261004; `results/splits_manifest.json` stores the SHA-256 of each split's
id list): 60% discovery / 20% validation / 20% locked random holdout; all NUFORC events from 2016 onward
form a locked temporal holdout. Geographic holdout: GEIPAN (France) and non-US NUFORC. Independent-source
replication: Hatch *U* database, Blue Book unknowns, NICAP. Discovery uses ONLY NUFORC discovery events.
Candidates are selected by rules written before discovery results were inspected
(`select_candidates.py`; rules R1-R4 as amended after the external audit, see section 10) and frozen to
`results/frozen_hypotheses.json` whose SHA-256 is written before `validate.py` (which refuses to run on a
modified spec) reads any validation or holdout row.

## 8. Families tested in discovery (`discovery_tests.py`)

A global temporal (Kp, Dst, AE, IMF Bz, solar wind, density, pressure, E-field, protons, F10.7, sunspots,
flares, Moon, Venus, Jupiter, meteor showers, media events, Wikipedia attention) - CS1, CS2.
B local temporal (earthquakes M>=2.5 and M>=4 in 5 windows x 5 radii; storm reports; 17 weather variables
incl. fronts, clearing, inversion proxy; ISS; bolides; launches) - CS1, CS2, CS4.
C spatial (airports, DoD sites and boundaries, nuclear plants operating and closed, ICBM fields active by
year, DOE weapons sites, nuclear test sites, launch sites, special-use airspace MOA / restricted / alert /
prohibited, coast) at 1-500 km and log distance - CS4.
G window x radius grid (+-1 min ... +-30 d; 1 ... 500 km) for quakes, storm reports and launches - CS1, CS4.
Subsets: ALL, HIGH_QUALITY, HQ_UNEXPLAINED, MULTI_SENSOR, UNEXPLAINED, EXPLAINED (contrast).

Additional discovery tools: LightGBM / random forest / L1-logistic within matched sets with SHAP
interaction ranking (`ml_discovery.py`), every top interaction re-tested by a transparent conditional
logit product term; spline dose-response (GAM-like) in conditional logit; ordered sequences and
superposed-epoch analysis (`sequences.py`); spatial clustering with Kulldorff scan (Poisson AND
overdispersed nulls), residual Moran's I and Knox space-time tests (`spatial_analysis.py`); natural
experiments by difference-in-differences (`natural_experiments.py`); reporting-propensity analyses
(`media_waves.py`); multi-dataset convergence ranking incl. USGS magnetometers (`convergence.py`).

## 9. Validation (`validate.py`)

Each frozen candidate (variable, threshold, window, radius, subset, control strategy, direction) is
re-estimated without modification on: validation split; locked random holdout; locked temporal holdout
(2016-2023); GEIPAN; non-US NUFORC; Hatch; Blue Book unknowns; NICAP; leave-one-source-out pools.
Sensitivity grid: quality thresholds 6/7/8 and no score; exclude mass sightings; exclude military
witnesses; trained witnesses only; exclude events within 25 km of a large airport; drop each Census region
and each decade; all control strategies. Structure-preserving permutation: temporal exposures recomputed
after shifting all point times by k x 364 days; spatial exposures recomputed after translating the
facility layer by random 1-4 degree offsets.

## 10. External audit and corrections

An independent audit (Codex agent) of commit 963c37e found: DST drift in CS1/CS2; Wald inference under
separation; exposed-case counts taken from rows the model dropped; placebo calibration treating missing as
zero and reusing shifts; untestable Starlink controls counted as failures; non-significance counted as
specificity; and a selection rule weakened by untestable strategies. All were corrected (commit c793379
and later); every affected output was recomputed; superseded registry rows are retained and labelled.
Selection-rule changes were made after only the family-A discovery output had been viewed and only in the
stricter direction.

## 11. Evidence grades (operational definitions, fixed before validation)

Replication sets: R = {E1 validation, E2 locked random holdout, E3 locked temporal holdout (2016-2023)};
independent sets: I = {E4 GEIPAN, E5 non-US NUFORC, E6 Hatch, E7 Blue Book unknowns, E8 NICAP}.
"Replicated" = same direction and p < 0.05 with the frozen specification.

* GRADE A - replicated in >= 2 of R AND in >= 1 independent set of I; structure-preserving permutation
  p < 0.05; same direction in >= 80% of sensitivity variants; survives adjudication (spatial: urbanisation-
  adjusted CI excludes 1; and the effect is not equal in unexplained vs explained events, i.e. ROR CI
  excludes 1); not attributable to a conventional stimulus or to observation opportunity.
* GRADE B - replicated in >= 2 of R and permutation p < 0.05, but one or more of: no independent-source
  replication (or not testable there), adjudication failure, < 80% sensitivity agreement.
* GRADE C - discovery-only signal, replicated in only one of R, or any SECONDARY (non-promoted) hypothesis.
* GRADE D - replicated but adjudication identifies a conventional mechanism (observation opportunity,
  urban gradient, reporting system) or the effect is fragile (sign flips across sensitivity variants).
* GRADE F - not replicated in any of R, replicated in the opposite direction, or fully explained by a known
  stimulus (positive controls, e.g. launches, are graded F as 'explained' while counting as pipeline successes).
