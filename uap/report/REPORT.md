# UAP investigation: blocked rerun and reproducible continuation

**Scientific status: `BLOCKED_NOT_RERUN`.** The corrected investigation has not
produced a current discovery, replication, prediction, or scientific null result.
The runner stopped at input preflight with exit code **2**,
`state=BLOCKED_INPUTS`, `research_executed=false`, and
`scientific_conclusion=NOT_EVALUATED`. See
[`investigation_status.json`](../results/investigation_status.json).

This continuation starts from Claude's commit
`c793379dcca9537ed36a6f5486c9d9fcd5f08388` on
`claude/cool-cannon-qn5hbl`. The handoff says the positive controls, calibration,
discovery, machine-learning search and sequences had not been rerun after its
DST, inference and selection fixes. Older CSVs remain provenance artifacts;
their estimates and claimed significance are superseded as current findings.
This also applies to spatial/natural-experiment summaries that the handoff said
did not depend on those particular fixes: they were not independently rerun here.

## What is verified in this environment

**8 of 9,907 historical raw snapshots match their original SHA-256 hashes;
9,899 remain absent.** The missing files include GEIPAN, environmental catalogs,
population/gazetteers, facility layers and weather. Research hosts returned proxy
`CONNECT 403 Forbidden`; this is an observed runtime access restriction, separate
from upstream HTTP failures or missing credentials. Domain additions saved in an
environment draft do not establish that runtime access changed.

[`input_restoration.json`](../results/input_restoration.json) records the full
inventory. An offline audit classified missing files using previous denial
evidence; it did **not** make 9,899 separate successful or failed downloads.
The verified files are four NUFORC release files, Hatch, NICAP, Blue Book JSON,
and JPL DE421. The two NUFORC releases are one reporting system, and overlaps
between historical compilations require deduplication before independence claims.

| Verified raw file | Original snapshot SHA-256 |
|---|---|
| `nuforc/tt2023_ufo_sightings.csv` | `ad72ebc6cd34001144e8bea85524d34390f8e12d13b352b184ed57557a980647` |
| `nuforc/tt2023_places.csv` | `6fe1aff91fc019e29e0f0c6e88d66ca9b74df01550ed38e56ff2507ca79f80bf` |
| `nuforc/tt2023_day_parts_map.csv` | `c3969119d51829d387795df6bc9975d705f75ef7234757e9fa94adcea3dbb872` |
| `nuforc/planetsig_scrubbed.csv` | `48c804c5923b3ab31118fd5a7b5e4c7578ddb1dd66f6a1418fa1c954ea644405` |
| `catalogs/hatch_udb.json` | `e2186970c688a99366eee78778c4a3e1ec64c05daf0439c5749f56d0b1ce7cd5` |
| `catalogs/nicap_db.json` | `be101a665fb9c1d97df935a1426febae05e3bdfea07e0b42c561a5d7b5c10efe` |
| `catalogs/bb_unknowns_geldreich.json` | `b0317f23d6b3cc5384ab4015d3856ebc657db22a5c349da5e234c6147ec3b1bf` |
| `astro/de421.bsp` | `a20a7139da04cbc462454634918e9a9ca69127044e2cc9d4f9c16e238d2deedc` |

DE421 was recovered from the `skyfield-data` 7.0.0 PyPI wheel after verifying
both its published distribution SHA-256 and the extracted file's historical
SHA-256. The actual artifact URL, member and retrieval timestamp are appended to
[`manifest.jsonl`](../data/manifest.jsonl); original records and expected hashes
were preserved. No divergent current snapshot was accepted.

The restorer supports bounded concurrency, backoff, checksum resume, mismatch
quarantine and atomic publication without overwriting a conflicting file.
Its integrity/preservation checks passed. It can recover the **352 original
weather-station IDs** from successful and failed acquisition attempts; joining
them to the unavailable official station-history file remains blocked.

Python is pinned to 3.11.15 and dependency versions are recorded in
[`requirements.txt`](../requirements.txt). Software smoke/regression checks
verify development mechanics, not empirical UAP relationships. **99 synthetic
tests and nine environment smoke groups pass**; the smoke check ingested
96,429 NUFORC records and the installed 52 packages pass dependency compatibility
checks. The test commands and source fingerprints are recorded in
[`verification.json`](../results/verification.json). The runner
records actual input hashes, code/runtime fingerprints, commands, output hashes
and stage outcomes when research can start; no completed research journal exists
for this blocked preflight. The incoming research specification's SHA-256 is
`69116e19438c708fa8f7a6328c84650a691f40c30741aec713774c7923c0ceff`;
the incoming Claude handoff's is
`2c8d811888dbdf3899624b456b5ade4e2798466e8fe67d7ce59f28ee66197182`.

### Corrections verified with synthetic tests

- Discovery cohort selection excludes temporal, geographic and random holdouts
  across discovery searches, ML, sequences and relevant secondary analyses.
- DST referents preserve local clock time/weekday; unavailable coordinates,
  catalog coverage, weather and failed ISS propagation remain unavailable.
  Weather matches preserve actual observation timing, and astronomical sidereal
  rotation uses event timestamps instead of rounding them to ten-minute bins.
- Conditional estimates are invariant to measurement units. Invalid,
  nonconverged, separated and insufficient-cluster fits do not receive valid Wald
  inference. Independence-based exact results cannot become eligible inference
  when retained cases share supplied clusters.
- Frozen specifications are immutable and checksum-checked; validation
  reconstructs frozen grid exposures and applies Holm correction across all
  scheduled validation and leave-one-source-out tests, retaining nominal verdicts
  separately. Temporal shift diagnostics use at most
  **30 distinct shifts**, report excluded fits/coverage, and are explicitly
  sensitivity diagnostics rather than calibrated permutation p-values.
- ML imputation is fitted on training data; singular spline designs report an
  untestable status instead of aborting or inventing significance.

Geographic masks require an entire earthquake exposure radius within a fetched
catalog box and an entire storm radius within restored US Census boundaries.
These conservative rules exclude some otherwise supported boundary locations;
missing boundary data cannot establish zero storms. Astronomy still uses
approximate geocentric body positions in ten-minute bins, and event-grouped ML
cross-validation does not establish temporal or spatial generalization.

These tests exercise known constructions rather than blocked environmental data.
The mixed-layout hypothesis registry was recovered without adding research
records: **1,017 original rows and all original cells/test IDs were retained**;
new schema cells are blank. The original and normalized hashes and layout
evidence are in [`registry_recovery.json`](../results/registry_recovery.json).

## Historical follow-up inventory

The six rows below are **unranked historical leads**, not six qualifying
discoveries or a frozen selection. Reporting ten discoveries would invent
evidence. [`patterns_table.csv`](../results/patterns_table.csv) preserves every
requested machine-table column and adds fields covering all 24 requested
per-candidate items, status and provenance. Current event/control counts,
estimates, intervals, p-values and robustness scores are empty because they
are not evaluated; emptiness never means zero. No letter evidence grade or
novelty claim is assigned.

| ID | Historical lead and source | Main qualification needed |
|---|---|---|
| H01 | Sunspots/F10.7: the handoff reported a preview association only under the different-year controls. | A database reporting trend can produce this contrast; require both temporal strategies and independent replication. |
| H02 | Moonlight: the handoff reported a small suppression under both temporal strategies. | A conventional visibility/observer-behavior relationship; quantify local geometry, weather and baseline reporting before interpretation. |
| H03 | Washington-centered clusters: historical [`spatial_scan_clusters.csv`](../results/spatial_scan_clusters.csv), also described in the handoff. | Clusters also appeared in explained reports; reporting-system geography is a strong alternative. |
| H04 | Space-time proximity: historical discovery rows in [`spatial_knox.csv`](../results/spatial_knox.csv). | Deduplication, correlated reporting and geography can create interacting event pairs; rerun structured nulls and held-out checks. |
| H05 | ICBM deactivation: historical [`natural_experiments.csv`](../results/natural_experiments.csv). | The historical bootstrap interval included no effect; few facilities, nonparallel trends and treatment-date uncertainty remain. |
| H06 | Nuclear-plant shutdown: the same historical summary, using separate 25/50 km analyses. | Historical estimates did not support a clear effect; exposure denominators, facility comparability and operational dates need verification. |

The historical summaries do not supply independently checked representative
cases. None were selected because their narratives fit a lead. The current run
has not read processed validation/holdout outcomes or tested a new candidate.

## TOP RESULT

**No defensible current top candidate can be identified pending the corrected
rerun.** This does not establish that no reproducible pattern exists. What
happens, when/where it happens, its excess frequency, preceding/following
variables, conventional-explanation coverage and replication are all unresolved
for a current selected finding. There is no supported empirical temporal chain
to draw, so no arrows implying an observed ordering are supplied.

| Requested evidence-chain element | Current evidence |
|---|---|
| VARIABLE A | No qualifying exposure has been selected and frozen. |
| VARIABLE B | No independently replicated preceding/following variable. |
| CONDITION C | No robust combination established. |
| ELEVATED UAP EVENT RATE | Not estimated against a verified matched baseline. |

The strongest immediate follow-up is restoration followed by calibration. A
future prospective rule must be frozen after replication, use only information
available before each forecast, outperform a matched reporting baseline in a
rolling backtest, and log coverage and non-event intervals. No prediction claim
is made here.

Historical descriptive features include three-hour Kp, hourly OMNI and daily
indices. A value assigned to a bin can include measurements or publication
information later than an event inside that bin. Weather matching can also use
a later nearby observation. Prospective prediction therefore requires verified
measurement/publication availability times, explicit lags and strictly past
weather observations; contemporaneous historical bins are not before-event
predictors merely because their column names include “prior.”

## Scientific gates before any claim

1. Restore and verify required inputs; disclose reviewed replacements rather
   than changing historical checksums. Build reports/events with explicit time,
   location and measurement uncertainty and audit deduplication/exclusions.
2. Verify matched controls and rediscover the available meteor, astronomy,
   aircraft and weather relationships. Keep `PASSED`, `FAILED`, `INCONCLUSIVE`
   and `NOT_TESTABLE` distinct. Successful program execution alone does not
   establish successful calibration. Starlink is not testable in the historical
   discovery era and belongs to a later integrity check after hypothesis freeze.
3. Record every discovery test, missing exposure and model status. Apply the
   declared control-strategy requirements, practical-effect/count thresholds,
   multiplicity correction and structured null checks. Repeated or overlapping
   time shifts do not provide thousands of independent permutations.
4. Freeze exact variables, thresholds, windows, radii, subsets and directions
   with a checksum before reading held-out outcomes. Reproduce unchanged results
   geographically and across independent systems; assess leave-one-source-out,
   quality, uncertainty, region and conventional-explanation sensitivity.
5. Complete candidate-specific literature review and prediction qualification.
   A computation-complete runner status still requires these scientific gates.

Missing ADS-B, radar, satellites or sensor data cannot exclude those explanations.
“Unexplained by matcher” is not a validated physically unexplained observation;
the handoff acknowledges limited sensitivity. Population proxies are not complete
observable person-hours, and unreported observations are not verified absences.
No causal, extraterrestrial, technological or paranormal inference is supported.

[`LITERATURE.md`](LITERATURE.md) and
[`literature_sources.csv`](../results/literature_sources.csv) distinguish material
consulted from blocked requests and handoff leads. NASA/AARO/RAND and primary
scholarly access was blocked. Exact-pattern novelty remains `NOT_ASSESSED`;
unreachable literature cannot justify “no precedent.”

## Resume in the existing checkout

Enable the listed research hosts in the environment's **runtime** network
settings; requirements appear in `investigation_status.json`. No secret values
should be supplied in chat. Then use the existing isolated checkout:

```bash
cd /workspace/Gaming-Test
/workspace/.venvs/gaming-test/bin/python uap/src/run_investigation.py --preflight-only
/workspace/.venvs/gaming-test/bin/python uap/src/run_investigation.py --restore --resume
```

The first command currently exits 2. After access changes, the second restores
manifest snapshots, builds the tables/features/weather, runs controls and
discovery, freezes selection, and proceeds to validation and secondary analyses.
It reuses only completed stages whose commands and output/code/data/runtime
hashes still match. The default feature chunk is 50,000 rows; the handoff's full
workload needs roughly 14 GB RAM and hours of runtime. If restoration reports a
changed mutable source, review its quarantined response and separately record any
accepted replacement before retrying. Preserve existing frozen designs; do not
reopen discovery against their holdout outcomes.

Once the corrected computation and scientific gates complete, replace this
blocked report/table with actual estimates, independent replication, exact
novelty classifications, representative-case provenance and any qualified
backtest. A null finding can then be reported if supported by the completed
analysis.
