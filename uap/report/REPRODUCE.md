# Reproducing and resuming the investigation

Use the existing checkout on `claude/cool-cannon-qn5hbl`; cloud tasks already have
isolated environments. The pinned interpreter is
`/workspace/.venvs/gaming-test/bin/python` (Python 3.11.15). Install the exact
`uap/requirements.txt` pins with the saved environment installation script first.
The recorded machine has four CPUs and approximately 32 GB RAM. The runner uses
50,000-point feature chunks, four Numba threads, and two BLAS/OpenMP threads.
These reduce temporary allocations; they are not a hard memory limit. The full
analysis remains a substantial computation.

From the repository root, perform the offline readiness check:

```bash
/workspace/.venvs/gaming-test/bin/python uap/src/run_investigation.py --preflight-only
```

It verifies **actual local SHA-256 values** against successful historical manifest
entries. A stale `input_restoration.json` cannot establish readiness. Missing,
mismatched or unsafe inputs cause exit **2** before scientific stages. Missing
files are never interpreted as zero events, and old result CSVs are never accepted
as results of this run. The small tracked
[`investigation_status.json`](../results/investigation_status.json) records readiness
and the current stage; detailed run journals and command logs use ignored `.log`
files under `uap/logs/`.

After the runtime permits the required public domains, restore and run:

```bash
/workspace/.venvs/gaming-test/bin/python uap/src/run_investigation.py --restore --resume
```

`--restore` invokes `restore_inputs.py --stage all`. Existing snapshots are
checksum-verified and reused; TLS verification remains enabled. Changed source
responses are quarantined. A deliberately reviewed current snapshot can be
accepted separately with `restore_inputs.py --accept-current data/raw/PATH`; the
historical expectation remains intact and the runner records the replacement.
This changes the data fingerprint and is not an exact historical reproduction.
The runner itself never automatically accepts changed snapshots.

Without network retrieval, use `--resume` after inputs have been restored. Resume
requires the same code, data, pinned software and option fingerprints, the same
commands, and matching hashes of every completed stage's outputs. Changed or
missing outputs block resume. A preflight check or temporarily missing input keeps
the previous journal reference so it does not discard resumption history.
An existing hypothesis freeze without a matching runner journal blocks reopening
discovery. Preserve and investigate a blocked run instead of deleting its freeze
or rewriting its fingerprints.

The initial ready manifest is pinned by its byte-prefix length and SHA-256 in the
run journal. Resume verifies that unchanged prefix and uses it as the frozen raw
input contract. Later supplementary measurements, such as convergence-stage
magnetometer queries, are logged with the current manifest and append hashes and
their provenance entries after each stage. Resume verifies every successful
supplementary raw file's checksum; appending a valid new measurement does not
silently change the frozen input fingerprint. A changed prefix or mutated
supplementary file blocks resume. New snapshot promotions cannot rewrite the
already qualified input contract.

The execution order is fixed:

1. Extract verified gazetteer archives and reconstruct the original weather
   station set from acquisition history, without reading scientific outcomes.
2. Build reports, deduplicated/scored events and deterministic split manifests.
3. Generate matched controls and features, then add weather.
4. Rerun positive controls, placebo calibration and detection-power checks.
5. Run registered discovery tests, discovery-only ML, sequences and spatial
   analyses.
6. Select eligible candidates and create an immutable hypothesis specification
   and SHA-256 before validation. The freeze includes code/data/runtime identity,
   effect scales, validation settings and the split-manifest hash.
7. Validate frozen candidates; evaluate Starlink integrity checks after unlocking.
8. Run solar-cycle adjudication, convergence screening, natural experiments,
   media/reporting checks and the data-source report after verifying the freeze.

Existing stage outputs and the hypothesis registry are copied to ignored,
timestamped `.backup.log` files before a rerun. Failed subprocesses stop immediately
(exit **1**); required missing output files also fail the stage. The journal records
commands, start/finish times, exit codes, durations, software versions, Git revision,
source fingerprints and output hashes. Historical outputs remain available as
backups, and are not fresh evidence. Registry schema repair, when necessary, is a
separate documented lossless operation.

After inputs pass preflight, existing generated `data/interim/*.parquet` caches
are archived with checksum verification and removed before the first rebuild.
The journal records this reset once; a resume does not invalidate the newly built
caches. This prevents old cached astronomy, weather, covariate or explanation
features from bypassing the corrected source code. Raw inputs remain intact.

`--feature-chunk N` changes temporary feature batch size. `--permutations N`
requests exposure-shift diagnostics, default **30**. The temporal diagnostic has
at most 30 distinct offsets; requesting more does not create thousands of
independent randomized worlds. Its empirical ranks are not calibrated permutation
p-values. See the validation outputs for successful distinct-null counts and
limitations. Options become part of the immutable run/freeze identity.

`COMPUTATION_COMPLETE_REVIEW_REQUIRED` means the scheduled computations finished;
it does not certify a discovery. Assess positive-control failures, calibration,
missingness, corrected significance, independent replication, sensitivity and
conventional explanations before assigning evidence grades. Successful relevant
positive-control gates are mandatory before any novel relationship can be trusted.
A failed or untestable control is not a passed gate even if the computation
continues for exploratory description. Discovery ML and
sequence results currently remain exploratory; they are not automatically promoted
into frozen candidates by the selection rules. Prediction backtesting and a
prospective claim require separate candidate qualification and implementation.
The runner records prospective prediction as unqualified and does not invent a
forecast if no signature survives.

Development verification uses synthetic fixtures and mocks only:

```bash
/workspace/.venvs/gaming-test/bin/python -m unittest discover -s uap/tests -p 'test_runner.py' -v
```

Those tests do not execute research stages or consume holdout data. Full scientific
validation requires restored inputs and the complete ordered run.
