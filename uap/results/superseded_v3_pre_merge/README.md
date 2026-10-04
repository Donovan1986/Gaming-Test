# Superseded v3 discovery outputs and freeze (never validated)

These files were produced with inference v3 and the pre-merge covariate code. The freeze
(sha256 532e42482d1aba80cc2a6639bb1e0714dcfd6af37a93d96e80c37fed0fd73575, commit f162b1f) was
superseded BEFORE any validation or holdout row was read, because the code base was then merged
with independent hardening (commit 494c248: inference v4 cluster-safe/scale-invariant estimation;
catalogue-coverage masks for quakes and storm reports; nearest-observation weather matching
without imputation; exact-time Earth rotation for altitudes; ISS completeness). Those changes alter
covariate values, so discovery, selection and the freeze were rerun from scratch with the merged
code under the journaled runner (`uap/src/run_investigation.py`). Kept for the hypothesis history.
