# Inference versions

* v1 (rows timestamped before 2026-10-04T07:15:04Z): conditional-logit Wald SE from the model Hessian, then one-way
  cluster-robust SE by 0.5-degree case-location cell. Placebo calibration (real catalogs shifted by whole
  multiples of 364 days) showed 15-19% of placebo tests with p<0.05 -> anti-conservative. SUPERSEDED.
* v2 (rows from 2026-10-04T07:15:04Z): two-way cluster-robust (sandwich) SE, clusters = case night (UTC-12h date) x
  0.5-degree case-location cell (Cameron, Gelbach & Miller 2011). Placebo calibration: 5.1% of 175 placebo
  tests with p<0.05 (noise 4%, shifted launches 0%, shifted quakes 12%). Local-catalog exposures (quakes)
  additionally require a structure-preserving permutation test before any claim.
