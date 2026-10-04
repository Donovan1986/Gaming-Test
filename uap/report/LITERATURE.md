# Literature, conventional explanations, and novelty limits

This review separates material actually consulted from bibliographic leads supplied in
the handoff. It does not inspect validation or holdout outcomes. Public scholarly and
government hosts returned proxy `CONNECT 403` in this runtime, including Nature,
Crossref, DOI.org, NASA, NCBI/PMC, RAND, AARO, NOAA, and IMO. GitHub-hosted project
documentation and one archived bibliographic link were reachable with TLS
verification enabled. Therefore this is a **partial literature review**, not a
completed systematic search. No candidate receives `NO DIRECT PRECEDENT LOCATED`
on the strength of these blocked searches.

The machine-readable consultation log is
[`literature_sources.csv`](../results/literature_sources.csv). Successful documents
have content hashes and retrieval timestamps; blocked requests have no retrieved
content. Source-development documentation supplies measurement or statistical
context, not independent evidence that a UAP association exists.

## What the consulted sources support

| Reference | Supported point | Limit on its interpretation |
|---|---|---|
| [TidyTuesday, *UFO Sightings Redux*, 2023-06-20](https://github.com/rfordatascience/tidytuesday/blob/main/data/2023/2023-06-20/readme.md) | The dataset is a cleaned/enriched NUFORC release. Its `day_part` uses coordinates rounded to tens of degrees and dates rounded to weeks; coordinates and population describe gazetteer places. | A posting is an observation report, not a sampled event rate. These coarse annotations cannot establish exact local sky conditions or person-hours of exposure. Two NUFORC releases are not independent reporting systems. |
| [Skyfield, *Earth Satellites*](https://github.com/skyfielders/python-skyfield/blob/master/documentation/earth-satellites.rst) | Historical orbital elements and their epoch matter. SGP4/TLE position accuracy degrades away from epoch; visibility requires more than proximity. The documentation cites Vallado, Crawford, and Hujsak, *Revisiting Spacetrack Report #3* ([CelesTrak source](https://celestrak.org/publications/AIAA/2006-6753/)). | A possible ISS/satellite passage is not an identification without a compatible time, position, viewing direction, brightness and motion. A lack of archived matches does not exclude satellites. This documentation does not measure Starlink misidentification prevalence. |
| [Skyfield, *Almanac Computation*](https://github.com/skyfielders/python-skyfield/blob/master/documentation/almanac.rst) and [Astroplan, lunar illumination implementation](https://github.com/astropy/astroplan/blob/main/astroplan/moon.py) | Lunar phase/illumination and local altitude are different quantities; local horizon and twilight conditions must be computed. | Illuminated fraction alone is not local sky brightness. Moon altitude, weather, terrain, artificial light and observer behavior can mediate any apparent lunar association. These sources establish geometry, not a lunar effect on UAP reports. |
| [R `survival`, `clogit` documentation](https://github.com/therneau/survival/blob/master/man/clogit.Rd) | Conditional logistic regression conditions on matched sets. With one case and controls per set, the matched-set conditional likelihood has a specific equivalence to a stratified Cox likelihood. | Correct likelihood computation cannot repair unsuitable control times, reporting trends, exposure measurement error, or dependence between sets. An exact test remains conditional on its null and exchangeability assumptions. |
| Benjamini and Hochberg (1995), *Controlling the false discovery rate: a practical and powerful approach to multiple testing*, JRSS B 57(1):289–300; bibliographic entry consulted in [SciPy's official implementation/documentation](https://github.com/scipy/scipy/blob/main/scipy/stats/_morestats.py) | BH controls FDR for independent or suitable positively dependent tests. | The original paper was not retrieved. FDR is not a probability that an individual selected hypothesis is true, and it does not repair biased input p-values or an incompletely recorded search. |
| Benjamini and Yekutieli (2001), *The control of the false discovery rate in multiple testing under dependency*, Annals of Statistics, pp.1165–1188; bibliographic entry consulted in the same SciPy source | BY supplies a more conservative FDR procedure for general dependence. | The original paper was not retrieved. Overlapping radii, time windows, subsets and control strategies make dependence relevant; a BY sensitivity analysis and structured null checks are informative. |

The quoted statistical-paper bibliographic entries were verified against the
consulted SciPy document. No unchecked DOI has been added to those entries.

## Bibliographic leads that remain unresolved

**Environmental reporting geography.** The handoff's “Medina et al., Scientific
Reports 2023” lead corresponds to a located title and publisher link:
[*An environmental analysis of public UAP sightings and sky view potential*](https://www.nature.com/articles/s41598-023-49527-x),
DOI `10.1038/s41598-023-49527-x`. A
[GitHub archive of a 2023-12-15 news listing](https://github.com/jiacai2050/mofish/issues/496)
independently links that exact title to that publisher article. This verifies the
title/link as a **secondary bibliographic record**, not its author list, sample,
methods, estimates, or claims about airports and military areas. The primary
article was blocked. Spatial reporting associations should therefore be described
as having a located environmental-analysis precedent whose detailed comparability
remains unverified. Do not label this investigation a direct replication of that
paper without examining its methods.

**Tectonic strain.** Persinger's tectonic-strain theory was supplied as a handoff
lead; the targeted Crossref request was blocked. No title, publication year,
coauthor, estimate or specific physical mechanism is represented here as
bibliographically verified. A report–earthquake association, if observed, would
not itself test the proposed mechanism: catalog completeness, earthquake geography,
population, publicity, window overlap and reporting dependencies are alternative
explanations. It would require a structured null and independent replication.

**Nuclear facilities.** Hastings' nuclear-site claims were supplied as a handoff
lead. The author's public site was blocked. No particular book edition, incident,
quote or causal claim is adopted as established evidence. Retrospective incident
collections alone cannot supply the exposed and unexposed observation denominators
needed to estimate a rate ratio. Facility proximity and operational-change analyses
must separately account for settlement, military activity, surveillance and
reporting opportunity; temporal operational changes must be independently sourced.

**Government and institutional assessments.** The NASA 2023 UAP independent-study
PDF URL, NASA UAP page, an AARO report URL, and RAND report-page URL were attempted
but blocked. A failed URL fetch cannot verify a report's exact bibliographic
identity or conclusions. These organizations are recorded as search targets, not
as authorities for quotations or findings that were not read.

**Meteors, bright planets, Starlink and launches.** The orbital/sky-geometry sources
above support how to check visibility and uncertainty. They do not verify a
specific published UAP misidentification study. The IMO calendar was blocked, as
were scholarly search hosts. Meteor-shower, planet and aircraft positive controls
are internal validation tests, not literature novelty tests. Starlink-era reporting
must also distinguish deployment/launch dates from observer-specific visible
passes. Rocket-plume proximity needs viewing geometry and launch trajectories.

## Candidate interpretation and novelty rules

| Candidate class | Literature status usable now | Strong conventional alternative to test |
|---|---|---|
| Airports, military areas, population and sky-view geography | Environmental-analysis precedent located by title/link; methods not verified. Exact-result novelty `NOT_ASSESSED`. | Observer population and activity, flight traffic, local reporting-system reach, visibility and settlement. |
| Moon illumination, phase, altitude and darkness | Geometry documented; UAP-specific precedent search incomplete. `NOT_ASSESSED`. | Changes in sky contrast, visibility and observing/reporting behavior. |
| Sunspots/F10.7, geomagnetic and solar-weather variables | Official solar-cycle page and scholarly searches blocked. `NOT_ASSESSED`. | Long-term changes in database coverage/reporting, correlated solar exposures, seasonality and multiple testing. |
| Earthquakes/tectonic variables | Persinger handoff lead remains unverified. `NOT_ASSESSED`. | Geographic/reporting structure, catalog coverage and dependent event windows. |
| Nuclear installations and operational changes | Hastings handoff lead remains unverified. `NOT_ASSESSED`. | Military/sensor presence, nearby population and publicity; treatment-date misclassification and nonparallel trends. |
| Satellites, ISS, Starlink, meteors, planets and launches | Conventional match calculations have documented measurement requirements; UAP empirical precedents not fully searched. `NOT_ASSESSED`. | Known luminous sky phenomena plus imperfect matcher sensitivity. |
| Washington/reporting-source clusters | Release provenance documented; independent spatial literature search incomplete. `NOT_ASSESSED`. | Reporting-system geography and spatial exposure structure. |
| Multivariable interactions, sequences and convergence | Exact-pattern searches await a frozen definition and accessible literature sources. `NOT_ASSESSED`. | Reused correlated predictors, timestamp uncertainty, missing sensors and selection over many alternatives. |

`NOT_ASSESSED` is a missing-assessment status, not a novelty claim. Once accessible
sources have been read, use the requested categories (`WELL KNOWN`, `PREVIOUSLY
SUGGESTED`, `SIMILAR RESULT EXISTS`, `NO DIRECT PRECEDENT LOCATED`) for the exact
relationship. Distinguish a known measurement/physical phenomenon from an empirical
association in unexplained observations. Failed searches alone never justify
“never discovered before.”

For any future completed review, record searches in academic databases and the
NASA/AARO/RAND and field-specific literature, including exact variable combinations,
windows, radii and population. Document search dates, successful texts consulted,
and differences in outcomes and control designs. A frozen hypothesis and holdout
hash prevent changing the hypothesis after evaluating its held-out result; they
do not retroactively preregister an exploratory discovery. Novelty and evidence
grade must remain separate assessments.
