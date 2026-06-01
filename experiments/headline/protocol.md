# STRIDE Headline Protocol

Status: locked draft for Stage 1.4. Tag this commit as `headline-protocol-v1` before any adaptive benchmark runs.

Locked date: 2026-06-01

## Evaluation Set

Primary benchmark proteins are fixed as the following candidate list. Entries with unresolved access or license remain blocked until verification debt is closed; they must not be used for training.

| Protein / family | Role | State definitions | Access status |
|---|---|---|---|
| DESRES Anton BPTI 1 ms | primary headline equilibrium benchmark | 5 PCCA+ macrostates from Shaw et al. 2010 reference MSM | access request required |
| Villin | fast-folder time-to-first-fold | folded / unfolded / intermediate by literature RMSD/contact thresholds | trajectory license audit required |
| Trp-cage | fast-folder time-to-first-fold | folded / unfolded / intermediate by literature RMSD/contact thresholds | trajectory license audit required |
| WW domain | fast-folder time-to-first-fold | folded / unfolded / intermediate by literature RMSD/contact thresholds | trajectory license audit required |
| Chignolin | fast-folder time-to-first-fold | folded / unfolded / intermediate by literature RMSD/contact thresholds | trajectory license audit required |
| NTL9 | fast-folder time-to-first-fold | folded / unfolded / intermediate by literature RMSD/contact thresholds | trajectory license audit required |
| Src-family kinase | kinase conformational benchmark | DFG-in / DFG-out / alphaC-in / alphaC-out with explicit torsion and distance thresholds | source trajectory TBD |
| beta2-AR or related GPCR | membrane-protein benchmark | ionic lock, NPxxY, TM6 outward swing | GPCRmd per-deposition audit required |
| 2-3 held-out mdCATH kinase/transporter domains | mdCATH-held-out qualitative backup | pre-registered structural thresholds only | must be selected before Stage 4 |

## State Definition Rules

Evaluation states must be independent from Stage A0 labels and model embeddings.

Allowed state sources, in order:

1. Literature-named states with explicit structural thresholds.
2. PCCA+ macrostates from a reference long-equilibrium MSM built once before adaptive runs, using features disjoint from training labels.

Training-label clusters, VAMPnet states, learned embeddings, or scheduler archive regions are not valid evaluation states.

## Metric

Headline metric is AUC of the states-discovered curve versus simulated nanoseconds.

A state is discovered when either condition is met:

- at least one adaptive-run frame falls inside the locked structural definition, or
- at least 1 ns accumulated simulation time falls inside that definition.

Report per-protein, per-seed curves and the aggregate AUC delta versus each baseline.

## Statistical Test

Use paired bootstrap over protein x seed with 1000 resamples. Report 95% confidence interval on AUC delta against each baseline.

STRIDE wins a baseline only if the 95% CI excludes 0. Required baselines are Random, Round-robin, RMSD-velocity, tICA-distance, MSM-counts, FAST, REAP, and AdaptiveBandit.

## Test-Set Touch Policy

The locked evaluation set may be touched twice after this protocol is tagged:

1. the headline benchmark run,
2. a camera-ready rerun with documented code fixes only.

Any change to protein list, state definitions, metric, bootstrap procedure, or test-set touch policy requires a new protocol tag and must be disclosed.

## Current Stage 1.4 Gaps

- DESRES BPTI access request must be submitted manually by the project owner because the form requires identity/account details.
- Public fast-folder trajectory licenses must be audited before download or use.
- GPCRmd per-deposition license must be audited before GPCR inclusion.
- Exact kinase/GPCR threshold constants must be filled from cited literature before Stage 4.
