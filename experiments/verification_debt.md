# Verification Debt

Track claims that must be verified before code or training depends on them.

| Item | Stage gate | Status | Notes |
|---|---|---|---|
| STRIDE-MD license and bulk endpoint | Stage 1 | Open | Re-verify before Stage 1 pretraining uses STRIDE-MD. Do not download or train on STRIDE-MD yet. |
| DESRES BPTI access | Stage 1.4 | Owner action required | Submit request through D. E. Shaw Research download form; required for headline benchmark, not for Stage A0 smoke labels. |
| GPCRmd per-deposition license | Stage 4 | Open | Required before using GPCRmd in evaluation or training. Audit each deposition independently. |
| Public fast-folder trajectory licenses | Stage 4 | Open | Audit each Zenodo or source deposit independently before download or benchmark use. |
| Kinase/GPCR structural thresholds | Stage 4 | Open | Fill exact torsion/distance constants from cited literature before adaptive benchmark runs. |
