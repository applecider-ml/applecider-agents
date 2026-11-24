# Joint FLARE + ASTRANet workflow

The joint workflow runs photometric classification/anomaly assessment with FLARE
and spectroscopic classification/novelty assessment with ASTRANet. It retains the
original outputs separately, maps `SN_Ia` to `Ia` and `SN_CC` to `II`/`Ibc` for
comparison, and flags disagreements for human review. It does not average the
models' probabilities or interpret their novelty scores as discovery probabilities.

## Install and configure

From the repository root, in your chosen virtual environment:

```bash
python -m pip install -e '.[flare,astranet]'
# Optional, for MCP serving:
python -m pip install -e '.[mcp]'
```

FLARE model files and ASTRANet checkpoints are still required. ASTRANet expects
`ASTRANET_MODELS` (the twelve checkpoints) and `ASTRANET_NOVELTY` (the novelty
bundle; defaults to `novelty_v1.joblib` in the models directory). Installing Python
packages alone does not verify these artifacts. LightGBM also requires a working
native OpenMP runtime; on macOS a missing `libomp.dylib` prevents FLARE inference.
Use an OpenMP runtime compatible with the architecture of your Python installation.

Set these environment variables locally. The application reads the process
environment; it does **not** automatically load `.env` files.

```bash
export SKYPORTAL_URL='https://your-skyportal-instance.example'
export SKYPORTAL_TOKEN='your-token'
export ASTRANET_MODELS="$PWD/models/astranet"
export ASTRANET_NOVELTY="$ASTRANET_MODELS/novelty_v1.joblib"
export FLARE_MODE=dry-run
```

The example model directory must be replaced with the directory holding your actual
checkpoints. The SkyPortal token needs access to the requested source, photometry
and spectra. Missing permissions are reported as errors, not empty successful results.
Optional TNS credentials enable known-object checks; missing credentials are reported.

## SkyPortal source and batch commands

```bash
applecider-agents skyportal triage SOURCE_ID --model joint --detailed
applecider-agents skyportal scan --since-days 1 --limit 20
```

Both commands are read-only. Scan respects `SKYPORTAL_GROUP_IDS`, follows pagination
and returns separate `reports` and per-source `errors` arrays. Results are logged to
`APPLECIDER_AGENTS_LOG/collaboration_reports.jsonl` (default: logs under
`APPLECIDER_AGENTS_HOME`). JSON output includes:

- `model_results.flare`: photometric classification and anomaly outputs.
- `model_results.astranet`: spectral classification, novelty and spectral report.
- `data_status`: `ok`, `partial` or `unavailable` for each requested model.
- `specialist_assessments`: findings, errors, caveats and raw evidence.
- `provenance`: snapshot hash, fetch time, input counts, excluded photometry and
  unresolved issues. The hash identifies the input snapshot; it is not a stored
  copy of the original observations.

SkyPortal ingestion uses AB magnitudes in `ztfg`, `ztfr`, `ztfi`. Other filters,
upper limits, invalid measurements and exact duplicates are counted and excluded;
other instruments' filters are not relabelled as ZTF. Spectra are supplied in the
observed frame, wavelengths in Angstrom and fluxes as returned by SkyPortal.
The source redshift is used by ASTRANet, not mislabelled as a host photometric redshift.
The connector does not invent host/Gaia/WISE context from a source record; its absence
is explicitly reported. SkyPortal classifications are retained in provenance, not
assumed to be independent spectroscopic truth (some may be machine-generated).

A fresh connector read invalidates earlier feature and inference caches for that
object. If one modality fails, the other can still contribute, with the missing
modality explicitly marked. A missing anomaly/novelty stage is not evidence that
an object is ordinary.

To explicitly publish a report, both the command and the mode must enable it:

```bash
FLARE_MODE=comment applecider-agents skyportal triage SOURCE_ID --post
```

This writes a comment and an annotation under `AppleCiDEr-collaboration`, separate
from FLARE/ASTRANet service annotations. No observations or follow-up requests are
submitted. Repeating `--post` can create another comment; publishing is not an
idempotent job queue.

## Other inputs and optional LLM review

```bash
# Existing local/broker photometry plus locally configured or SkyPortal spectra:
applecider-agents collaborate SOURCE_ID --model joint --detailed
# Spectroscopy only (does not run FLARE classification):
applecider-agents collaborate SOURCE_ID --model astranet --detailed
# Expose individual and joint tools to an MCP client:
applecider-agents mcp --models flare,astranet
```

For local spectra use `APPLECIDER_SPECTRA_DIRS`, with files named `SOURCE_ID*.dat`,
`.ascii` or `.txt` containing wavelength and flux columns. Directory lists use the
operating system's path separator. MCP exposes `joint_triage` and the read-only
`skyportal_joint_triage` in addition to individual model tools.

`--llm` now calls the Anthropic API for an **advisory review of the collected
measurements**. Set `ANTHROPIC_API_KEY` and `APPLECIDER_AGENTS_MODEL` to a model
available to your account. This sends the report evidence to the configured model
provider. The LLM cannot overwrite measurements, the rule-based verdict or ranking.
`llm_status` and `llm_error` expose failures while preserving the deterministic report.
This is one synthesis call, not an autonomous debate among independent LLM agents.

## Scientific interpretation and validation

`discovery_potential` is retained for compatibility but is a **heuristic follow-up
ranking**, not `P(discovery)`. `score_kind` and output labels make that explicit.
Specialist heuristic confidences likewise are not calibrated discovery probabilities.
For joint/spectral reports, read `model_results` rather than the legacy top-level
FLARE-specific fields. Agreement of class labels is not an independent confirmation
of a physical discovery. Disagreement remains unresolved until inspected.

Before using rankings to allocate observing time, evaluate on held-out objects
with realistic missing-data patterns and compare FLARE-only, ASTRANet-only and joint
selection at the same follow-up budget. Measure precision/recall, false positives,
and recovery by class; use time-based splits to avoid future-data leakage. The
photometric `--horizon` does not time-truncate spectra: all supplied spectra are
analyzed. For early-time evaluation, provide only spectra available at that epoch.

The existing `eval --model joint --list ood` / `--list test` paths hide BTS/TNS labels
from the workflow, but require suitable spectra and model assets. They do not by
themselves establish improved scientific performance or calibrate the ranking.

Offline regression tests for the wiring and error handling:

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_joint_workflow.py tests/test_skyportal_connector.py
```

These tests mock inference and HTTP responses. They validate orchestration, taxonomy
comparison, error propagation, cache invalidation, pagination, publishing gates and
LLM routing; they do not measure model accuracy or certify a live deployment.

API contract references: SkyPortal's
[photometry handler](https://github.com/skyportal/skyportal/blob/main/skyportal/handlers/api/photometry.py)
and [spectrum handler](https://github.com/skyportal/skyportal/blob/main/skyportal/handlers/api/spectrum.py).
