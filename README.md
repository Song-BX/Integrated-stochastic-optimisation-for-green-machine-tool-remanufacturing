# Integrated stochastic optimisation for green machine-tool remanufacturing

This repository contains the modelling code, parameterised data, computational
results and manuscript artifacts for an IJPR-oriented study on green
machine-tool remanufacturing.

The study formulates remanufacturing as a risk-aware production planning
problem. It jointly considers recovered-core acceptance, component routing,
procurement, inventory, capacity, delivery, residual-life reliability, CVaR risk
control, selective assembly, multi-objective Pareto analysis and matheuristic
scalability.

## Research context

The industrial background is machine-tool green remanufacturing. The
parameterised instances are simulated from local machine-tool remanufacturing
case evidence and are structured within the product, process and parameter
ranges motivated by:

Song et al. (2023), *The promotion and application of green remanufacturing: a
case study in a machine tool plant*.

The repository does not expose partner-identifying raw enterprise records. The
data are anonymised and parameterised for optimisation experiments.

## Main contributions

- Multi-period, multi-component, multi-route and multi-scenario stochastic MILP
  for machine-tool remanufacturing.
- Sample average approximation for demand, component-quality and route-outcome
  uncertainty.
- Residual-life chance constraints and CVaR risk aversion.
- Feature-level selective assembly with compatibility, dimension-chain and
  assembly-quality constraints.
- Augmented epsilon-constraint Pareto analysis for economic risk,
  environmental impact and assembly-quality loss.
- ALNS plus restricted MILP repair matheuristic for larger benchmark instances.
- Experiment-suite, evidence-audit and manuscript-ready table/figure pipeline.

## Repository layout

```text
.
├── AGENTS.md
├── README.md
├── data.md
├── data/
│   ├── raw/          # parameterised input data
│   ├── processed/    # processed model and reporting data
│   └── results/      # optimisation, experiment and figure outputs
├── scripts/          # command-line stage runners
└── src/              # stage-specific Python packages
```

## Stage structure

| Stage | Purpose |
| --- | --- |
| Stage 1 | Data catalogue and validation |
| Stage 2 | Deterministic MILP baseline |
| Stage 3 | Multi-period, multi-component and multi-route MILP |
| Stage 4 | Stochastic SAA model |
| Stage 5 | Chance constraints and CVaR risk aversion |
| Stage 6 | Selective assembly and dimension-chain constraints |
| Stage 7 | Augmented epsilon-constraint Pareto analysis |
| Stage 8 | ALNS plus restricted MILP repair matheuristic |
| Stage 9 | Experiment suite and cross-stage result synthesis |
| Stage 10 | Pair-carbon objective strengthening and shared-capacity extension |
| Stage 11 | Manuscript-ready tables and figures |
| Stage 12 | Final experiment completion, result audit and figure audit |

## Environment

The code is written in Python and uses standard scientific-computing packages.
The exact environment used during development included:

- Python 3.10 or later
- numpy
- pandas
- scipy
- matplotlib

LaTeX manuscript compilation was tested with MiKTeX. The optimisation models use
`scipy.optimize.milp`; no commercial solver is required for the included runs.

Install the core Python dependencies in your preferred environment:

```bash
pip install numpy pandas scipy matplotlib
```

## Quick checks

From the repository root:

```bash
python -m compileall scripts src
python scripts/scan_data_catalogue.py
```

## Reproducing key stages

The main stage runners are under `scripts/`. Typical commands are:

```bash
python scripts/run_stage2_deterministic.py --machine-type CK6150 --period-start T0001 --period-count 52
python scripts/run_stage3_multiperiod.py --machine-type CK6150 --period-start T0001 --period-count 52
python scripts/run_stage4_stochastic.py --machine-type CK6150 --period-start T0001 --period-count 52 --processing-window-periods 8
python scripts/run_stage5_risk_averse.py --machine-type CK6150 --period-start T0001 --period-count 52 --processing-window-periods 8
python scripts/run_stage6_selective_assembly.py --machine-type CK6150 --period-start T0001 --period-count 52 --processing-window-periods 8
python scripts/run_stage7_pareto.py --machine-type CK6150 --period-start T0001 --period-count 52 --processing-window-periods 8 --epsilon-grid-size 5
python scripts/run_stage8_matheuristic.py --machine-type CK6150 --period-start T0001 --period-count 52 --processing-window-periods 8 --epsilon-grid-size 3 --max-iterations 24 --repair-time-limit 20
```

To collect and audit the manuscript evidence:

```bash
python scripts/run_stage9_experiments.py --profile manuscript --execution-mode collect-existing
python scripts/run_stage10_strengthening.py --machine-types CK6150 CK6140 --period-start T0001 --period-count 26
python scripts/run_stage11_paper_artifacts.py --profile manuscript --execution-mode collect-existing
python scripts/run_stage12_finalization.py --profile manuscript --execution-mode complete-and-audit
```

The generated artifacts are written to:

- `data/processed/stage*/`
- `data/results/stage*/`
- `data/results/stage11/tables/`
- `data/results/stage11/figures/`

## Manuscript files

The current manuscript draft and Supplementary Information are in `manuscript/`:

- `ijpr_outline_methods.tex`
- `ijpr_outline_methods.pdf`
- `ijpr_supplementary_information.tex`
- `ijpr_supplementary_information.pdf`
- `references.bib`

The manuscript narrative should be read with the following interpretation:

- Stage 3 and Stage 4 show active recovered-core acceptance and old-part route
  selection.
- Stage 5 and Stage 6 show conservative procurement/backlog fallback under
  stricter reliability, CVaR and selective-assembly screening.
- Zero environmental or assembly-quality objective coordinates in Pareto tables
  are optimisation-coordinate values, not literal zero physical impact or zero
  quality loss.
- Stage 10 pair-carbon and shared-capacity outputs are strengthening and
  supplementary evidence, not replacements for the main Stage 6-8 model.

## Data and GitHub upload notes

The repository includes generated input data and result files. Some CSV/JSON
files are large. In particular, `data/raw/demand_scenarios.csv` can exceed the
100 MB single-file limit for normal GitHub commits.

For GitHub use:

- Upload the full zip as a GitHub Release asset, or
- Use Git LFS for large data files, or
- Keep only a reduced sample dataset in the main branch and place full data in a
  release artifact.

The `dist/` directory is used for generated upload packages and is not required
for running the models.

## Current evidence status

The latest evidence audit classifies the manuscript as ready for Methods and
Results drafting, with no blocking gaps for the current main claims. The main
conclusion is that green machine-tool remanufacturing should be managed as a
risk-aware production planning problem: reuse is valuable when it remains
reliable, assembleable and deliverable under uncertainty.

## Citation

If you use this repository or adapt the modelling framework, please cite the
associated manuscript when it becomes available. The industrial background case
is:

```text
Song et al. (2023). The promotion and application of green remanufacturing:
a case study in a machine tool plant.
```

## License

No open-source licence has been assigned yet. Until a licence is added, all
rights are reserved by the authors.
