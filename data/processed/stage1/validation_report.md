# Stage 1 Validation Report

Generated at UTC: `2026-06-16T11:20:45.380931+00:00`

## Summary

- Files scanned: `28` / `28`
- Total rows: `273013`
- Total size: `353.13 MB`
- Passed checks: `161`
- Warnings: `1`
- Failed checks: `0`

## Catalogue

| File | Rows | Columns | Size MB | SHA256 prefix |
|---|---:|---:|---:|---|
| `assembly_candidates.csv` | 19876 | 78 | 18.05 | `87050da25ab1` |
| `assembly_compatibility.csv` | 14110 | 88 | 15.73 | `ce3bba6a6824` |
| `assembly_requirements.csv` | 84 | 66 | 0.11 | `4a2119330c8d` |
| `baseline_rules.csv` | 22 | 86 | 0.04 | `6df2185ac9f2` |
| `bom.csv` | 140 | 36 | 0.06 | `4f0df027c531` |
| `capacity_calendar.csv` | 14758 | 54 | 6.04 | `f87dd1c310b8` |
| `component_inspection.csv` | 8852 | 61 | 5.25 | `a63c8abdcb3d` |
| `component_quality_scenarios.csv` | 3564 | 94 | 3.87 | `5fd9c920b83b` |
| `demand_scenarios.csv` | 127170 | 106 | 186.62 | `7e39b2b0ec20` |
| `environmental_parameters.csv` | 171 | 69 | 0.2 | `5774215bd140` |
| `historical_performance.csv` | 23760 | 124 | 38.37 | `18f8821981e5` |
| `initial_inventory.csv` | 263 | 78 | 0.22 | `021e2520e741` |
| `machine_types.csv` | 15 | 50 | 0.01 | `838532cf8202` |
| `machines.csv` | 47 | 49 | 0.03 | `e3d9a37b3a1b` |
| `orders.csv` | 607 | 49 | 0.35 | `2262ec50163a` |
| `processing_parameters.csv` | 2908 | 73 | 2.03 | `d74111455a87` |
| `procurement_parameters.csv` | 474 | 79 | 0.38 | `c84a255e424a` |
| `quality_states.csv` | 4 | 47 | 0.0 | `f78624d20282` |
| `reliability_parameters.csv` | 1173 | 79 | 1.3 | `eab99dd21e62` |
| `returned_cores.csv` | 1232 | 50 | 0.64 | `74235af5dced` |
| `risk_parameters.csv` | 10557 | 59 | 11.35 | `75689ccf5a63` |
| `route_feasibility.csv` | 868 | 55 | 0.58 | `af1611a3ac4d` |
| `route_operations.csv` | 57 | 50 | 0.03 | `bd3a5e6a4051` |
| `route_outcome_scenarios.csv` | 40780 | 97 | 60.92 | `16db978128fb` |
| `route_state_transition.csv` | 1173 | 46 | 0.78 | `1bc1bd848dac` |
| `routes.csv` | 7 | 69 | 0.01 | `07f3963b5492` |
| `scenarios.csv` | 27 | 105 | 0.04 | `4a4e748412a5` |
| `time_periods.csv` | 314 | 29 | 0.12 | `40c4d58c3fdc` |

## Warnings And Failures

| Severity | Check | File | Message | Observed | Expected |
|---|---|---|---|---|---|
| warning | route_transition_probability_sum | `route_state_transition.csv` | Transition probability sums are within failure tolerance 1e-05, but 163 rows exceed warning tolerance 1e-06. | max_deviation=2.0000000000575113e-06 | <= 1e-05 |
