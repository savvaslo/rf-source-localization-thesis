# Simulation-Based Multi-Receiver RF Source Localization

Companion source-code package for the University of Cyprus diploma thesis
*Simulation-Based Multi-Receiver RF Source Localization Using Direction-of-Arrival Measurements* (2026).

This package contains the two Python desktop applications, the automated experiment
runner, and the final **aggregate** experiment manifest and summary. It describes a
**synthetic measurement-level simulation**; it does not include or claim physical
KrakenSDR measurements.

## Repository structure

| Path | Purpose |
| --- | --- |
| `KrakenSDR_Simulator-main/simulator_qt.py` | Virtual RF receiver/target configuration, target routes, synthetic DoA/RSSI generation and UDP packets. |
| `KrakenSDR_Simulator-main/simulator_gui.py` | Desktop GUI entry point for the simulator. |
| `KrakenScope/main.py` | KrakenScope GUI, UDP reception, cycle management, visualization and localization workflow. |
| `KrakenScope/core.py` | Bearing geometry, robust stationary estimation, automatic mode inference and bearing-only EKF. |
| `KrakenScope/run_experiments.py` | Headless seeded scenario runner, summary statistics, optional per-run CSV and plots. |
| `KrakenScope/plot_experiment.py` | Plotting of exported per-run trajectory and error data. |
| `KrakenScope/test_core.py` and `KrakenSDR_Simulator-main/test_simulator_qt.py` | Selected automated unit tests. |
| `results/manifest.json`, `results/summary.md` | Archived parameters and **aggregate results** reported for the final 70-run evaluation. |

## Requirements

Use Python 3.12 with the dependencies declared in the two `requirements.txt` files.
A typical setup from the repository root is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
python -m pip install -r KrakenScope/requirements.txt
python -m pip install -r KrakenSDR_Simulator-main/requirements.txt
python -m pip install pytest
```

For interactive demonstrations, open separate terminals in the same environment:

```bash
python KrakenScope/main.py
python KrakenSDR_Simulator-main/simulator_gui.py
```

Configure two simulated receivers targeting `127.0.0.1:4200`, and enable the
KrakenScope UDP listener. These interactive demonstrations are not part of the
archived final 70-run suite.

For the automated experiments:

```bash
cd KrakenScope
python run_experiments.py --list
python run_experiments.py --repeats 10 --seed 20261201
```

**Important:** The runner's default seed differs from the final base seed, so
`--seed 20261201` is required to match the configuration in the archived manifest.
The scenario order and software dependencies must also remain unchanged. The
runner directly calls the simulation and localization functions (no UDP transport).

## Availability and verification scope

The supplied `results/summary.md` and `results/manifest.json` are the available
aggregate final experiment records. **The original 70 per-run CSV exports are not
included.** Re-executing this source may generate new per-run data, but the archived
aggregate statistics have not been independently recomputed in this package.
No physical hardware or real RF propagation validation is claimed.

This source package was prepared from the files supplied for the thesis. Before
publishing the repository, confirm it contains the intended final application
revisions and that you have permission to distribute each included source file.
No open-source license has been selected or included in this draft package.
