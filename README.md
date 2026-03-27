# sysid_pipeline

A System Identification (SysID) pipeline for ArduPilot-based drones. It automates flight data acquisition in SITL, processes the data, and estimates inertia parameters by analyzing excitation maneuvers (doublets).

## Main Scripts

- **`scripts/fly_doublets.py`**: Connects to SITL (ArduCopter) via MAVLink and automatically executes doublet maneuvers on the three axes (roll, pitch, yaw) to generate excitation data.
- **`scripts/verify_bin.py`**: Analyzes and validates the generated `.BIN` logs. It checks the required message rates (IMU, ATT, RCOU, and velocity estimates like XKF1/NKF1) and the excitation quality to ensure the data is suitable for SysID.
- **`scripts/extractor.py`**: Parses the `.BIN` DataFlash log, extracting IMU, ATT, and motor commands (RCOU), and time-aligns them at a constant frequency into a single `data/synchronized_flight_data.csv` file.
- **`scripts/sysid_optimizer.py`**: Performs vector optimization and data fitting (using Least Squares). It isolates excited flight segments to decouple and compute the inertia on the three axes by comparing the predicted and measured angular accelerations.
- **`scripts/generate_figures.py`**: Generates publication-ready figures for technical reports or the GSoC proposal (e.g., `figures/flight_data_overview.png`, `figures/excitation_segments.png`), illustrating sensor records, motor inputs, and per-axis variance.

## Workflow

1. Start SITL and run `python scripts/fly_doublets.py` to collect a new DataFlash flight log with doublet excitations.
2. Run `python scripts/verify_bin.py <log.BIN>` to verify that the log contains sufficient excitations and to extract the analyzed data into a `.csv` file.
3. (Alternatively) Extract the raw data using `python scripts/extractor.py <log.BIN>`.
4. Run `python scripts/sysid_optimizer.py <data.csv>` to obtain the estimated inertia parameters along with a graphical validation of the prediction.
5. To produce reporting visualizations only, run `python scripts/generate_figures.py <data.csv>`.