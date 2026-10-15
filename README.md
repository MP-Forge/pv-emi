# pv-emi

Measurement-based EMI modeling of the DC side of photovoltaic (PV) systems, in Python.

`pv-emi` represents the power converter as an equivalent noise source (a Norton current source with its source impedance), and the rest of the system (PV panel, DC cable, EMI filter, LISN) as frequency-domain impedances and two-port networks. Both are extracted from in-circuit measurements taken while the system is operating, with separate models for common-mode (CM) and differential-mode (DM) noise. The model then predicts the emissions of new configurations, such as adding a filter or using a longer cable, without measuring each one.

This package is a Python port of the original MATLAB implementation of the method described under [Background](#background).

> **Status:** early development (v0.1). The API will change.

## Features

Available now:

- Frequency grids, complex impedances, and ABCD two-port networks with cascading and de-embedding
- Loading measured impedance data from CSV (magnitude and phase) and MATLAB `.mat` files
- Extracting the CM or DM equivalent noise-source current from a measured noise current
- Predicting the conducted noise current into a load, with an optional series impedance

Planned:

- EMI filters and long DC cables in the prediction, using two-port networks
- Radiated-emission estimation from the antenna factor of the PV panel and cable
- Automated in-circuit measurements with a vector network analyzer

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/MP-Forge/pv-emi.git
cd pv-emi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Example

Extract a CM noise source from a measured current, then predict the current after adding a CM choke:

```python
import numpy as np

from pv_emi.models.conducted_emission import predict_cm_conducted_emission
from pv_emi.models.noise_source import estimate_cm_noise_source_current

f = np.array([150e3, 1e6, 10e6])  # frequency, Hz
i_measured = np.array([1e-3, 5e-4, 1e-4])  # CM current measured at the PV side, A
z_source = 1 / (2j * np.pi * f * 1e-9)  # CM noise-source impedance (about 1 nF), ohm
z_pv = 1 / (2j * np.pi * f * 5e-9)  # CM impedance of the PV panel (about 5 nF), ohm

source = estimate_cm_noise_source_current(f, i_measured, z_source, z_pv)

z_choke = 2j * np.pi * f * 10e-3  # 10 mH CM choke in series
result = predict_cm_conducted_emission(f, source.current_a, z_source, z_pv, z_choke)

print(result.current_magnitude_dbua)  # predicted CM current, dBµA
# [ 43.87   3.66 -50.34]
```

## Development

```bash
pytest            # run the tests
ruff check .      # lint
ruff format .     # format the code
```

## Project layout

```
src/pv_emi/
├── core/       frequency grid, impedance, noise source
├── network/    ABCD two-port networks
├── io/         loaders for measured data
└── models/     noise-source extraction and conducted-emission prediction
tests/          unit tests
```

## Background

1. M. Prajapati and K. Y. See, "Extraction of equivalent noise source model from photovoltaic systems," *IEEE Transactions on Electromagnetic Compatibility*, vol. 61, no. 3, pp. 903–910, 2019. [doi:10.1109/TEMC.2018.2836665](https://doi.org/10.1109/TEMC.2018.2836665)
2. M. Prajapati, F. Fan, Z. Zhao, and K. Y. See, "Estimation of radiated emissions from PV system through black box approach," *IEEE Transactions on Instrumentation and Measurement*, vol. 70, 2021, Art. no. 9004304. [doi:10.1109/TIM.2021.3102752](https://doi.org/10.1109/TIM.2021.3102752)

## License

No license has been chosen yet. Until a `LICENSE` file is added, all rights are reserved.
