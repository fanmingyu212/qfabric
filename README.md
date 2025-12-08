## qFabric

![Python Version](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![License](https://img.shields.io/github/license/fanmingyu212/qfabric)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://fanmingyu212.github.io/qfabric)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![CI](https://github.com/fanmingyu212/qfabric/actions/workflows/ci.yaml/badge.svg)
![Docs](https://github.com/fanmingyu212/qfabric/actions/workflows/docs.yaml/badge.svg)

qFabric is a python package for controlling arbitrary waveform generators (AWGs) using a universal interface designed for quantum physics experiments.

It offers:
* One unified control interface across AWGs of different manufacturers and models.
* Serializable pulse sequence abstraction without AWG-specific details.
* Interactive visualization of pulse sequences.
* Efficient AWG programming maximizing its capability and minimizing experiment downtime.
* Extensible with new AWGs and pulse types.

### Who is qFabric for?

* AMO / quantum-optics labs using AWGs to control their experiment with hardware timing.
* Scientists who want to write physics-oriented pulse sequences without thinking about AWG-specific details.
* In particular, we write it with the following goals:
    - The same pulse sequence can be shared between experiments using different hardware.
    - Users can easily show a timing diagram of the pulse sequence and share it with collaborators.
* However, it is not designed for RTIO systems (e.g. ARTIQ/Sinara, Quantum Machines) where the pulse sequence dynamically changes with input events.

### Example (a mock Ramsey phase scan experiment)

```python
import numpy as np

from qfabric import AnalogSequence, ExperimentManager, Sequence, SineWave, Step

ramsey_channel = 0
f_carrier = 1e6
amplitude = 1
tau = 1e-3
wait_time = 1e-2
phase_diffs = np.linspace(0, 2 * np.pi, 10, endpoint=False)  # Ramsey phase scan.

# sequence definition
sequences = []
for phase in phase_diffs:
    ramsey_sequence = Sequence()
    step = Step("ramsey")
    ramsey_func = AnalogSequence()
    piov2_pulse = SineWave(f_carrier, amplitude, stop_time=tau)  # first pi/2 pulse.
    ramsey_func.add_function(piov2_pulse, coherent_phase=True)
    piov2_pulse_phase_offsetted = SineWave(f_carrier, amplitude, phase=phase, stop_time=tau)
    ramsey_func.add_function(
        piov2_pulse_phase_offsetted, delay_time_after_previous=wait_time, coherent_phase=True
    )  # second pi/2 pulse.
    step.add_analog_function(ramsey_func)
    ramsey_sequence.add_step(step)
    sequences.append(ramsey_sequence)

# execute sequences
manager = ExperimentManager(config_file_path)  # config file for the AWG.
manager.schedule(sequences, repeats=1)
manager.setup()
while len(manager.scheduled_sequence_indices) > 0:
    next_sequence = manager.program_next_sequence()
    manager.run()
```

### Installation

To use `qFabric`, run `pip install git+https://github.com/fanmingyu212/qfabric`.

If you want to develop `qFabric`, you can fork this project, clone the fork, navigate to the unzipped folder (containing `pyproject.toml`), and run `pip install -e .[dev,docs]` to install it in the editable mode with development dependencies.

### [Documentation](https://fanmingyu212.github.io/qfabric/)

### Beta test

AWG implementations are still being tested and may change. Estimated first stable release date is Jan 2026.
