Experiment Manager
=====================

In this page, we outline the structure for the components
of :class:`~qfabric.manager.experiment_manager.ExperimentManager`.

The :class:`~qfabric.manager.experiment_manager.ExperimentManager` class is the control entry point for users.
It manages experiment sequences (:class:`~qfabric.sequence.sequence.Sequence`) that are scheduled,
converts sequences to AWG data, and programs the data onto AWGs, and executes the sequences.

Config
--------------------
All AWG-specific information is defined in the config TOML file used to build the
:class:`~qfabric.manager.experiment_manager.ExperimentManager`.
The config file is parsed and validated by :meth:`~qfabric.manager.config.load_hardware_config`.

Planner
-------------
:class:`~qfabric.planner.planner.Planner` handles scheduling experiment sequences and preparation for these sequences.
It is defined as the ``planner`` attribute in :class:`~qfabric.manager.experiment_manager.ExperimentManager`.

It saves a list of sequences that are scheduled, and controls the iteration through each sequence.
It talks to the ``Programmer`` (see below) to program the AWG memory and segment steps when needed.
It also uses segmenters (:class:`~qfabric.planner.segmenter.Segmenter`) to convert the functions
defined in the sequences to AWG data.

Segmenter
^^^^^^^^^^^^^
Each AWG instrument has a corresponding :class:`~qfabric.planner.segmenter.Segmenter` object, defined as an element of the
``_segmenters`` attribute in :class:`~qfabric.planner.planner.Planner`.

The segmenter converts functions in each :class:`~qfabric.sequence.step.Step` of the sequence to
:class:`~qfabric.planner.segmenter.Segment` objects that stores data recognizable by the AWG instrument that it controls.

Programmer
-------------
:class:`~qfabric.programmer.programmer.Programmer` handles programming of the AWGs.
It is defined as the ``programmer`` attribute in :class:`~qfabric.manager.experiment_manager.ExperimentManager`.

It uses devices (:class:`~qfabric.programmer.device.Device`) to communicate with each AWG.

Device
^^^^^^^^^^^^^
Each AWG instrument has a corresponding :class:`~qfabric.programmer.device.Device` object, defined as an element of the
``devices`` attribute in :class:`~qfabric.programmer.programmer.Programmer`.

The device class defines AWG protocol for operations such as load AWG data, set steps and repeats, start, and stop.
Typically the device class uses a driver / header class that implements the AWG protocol.

The :class:`~qfabric.programmer.device.Device` class and the :class:`~qfabric.planner.segmenter.Segmenter` class must be
designed in tandem to correctly operate the AWG.

Public methods of the :class:`~qfabric.manager.experiment_manager.ExperimentManager` class
-----------------------------------------------------------------------------------------------
:meth:`~qfabric.manager.experiment_manager.ExperimentManager.schedule`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calls ``Planner.schedule()``, which adds the sequences to the schedule list.

:meth:`~qfabric.manager.experiment_manager.ExperimentManager.setup`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calls ``Planner.setup()``, which generates the :class:`~qfabric.planner.segmenter.Segment` objects.

:meth:`~qfabric.manager.experiment_manager.ExperimentManager.program_next_sequence`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calls ``Planner.program_next_sequence()``, which programs the AWG memory via
``Programmer.program_memory_single_device()``, and programs the AWG segment steps
via ``Programmer.program_segment_step_single_device()``.

:meth:`~qfabric.manager.experiment_manager.ExperimentManager.run`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calls ``Programmer.run()``. Executes the sequence.

:meth:`~qfabric.manager.experiment_manager.ExperimentManager.wait_until_complete`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calls ``Programmer.wait_until_complete()``. Waits for the sequence to complete execution.

:meth:`~qfabric.manager.experiment_manager.ExperimentManager.set_principal_device_trigger`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calls ``Programmer.set_principal_device_trigger()``. Sets the trigger state on the principal device.








