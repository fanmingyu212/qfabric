Sequence
==================

Each experiment sequence is described by an object of the :class:`~qfabric.sequence.sequence.Sequence` class.

A sequence is consisted of multiple :class:`~qfabric.sequence.step.Step` objects.
Use :meth:`~qfabric.sequence.step.Step.add_analog_function` or :meth:`~qfabric.sequence.step.Step.add_digital_function`
to add analog or digital functions to different channels of the sequence.
The user-added steps are preceded by a short :class:`~qfabric.sequence.step.StartStep` and followed by a short
:class:`~qfabric.sequence.step.StopStep` when programmed on the AWG.
During the :class:`~qfabric.sequence.step.StartStep`, a digital channel of an principal AWG device can be turned on
to trigger other AWG devices controlled by the principal device. See the ``start_step_digital_channel_on`` parameter
of :class:`~qfabric.sequence.sequence.Sequence`.

Each channel corresponds to an analog / digital output of an AWG device.
The duration of the step is defined by the longest function included in this step, by default.
To override, set :meth:`~qfabric.sequence.step.Step.duration`.

AWG outputs within a :class:`~qfabric.sequence.step.Step` should have exact time and phase offsets as defined
in the analog and digital functions. However, the time or phase relationship between different steps are not guarenteed.
To satisfy AWG hardware requirements, it could be possible that a step needs to be extended (usually by a small amount).
Therefore, the output pulses between steps may not have the same time offset as defined in the code.
