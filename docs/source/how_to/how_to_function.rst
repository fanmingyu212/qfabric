How to write new analog or digital functions
==============================================

Analog and digital functions are the most basic building blocks for a sequence.
A collection of common analog and digital functions are included in
:mod:`qfabric.sequence.function` and :mod:`qfabric.sequence.basic_functions`, which can be directly
imported from :doc:`../api/api_qfabric`.
This page shows how to add more functions that you may need.

Add a digital function
----------------------------
Digital functions should inherit from the :class:`~qfabric.sequence.function.DigitalFunction` class.
All digital function classes are automatically made to be a ``dataclass``. Attributes of the function
that affects the output in any way should therefore be included as a ``field`` of the class. These
attributes are compared when determining if two functions are identical. Such comparisons are used
to check if AWG segments are identical for reducing AWG memory use.

For example, we want to add a digital pulse width modulation (PWM) function that periodically turns on and off,
with cycle time and duty cycle set by the users.

.. code-block:: python

    import numpy as np
    import numpy.typing as npt

    from qfabric.sequence.function import DigitalFunction

    class DigitalPWM(DigitalFunction):
        """Digital pulse width modulation function."""
        # These fields are critical for function comparison.
        period: float
        duty_cycle: float
        start_time: float
        stop_time: float

        def __init__(
            self,
            period: float,
            duty_cycle: float,
            start_time: float = None,
            stop_time: float = None,
        ):
            if duty_cycle < 0 or duty_cycle > 1:
                raise ValueError("Duty cycle must be between 0 to 1.")
            self.period = period
            self.duty_cycle = duty_cycle
            self.start_time = start_time
            self.stop_time = stop_time

        @property
        def min_duration(self) -> float:
            # if stop_time is defined, the sequence is at least stop_time long.
            if self.stop_time is not None:
                # otherwise, check if start_time is defined.
                if self.start_time is None:
                    return 0
                else:
                    return self.start_time
            return self.stop_time

        def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.bool]:
            # use np.piecewise to define the function.
            condlist = []
            funclist = []
            if self.start_time is not None:
                condlist.append(times < self.start_time)
                funclist.append(False)
            if self.stop_time is not None:
                condlist.append(times >= self.stop_time)
                funclist.append(False)
            
            def pwm(ts):
                remainders = np.remainder(ts, self.period)
                time_to_transition = self.duty_cycle * self.period
                return remainders < time_to_transition
            funclist.append(pwm)

            if len(condlist) == 0:
                return pwm(times)
            else:
                return np.piecewise(times, condlist, funclist)


Add an analog function
----------------------------
Analog functions should inherit from the :class:`~qfabric.sequence.function.AnalogFunction` class.
All analog function classes are automatically made to be a ``dataclass``. Attributes of the function
that affects the output in any way should therefore be included as a ``field`` of the class. These
attributes are compared when determining if two functions are identical. Such comparisons are used
to check if AWG segments are identical for reducing AWG memory use.

When writing periodic analog functions with phase a parameter, be careful about the ``time_offset``
parameter in :meth:`~qfabric.sequence.function.AnalogFunction.output`. This time offset is for phase
coherence between multiple analog functions concatenated in time using :class:`~qfabric.sequence.function.AnalogSequence`.
The phase of the output result should be forwarded by 2πf x ``time_offset``.

Here we use a sine function as an example. This is the same code as :meth:`~qfabric.SineWave`,
but with additional comments for clarity.

.. code-block:: python

    class SineWave(AnalogFunction):
        """
        Sine wave.

        Args:
            frequency (float): Cyclic frequency of the sine wave.
            amplitude (float): Amplitude of the sine wave.
            phase (float): Phase of the sine wave. Default 0.
            start_time (float):
                Start time. If None, it starts from the beginning of the step. Default None.
            stop_time (float):
                Stop time. If None, it stops at the end of the step. Default None.
        """

        # all attributes should be defined here as fields.
        frequency: float
        amplitude: float
        phase: float
        start_time: float
        stop_time: float

        def __init__(
            self,
            frequency: float,
            amplitude: float,
            phase: float = 0,
            start_time: float = None,
            stop_time: float = None,
        ):
            self.frequency: float = frequency
            self.amplitude: float = amplitude
            self.phase: float = phase
            self.start_time: float = start_time
            self.stop_time: float = stop_time

        @property
        def min_duration(self) -> float:
            # if stop_time is defined, the sequence is at least stop_time long.
            if self.stop_time is not None:
                # otherwise, check if start_time is defined.
                if self.start_time is None:
                    return 0
                else:
                    return self.start_time
            else:
                return self.stop_time

        def output(
            self, times: npt.NDArray[np.float64], time_offset: float = 0
        ) -> npt.NDArray[np.float64]:
            # use np.piecewise to define the function.
            condlist = []
            funclist = []
            # the start and stop time comparison (which determines the envelope)
            # should not use the time_offset.
            if self.start_time is not None:
                condlist.append(times < self.start_time)
                funclist.append(0)
            if self.stop_time is not None:
                condlist.append(times >= self.stop_time)
                funclist.append(0)

            def sine(ts):
                # here the phase is offset by the time of `time_offset`.
                inst_phases = 2 * np.pi * self.frequency * (ts + time_offset) + self.phase
                return self.amplitude * np.sin(inst_phases)

            funclist.append(sine)
            # np.piecewise does not support zero-length condlist.
            if len(condlist) == 0:
                return sine(times)
            else:
                return np.piecewise(times, condlist, funclist)

Function with non-comparison attributes
-------------------------------------------

Either ``DigitalFunction`` or ``AnalogFunction`` subclasses forbid definition of public or private attributes
that are not dataclass fields. This ensures that all attributes that affect the function output are defined
as fields, so function comparisons are guaranteed to be correct (the output does not change between the two functions).

To write a function with non-comparison attributes, define them as ``field(compare=False)``. This is rarely needed,
but it could be helpful in some cases (mostly for removing repeated code executions). For example, for an analog function below
that has three constant parts of variable times:

.. code-block:: python

    from dataclasses import field

    class FunctionWithNonComparisonAttribute(AnalogFunction):

        time_1: float
        time_2: float
        time_3: float
        voltage_1: float
        voltage_2: float
        voltage_3: float
        # above fields are used in comparison.
        _total_time: float = field(compare=False)
        # this field is not used in comparison.
        # In this case, the field can be compared: if time_1, time_2, and time_3 are the same,
        # _total_time is also the same.
        # However this allows implementing other attributes which change during function
        # execution, but do not affect the function output.

        def __init__(
            self,
            time_1: float,
            time_2: float,
            time_3: float,
            voltage_1: float,
            voltage_2: float,
            voltage_3: float,
        ):
            self.time_1 = time_1
            self.time_2 = time_2
            self.time_3 = time_3
            self.voltage_1 = voltage_1
            self.voltage_2 = voltage_2
            self.voltage_3 = voltage_3
            self._total_time = time_1 + time_2 + time_3

        @property
        def min_duration(self) -> float:
            # does not need to add time_1, time_2, time_3 up every time.
            return self._total_time

        ...

Without a field definition, it raises an ``AttributeError`` when setting any attribute.

Summary
-----------------
To write a new function, always check the following points:

* Must inherit from ``DigitalFunction`` or ``AnalogFunction``.
* All attributes that affect the output must be defined as fields with ``compare=True`` (default).
* All attributes that do not affect the output must be defined as fields with ``compare=False``.
* The ``min_duration()`` and ``output()`` methods must be implemented.
* For an analog function with phase, always forward the phase with ``time_offset``. However, do not shift the pulse envelope in time.
