Quickstart
==================

First, set up the config file for the AWG devices installed (See :doc:`../how_to/how_to_config`).
If you just want to test the package without using a physical AWG, use the following config.
This config simply plots the AWG data. ``matplotlib`` is needed.

.. code-block:: toml

   [config]
   version = 1
   
   [[awgs]]
   name = "mock_1"
   
   segmenter_module = "qfabric.planner.segmenter.mock"
   segmenter_class = "MockSegmenter"
   device_module = "qfabric.programmer.device.mock"
   device_class = "MockDevice"
   
     [awgs.segmenter_config]
     analog_channels = [0, 1]
     digital_channels = [0]
     sample_rate = 1e6
   
     [awgs.device_config]
     resource = "mock_1"
     principal_device = true
     show_plot = true

Also, write a pulse sequence that you would like to run (See :doc:`sequence`). An example sequence is shown below:

.. code-block:: python
    
    from qfabric import Sequence, Step, LinearRamp, SineWave, DigitalOn, ConstantAnalog, AnalogSequence

    sequence = Sequence()
    step_1 = Step("step 1")
    step_1.add_analog_function(0, SineWave(frequency=1e5, amplitude=1))
    analog_ch1 = AnalogSequence()
    analog_ch1.add_function(LinearRamp(start_amplitude=0, stop_amplitude=1, start_time=0, stop_time=5e-5))
    analog_ch1.add_function(LinearRamp(start_amplitude=1, stop_amplitude=0, start_time=0, stop_time=5e-5))
    step_1.add_analog_function(1, analog_ch1)
    sequence.add_step(step_1)

    step_2 = Step("step 2")
    step_2.duration = 5e-5
    step_2.add_analog_function(1, ConstantAnalog(0.5))
    step_2.add_digital_function(0, DigitalOn())
    sequence.add_step(step_2, delay_time_after_previous=5e-5)

    step_3 = Step("step 3")
    step_3.duration = 5e-5
    step_3.add_analog_function(0, SineWave(2e5, 1, start_time=2e-5))
    sequence.add_step(step_3, repeats=5)

Finally, define the manager, and schedule and run the sequence using the manager.

.. code-block:: python
    
    from qfabric import ExperimentManager

    manager = ExperimentManager(config_path)  # define config_path before this line.
    manager.schedule(sequence)
    manager.setup()
    while len(manager.scheduled_sequence_indices) > 0:
        manager.program_next_sequence()
        manager.run(wait_for_finish=True)

All the above code is included in ``examples/quick_start``.

If you have multiple different sequences to run, you can schedule them with any order or repeat numbers,
set up all of them, and execute each of them.
