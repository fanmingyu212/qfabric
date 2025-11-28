Add support for a different AWG
=========================================

Prerequisites: :doc:`../developers/manager_components`. 

If your AWG is not supported by qFabric, you can add support for it
by writing device-compatible subclasses of :class:`~qfabric.planner.segmenter.Segmenter`
and :class:`~qfabric.programmer.device.Device`.

It is important to read the AWG manual carefully to understand the operation of the AWG device.
Here, we consider an example AWG device with the following specifications:

* 2 analog channels, 2 digital channels
* Fixed sample rate of 100 Msamples/s
* AWG supports the sequence mode, where we can define data blocks to stored in the AWG memory, executed with defined orders and repeats.
* Each block can contain up to 1 million samples, with maximum 100 blocks available.
* Minimum block length is 1000 samples, with valid block length a multiple of 8.

Segmenter
-----------------
Each AWG should have its own Segmenter class, inheriting :class:`~qfabric.planner.segmenter.Segmenter`.
Before defining the segmenter class, we define a ``ExampleBlock`` class representing
a single block of memory to be programmed on the AWG.


.. code-block:: python

    import numpy.typing as npt

    class ExampleBlock:
        """Represent AWG memory data blocks."""
        def __init__(self, analog_data: npt.NDArray[np.float64], digital_data: npt.NDArray[np.bool]):
            self.analog_data = analog_data
            self.digital_data = digital_data

We also define an ``ExampleSegment`` class, representing AWG data for a single step of the sequence.
The segment is split into ``ExampleBlock`` objects.

.. code-block:: python

    import numpy as np
    from qfabric.sequence.function import AnalogEmpty, DigitalEmpty
    from qfabric.sequence.step import DeviceStep
    from qfabric.planner.segmenter import Segment

    SAMPLE_RATE = int(100e6)
    MIN_BLOCK_LENGTH = 1000
    MAX_BLOCK_LENGTH = 1000000

    def get_allowed_sample_length(nominal_sample_length: int) -> int:
        """Gets the minimum allowed sample length longer than the nominal length."""
        if nominal_sample_length < MIN_SAMPLE_LENGTH:
            return MIN_SAMPLE_LENGTH
        elif nominal_sample_length > MAX_BLOCK_LENGTH:
            raise ValueError(f"Sample length cannot be longer than {MAX_BLOCK_LENGTH}")
        else:
            remainder = nominal_sample_length // 8
            if remainder == 0:
                return nominal_sample_length
            else:
                return nominal_sample_length + 8 - remainder


    class ExampleSegment(Segment):
        """AWG data for a device step, may contain multiple ExampleBlocks"""
        def __init__(self, device_step: DeviceStep, analog_channels: list[int], digital_channels: list[int]):
            super().__init__(device_step, analog_channels, digital_channels)
            sample_length, analog_data, digital_data = self._get_analog_and_digital_data(
                analog_channels, digital_channels
            )
            self.blocks: list[ExampleBlock] = self._split_to_blocks(sample_length, analog_data, digital_data)

        def __eq__(self, other: "ExampleSegment") -> bool:
            """For comparing segments to check if they contain the same content."""
            if self._device_step != other._device_step:
                return False
            return True

        def _get_analog_and_digital_data(
            self, analog_channels: list[int], digital_channels: list[int]
        ) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.bool]]:
            """Gets analog and digital data of the entire device step."""
            nominal_duration = device_step.duration
            nominal_sample_length = int(SAMPLE_RATE * nominal_duration)
            sample_length = get_allowed_sample_length(nominal_sample_length)

            times = np.arange(len(sample_length)) / SAMPLE_RATE
            analog_data = []
            for analog_channel in analog_channels:
                analog_function = self._device_step.analog_functions.get(
                    analog_channel, AnalogEmpty()
                )
                analog_data.append(analog_function.output(times))
            analog_data = np.transpose(analog_data)  # axis 0 is sample index.

            digital_data = []
            for digital_channel in digital_channels:
                digital_function = self._device_step.digital_functions.get(
                    digital_channel, DigitalEmpty()
                )
                digital_data.append(digital_function.output(times))
            digital_data = np.transpose(digital_data)  # axis 0 is sample index.

            return (sample_length, analog_data, digital_data)

        def _split_to_blocks(
            self,
            sample_length: int,
            analog_data: npt.NDArray[np.float64],
            digital_data: npt.NDArray[np.bool],
        ) -> list[ExampleBlock]:
            """Splits data into ExampleBlocks"""
            block_boundary_indices = [
                block_index * MAX_BLOCK_LENGTH for MAX_BLOCK_LENGTH in range(sample_length // MAX_BLOCK_LENGTH)
            ]
            block_boundary_indices += [sample_length]

            # checks if the last block is too short
            if len(block_boundary_indices) > 2:
                if block_boundary_indices[-1] - block_boundary_indices[-2] < MIN_BLOCK_LENGTH:
                    block_boundary_indices[-2] = block_boundary_indices[-1] - 1000

            blocks: list[ExampleBlock] = []
            for block_index in range(len(block_boundary_indices) - 1):
                start_sample_index = block_boundary_indices[block_index]
                stop_sample_index = block_boundary_indices[block_index + 1]
                blocks.append(
                    ExampleBlock(
                        analog_data=analog_data[start_sample_index:stop_sample_index],
                        digital_data=digital_data[start_sample_index:stop_sample_index],
                    )
                )
            return blocks

After that we can define the segmenter class which handles converting steps to segments.
In this class, we need to implement :meth:`~qfabric.planner.segmenter.Segmenter.set_steps`
and :meth:`~qfabric.planner.segmenter.Segmenter.get_awg_memory_data`.

.. code-block:: python

    from qfabric.planner.segmenter import Segmenter
    
    MAX_BLOCK_NUMBER = 100

    class ExampleSegmenter(Segmenter):
        def __init__(self, analog_channels: list[int], digital_channels: list[int]):
            super().__init__(analog_channels, digital_channels)
            if len(analog_channels) != 2:
                raise ValueError("Must have 2 analog channels.")
            if len(digital_channels) != 2:
                raise ValueError("Must have 2 digital channels.")

        def set_steps(self, steps: list[Step], sequence_to_steps_map: dict[int, list[int]]):
            super().set_steps(steps, sequence_to_steps_map)
            # self._device_steps and self._sequence_to_device_steps_map are defined by above.
            self._device_steps_to_segments()
            self._get_sequence_to_segments_map()

        def _device_steps_to_segments(self):
            """Builds segments from device steps."""
            self._segments: list[ExampleSegment] = []
            self._device_step_to_segment_map: dict[int, int] = {}
            for device_step_index, device_step in enumerate(self._device_steps):
                segment = ExampleSegment(
                    device_step, self._analog_channels, self._digital_channels
                )
                # checks for duplicates in segments.
                try:
                    segment_index = self._segments.index(segment)
                except ValueError:
                    self._segments.append(segment)
                    segment_index = len(self._segments) - 1
                
                # this is a mapping from device step indices to segment indices.
                self._device_step_to_segment_map[device_step_index] = segment_index

        def _get_sequence_to_segments_map(self):
            """Generates mapping from sequence indices to segment indices"""
            self._sequence_to_segments_map: dict[int, list[int]] = {}
            for sequence_index in self._sequence_to_device_steps_map:
                self._sequence_to_segments_map[sequence_index] = []
                for device_step_index in self._sequence_to_device_steps_map[sequence_index]:
                    self._sequence_to_segments_map[sequence_index].append(
                        self._device_step_to_segment_map[device_step_index]
                    )

        def get_awg_memory_data(
            self, sequence_indices: list[int]
        ) -> tuple[dict[str, list[ExampleSegment]], dict[int, int], list[int]]:
            # list of step indices that is used in the sequences requested
            step_indices: list[int] = []
            for sequence_index in sequence_indices:
                step_indices.extend(self._sequence_to_device_steps_map[sequence_index])
            # removes duplicates
            step_indices = list(dict.fromkeys(step_indices))

            # segments to be programmed
            segments: list[ExampleSegment] = []
            # mapping from step indices to indices in the above segments list.
            step_to_segment_map: dict[int, int] = {}
            for step_index in step_indices:
                segment = self._segments[self._device_step_to_segment_map[step_index]]
                # check for duplicates.
                try:
                    segment_index = segments.index(segment)
                except ValueError:
                    segments.append(segment)
                    segment_index = len(segments) - 1
                step_to_segment_map[step_index] = segment_index

            # check if there are too many blocks to program
            block_count = 0
            for segment in segments:
                block_count += len(segment.blocks)
            if block_count > MAX_BLOCK_NUMBER:
                # if there are too many blocks, attempt to program less sequences.
                if len(sequence_indices) > 1:
                    return self.get_awg_memory_data(sequence_indices[:-1])
                else:
                    raise RuntimeError("Sequence is too complex to be programmed in this AWG.")

            # this is the minimum amount of data to program the AWG.
            # if the AWG needs more data, it can be added as long as the Device class is compatible.
            awg_data = {"segments": segments}
            return awg_data, step_to_segment_map, sequence_indices


The above segmenter satisfies the specifications of the AWG listed above.
It can be made more AWG memory efficient, by checking if there are ``ExampleBlock``
contain the same data. If they do, they can be replaced by a single AWG memory block.

However, the above example shows what is generally needed in a segmenter.

Device
-----------------
The device class, inherting from :class:`~qfabric.programmer.device.Device`,
takes the data from the Segmenter class and programs the AWG.
Before writing the device class, let's consider a skeleton driver class,
mimicking a AWG driver class implementing its functions.

.. code-block:: python

    class ExampleDriver:
        def __init__(resource: str):
            self._awg = self._connect(resource)

        def _connect(resource: str): ...

        def set_memory_block(
            self,
            block_index: int, 
            analog_data: npt.NDArray[np.float64],
            digital_data: npt.NDArray[np.bool],
        ): ...

        def set_step(
            self,
            step_index: int,
            repeats: int,
            next_step: int = -1,
        ): ...

        def start(self): ...

        def wait_for_complete(self): ...

        def stop(self): ...

        def set_trigger_mode(self, external: bool): ...

Then we write the ``ExampleDevice`` class linking these AWG functions to
data from the Segmenter.

.. code-block:: python

    from qfabric.programmer.device import Device

    class ExampleDevice(Device):
        def __init__(segmenter: ExampleSegmenter, resource: str, principal_device: bool):
            super().__init__(segmenter, resource, principal_device)
            self._driver = ExampleDriver(resource)
            if self._principal_device:
                self.setup_software_trigger()
            else:
                # other devices are triggered by the principal device.
                self.setup_external_trigger()

        def program_memory(self, instructions: dict[str, list[ExampleSegment]]):
            """
            It needs to work with the first returned value in ExampleSegmenter.get_awg_memory_data.

            Adds each block in each segment into AWG memory.
            """
            self._segment_to_block_map: dict[int, list[int]] = {}
            block_counter = 0
            for segment_index, segment in enumerate(instructions["segments"]):
                block_indices = []
                for block in segment.blocks:
                    self._driver.set_memory_block(
                        block_counter, block.analog_data, block.digital_data
                    )
                    block_indices.append(block_counter)
                    block_counter += 1
                self._segment_to_block_map[segment_index] = block_indices

        def program_segment_steps(self, segment_indices_and_repeats: list[tuple[int, int]]):
            block_step_counter = 0
            for segment_step_index, (segment_index, segment_repeat) in enumerate(
                segment_indices_and_repeats
            ):
                block_indices_this_segment = self._segment_to_block_map[segment_index]
                for _ in range(segment_repeat):
                    for block_step_index, block_index in enumerate(block_indices_this_segment):
                        if (
                            (segment_step_index == len(segment_indices_and_repeats) - 1)  # last segment
                            and (block_step_index == len(block_indices_this_segment) - 1)  # last block
                        ):
                            next_block_step = -1
                        else:
                            next_block_step = block_step_counter + 1
                        self._driver.set_step(block_step_counter, 1, next_block_step)

        def start(self):
            self._driver.start()

        def wait_until_complete(self):
            self._driver.wait_until_complete()

        def stop(self):
            self._driver.stop()

        def setup_external_trigger(self):
            self.set_trigger_mode(True)

        def setup_software_trigger(self):
            self.set_trigger_mode(False)

Similar to the segmenter class above, this ``ExampleDevice`` class does not
optimize for everything (e.g. check previously loaded blocks for duplicates,
or combine multiple steps with the same block to be a single step with repeats).
However, it shows general steps to write a device class.

Following above, you can write a ``Segmenter`` and ``Device`` class for your device,
and then you can add them into the config file for use (See :doc:`../how_to/how_to_config`).