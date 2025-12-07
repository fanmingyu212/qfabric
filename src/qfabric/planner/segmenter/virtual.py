from qfabric.planner.segmenter import Segmenter


class VirtualSegmenter(Segmenter):
    """
    A virtual segmenter that accepts any number of digital and analog channels.

    It works with :class:`~qfabric.programmer.device.virtual.VirtualDevice`.
    It does not interface with any AWG device. However, it can be defined as an
    AWG in the config TOML file as a virtual AWG. It allows additional virtual
    channels to be visualized.
    """

    def set_steps(self, steps, sequence_to_steps_map):
        return super().set_steps(steps, sequence_to_steps_map)

    def get_awg_memory_data(self, sequence_indices):
        step_to_segment_map = {}
        for kk in range(len(self._device_steps)):
            step_to_segment_map[kk] = 0
        return None, step_to_segment_map, sequence_indices
