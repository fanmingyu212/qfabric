from qfabric.programmer.device import Device


class VirtualDevice(Device):
    """
    A virtual device that works with :class:`~qfabric.planner.segmenter.virtual.VirtualSegmenter`.
    """

    def program_memory(self, instructions):
        return

    def program_segment_steps(self, segment_indices_and_repeats):
        return

    def start(self):
        return

    def wait_until_complete(self):
        return

    def stop(self):
        return

    def setup_external_trigger(self):
        return

    def setup_software_trigger(self):
        return
