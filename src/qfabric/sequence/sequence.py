import numpy as np

from qfabric.sequence.step import EmptyStep, Step


class Sequence:
    """
    Pulse sequence for an experiment.
    """

    def __init__(self):
        self._steps: list[Step] = []
        self._repeats: list[int] = []

    def __eq__(self, other: "Sequence") -> bool:
        steps_equal = np.array_equal(self._steps, other._steps)
        repeats_equal = np.array_equal(self._repeats, other._repeats)
        return steps_equal and repeats_equal

    @property
    def nominal_duration(self) -> float:
        """
        Nominal duration of the entire pulse sequence in seconds.

        AWG may introduce a small duration change, due to segment length requirements
        or time to switch between segments.
        """
        timestamp = 0
        for kk in range(len(self._steps)):
            timestamp += self._steps[kk].duration * self._repeats[kk]
        return timestamp

    def add_step(self, step: Step, repeats: int = 1, delay_time_after_previous: float = 0):
        """
        Appends a step in the sequence.

        Args:
            step (Step): Step to append.
            repeats (int): Number of repeats for this step. Default 1.
            delay_time_after_previous (float):
                Optional delay time after the previous step. Default 0.
                This delay time, as well as duration of each step, may not be exact
                due to AWG limitations.
        """
        if delay_time_after_previous < 0:
            raise ValueError("Delay time after the previous step cannot be less than 0 s.")
        if delay_time_after_previous > 0:
            empty = EmptyStep(delay_time_after_previous)
            self._steps.append(empty)
            self._repeats.append(1)
        self._steps.append(step)
        self._repeats.append(repeats)

    def get_steps(self) -> list[Step]:
        """
        Gets all steps.

        The executed steps also include a start step and a stop step,
        which are not included in the return value of this function.
        The start and stop steps are added in the experiment manager before
        programming the AWG devices.

        Returns:
            list[Step]: Steps of this sequence.
        """
        return self._steps

    def get_repeats(self) -> list[int]:
        """
        Gets number of repeats of each step.

        Returns:
            list[int]: Number of repeats of each step of this sequence.
        """
        return self._repeats
