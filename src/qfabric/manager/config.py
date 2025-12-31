import tomllib
from typing import List

from pydantic import BaseModel


class SegmenterConfig(BaseModel, extra="allow"):
    """
    Config for the :class:`~qfabric.planner.segmenter.Segmenter` class.

    Additional arguments for custom :class:`~qfabric.planner.segmenter.Segmenter` are allowed.
    """

    analog_channels: list[int]
    digital_channels: list[int]


class DeviceConfig(BaseModel, extra="allow"):
    """
    Config for the :class:`~qfabric.programmer.device.Device` class.

    Additional arguments for custom :class:`~qfabric.programmer.device.Device` are allowed.
    """

    resource: str


class AWGConfig(BaseModel):
    """
    Config for a single AWG device.
    """

    name: str

    segmenter_module: str
    segmenter_class: str
    segmenter_config: SegmenterConfig
    device_module: str
    device_class: str
    device_config: DeviceConfig


class StartStepConfig(BaseModel):
    """
    Config for an optional start step.

    The start step is inserted at the begining of each sequence.
    """

    use: bool
    duration: float
    digital_channel_synchronize: int | None = None


class StopStepConfig(BaseModel):
    """
    Config for an optional stop step.

    The stop step is appened at the end of each sequence.
    """

    use: bool
    duration: float


class HardwareConfig(BaseModel):
    """
    Config for a system of AWGs.
    """

    version: int
    start_step: StartStepConfig
    stop_step: StopStepConfig
    awgs: List[AWGConfig]


def load_hardware_config(path: str) -> HardwareConfig:
    """
    Loads config file.

    Args:
        path (str): Path to the config file.

    Returns:
        HardwareConfig: Hardware config object.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = raw.get("config", {})
    data = {
        "version": config.get("version"),
        "start_step": raw.get("start_step"),
        "stop_step": raw.get("stop_step"),
        "awgs": raw.get("awgs", []),
    }

    analog_channels_defined: list[int] = []
    digital_channels_defined: list[int] = []
    for awg in data["awgs"]:
        analog_channels_defined.extend(awg["segmenter_config"]["analog_channels"])
        digital_channels_defined.extend(awg["segmenter_config"]["digital_channels"])
    if len(analog_channels_defined) != len(set(analog_channels_defined)):
        raise ValueError("Analog channels defined in the config file have duplicates.")
    if len(digital_channels_defined) != len(set(digital_channels_defined)):
        raise ValueError("Digital channels defined in the config file have duplicates.")
    if "digital_channel_synchronize" not in data["start_step"]:
        data["start_step"]["digital_channel_synchronize"] = None
    sync_channel = data["start_step"]["digital_channel_synchronize"]
    if sync_channel is not None and sync_channel not in digital_channels_defined:
        raise ValueError("Synchronization digital channel is not in the digital channels defined.")
    return HardwareConfig.model_validate(data)
