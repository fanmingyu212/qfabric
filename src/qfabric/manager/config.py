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


class HardwareConfig(BaseModel):
    """
    Config for a system of AWGs.
    """

    version: int
    digital_channel_synchronize: int | None
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
        "digital_channel_synchronize": config.get("digital_channel_synchronize", None),
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
    if data["digital_channel_synchronize"] not in digital_channels_defined:
        raise ValueError("Synchronization digital channel is not in the digital channels defined.")
    return HardwareConfig.model_validate(data)
