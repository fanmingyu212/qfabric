What is qFabric?
==================

qFabric is a package for quantum control using arbitrary waveform generators (AWGs).
It supports controlling multiple AWG instruments using a single interface,
with pulse sequence build tools designed for physicists.

qFabric aims to hide AWG technical details from general users,
so the users can define and visualize their experiment sequence
in a way that makes sense for physicists.

The AWGs currently supported by qFabric is listed in :doc:`awg_support`.
Adding support for a new AWG is straight-forward and modular though.
We encourage sharing your AWG implementation for qFabric in a repository,
and please feel free to submit a pull request to update the list.
