"""A simple IPyParallel based wrapper around IPython Kernel"""
__version__ = "2.0.0"
from .kernel import IPyParallelKernel
from .launcher import BodoPlatformMPIEngineSetLauncher
