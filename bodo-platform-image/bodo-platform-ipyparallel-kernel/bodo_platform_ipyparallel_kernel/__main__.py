from ipykernel.kernelapp import IPKernelApp
from . import IPyParallelKernel

IPKernelApp.launch_instance(kernel_class=IPyParallelKernel)
