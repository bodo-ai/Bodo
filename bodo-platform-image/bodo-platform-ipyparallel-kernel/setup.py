import setuptools

with open("README.md") as f:
    readme = f.read()


setuptools.setup(
    name="bodo_platform_ipyparallel_kernel",
    version="2.0.0",
    description="A simple IPyParallel based wrapper around IPython Kernel for the Bodo Platform",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Bodo-inc/bodo-platform-ipyparallel-kernel",
    packages=setuptools.find_packages(),
    author="Bodo, Inc.",
    author_email="noreply@bodo.ai",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        'ipyparallel.engine_launchers': [
            'bodo = bodo_platform_ipyparallel_kernel:BodoPlatformMPIEngineSetLauncher',
        ],
    },
)
