from setuptools import find_packages, setup

setup(
    name="bodoicebergconnector",
    version=0.1,
    description="Bodo Connector for Iceberg",
    long_description="Bodo Connector for Iceberg",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="data analytics cluster",
    url="https://bodo.ai",
    author="Bodo.ai",
    packages=find_packages(),
)
