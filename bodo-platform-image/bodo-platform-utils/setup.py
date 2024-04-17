from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().split("\n")

setup(
    name='bodo_platform_utils',
    version='2.0.0',
    packages=find_packages(include=["bodo_platform_utils", "bodo_platform_utils.*"]),
    url='https://github.com/Bodo-inc/bodo-platform-utils',
    scripts=[],
    author='Bodo Inc',
    author_email='noreply@bodo.ai',
    description='This is package contains utility functions to retrieve data from Cloud Providers for platform',
    long_description=long_description,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)


